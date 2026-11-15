"""P1A-only frozen CUDA policy and explicit action/outcome binding. Not serving."""
import copy
import hashlib
import io
import json
import math
from pathlib import PurePosixPath
import sqlite3
import time
import uuid
import zipfile

import numpy as np
from scarx.hybrid.learner import LocalALIF,features
from .hybrid_incident_encoder import observation_sequence
from .recovery_policy import ACTION_ORDER
from .outcome_contract import canonical,digest

FORMAT='scarx-frozen-neural-outcome-bridge-v1'
ADAPTER='scarx-task-observation-bridge-v1'
MACROS='scarx-owned-readonly-macros-v1'
DYNAMIC={'core.voltage','core.adaptation','core.spikes','core.pre','core.post','core.features'}


def validate_task(task):
    if not isinstance(task,dict) or set(task)!={'instruction','request'}:
        raise ValueError('Only a public instruction/request may enter the adapter')
    if not isinstance(task['instruction'],str) or not 2<=len(task['instruction'])<=1000:
        raise ValueError('Invalid bounded public instruction')
    request=task['request']
    if not isinstance(request,dict) or set(request)!={'tool','arguments'} or not isinstance(request['arguments'],dict):
        raise ValueError('Invalid original request')
    if request['tool']=='read_file':
        args=request['arguments'];path=args.get('path')
        if (set(args)!={'path'} or not isinstance(path,str) or not 1<=len(path)<=80
                or '\\' in path or ':' in path or PurePosixPath(path).is_absolute()
                or any(p in ('','..','.') for p in path.split('/'))):
            raise ValueError('Unbound or unsafe public source')
    elif request['tool'] in ('poll','create_alarm'):
        if request['arguments']:raise ValueError('Unsupported request arguments')
    else:raise ValueError('Unsupported public request')
    return copy.deepcopy(task)


def encode_task_observations(task,observations,cutoff):
    task=validate_task(task)
    if type(cutoff) not in (int,float) or not math.isfinite(cutoff) or cutoff<=0:
        raise ValueError('Invalid cutoff')
    if not isinstance(observations,(list,tuple)) or len(observations)>8:
        raise ValueError('Invalid bounded observation list')
    # Put public bindings before potentially long prose. Observation blocks
    # have their own text/numeric budget; private contracts never enter here.
    chunks=[features(canonical(task['request']).decode()+' '+task['instruction'])]
    if observations:
        chunks.append(observation_sequence('Observed event',observations,decision_epoch=cutoff))
    result=np.concatenate(chunks).astype(np.float32)
    if result.shape!=(8*(1+len(observations)),64):raise ValueError('Adapter shape mismatch')
    return result


def frozen_fingerprint(model):
    arrays={k:hashlib.sha256(model.host(v).tobytes()).hexdigest()
            for k,v in model.arrays().items() if k not in DYNAMIC}
    with np.load(io.BytesIO(model.dumps()),allow_pickle=False) as data:
        meta=json.loads(data['metadata'].tobytes())
    meta.pop('ticks');meta.pop('controller')
    return digest({'arrays':arrays,'metadata':meta})


class BindingJournal:
    """Immutable original decisions/receipts; never rewound by model rollback."""
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        self.db.executescript('''CREATE TABLE IF NOT EXISTS metadata(k TEXT PRIMARY KEY,v TEXT);
            CREATE TABLE IF NOT EXISTS contexts(task_id TEXT PRIMARY KEY,scope TEXT,task TEXT,active INTEGER);
            CREATE TABLE IF NOT EXISTS decisions(id TEXT PRIMARY KEY,payload TEXT);
            CREATE TABLE IF NOT EXISTS receipts(id TEXT PRIMARY KEY,payload TEXT);''')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('journal_id',?)",(uuid.uuid4().hex,));self.db.commit()
        self.identity=self.db.execute("SELECT v FROM metadata WHERE k='journal_id'").fetchone()[0]

    def register_context(self,task_id,scope,task):
        task=validate_task(task)
        if not isinstance(task_id,str) or not task_id or not isinstance(scope,str) or not scope:
            raise ValueError('Missing external context binding')
        self.db.execute('INSERT INTO contexts VALUES(?,?,?,1)',(task_id,scope,canonical(task).decode()));self.db.commit()

    def context(self,task_id,scope,task,*,allow_revoked=False):
        row=self.db.execute('SELECT scope,task,active FROM contexts WHERE task_id=?',(task_id,)).fetchone()
        if not row or row[:2]!=(scope,canonical(validate_task(task)).decode()):raise ValueError('Foreign task/scope')
        if not row[2] and not allow_revoked:raise PermissionError('Scope lease revoked')
        return bool(row[2])

    def revoke(self,task_id):
        self.db.execute('UPDATE contexts SET active=0 WHERE task_id=?',(task_id,));self.db.commit()

    def pin_frozen_model(self,seed,fingerprint):
        key='frozen:'+str(seed)
        row=self.db.execute('SELECT v FROM metadata WHERE k=?',(key,)).fetchone()
        if row and row[0]!=fingerprint:raise ValueError('Frozen model differs from pinned seed identity')
        self.db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)',(key,fingerprint));self.db.commit()

    def record_decision(self,decision):
        self.context(decision['task_id'],decision['scope'],decision['public_task'])
        self.db.execute('INSERT INTO decisions VALUES(?,?)',(decision['id'],canonical(decision).decode()));self.db.commit()

    def decision(self,identifier):
        row=self.db.execute('SELECT payload FROM decisions WHERE id=?',(identifier,)).fetchone()
        if not row:raise ValueError('Unknown decision')
        return json.loads(row[0])

    def check_decision(self,decision):
        if canonical(self.decision(decision['id']))!=canonical(decision):raise ValueError('Decision was altered')

    def record_receipt(self,decision,receipt):
        self.check_decision(decision)
        fields={'decision_id':decision['id'],'task_id':decision['task_id'],'scope':decision['scope'],
                'proposed_action':decision['proposed_action'],'adapter':ADAPTER,'macros':MACROS}
        if any(receipt.get(k)!=v for k,v in fields.items()):raise ValueError('Receipt binding mismatch')
        if receipt.get('learning_authorized') is not False:raise ValueError('P1A cannot authorize learning')
        self.db.execute('INSERT INTO receipts VALUES(?,?)',(decision['id'],canonical(receipt).decode()));self.db.commit()

    def validate_receipt(self,receipt,expected_decision_id):
        if receipt.get('decision_id')!=expected_decision_id:raise ValueError('Receipt belongs to a different expected decision')
        decision=self.decision(receipt['decision_id'])
        row=self.db.execute('SELECT payload FROM receipts WHERE id=?',(decision['id'],)).fetchone()
        if not row or row[0]!=canonical(receipt).decode():raise ValueError('Receipt differs from the authoritative journal')
        return decision

    def has_receipt(self,identifier):
        return self.db.execute('SELECT 1 FROM receipts WHERE id=?',(identifier,)).fetchone() is not None

    def close(self):self.db.close()


class FrozenBridgePolicy:
    def __init__(self,seed,journal,charge,save_blob,save_arrays):
        self.model=LocalALIF(seed,mode='frozen').to_device('gpu',0)
        self.journal=journal;self.charge=charge;self.save_blob=save_blob;self.save_arrays=save_arrays
        self.pending={};self.initial=frozen_fingerprint(self.model)
        journal.pin_frozen_model(seed,self.initial)

    def assert_frozen(self):
        if (self.model.device!='gpu' or self.model.gpu_device!=0 or self.model.mode!='frozen'
                or self.model.updates or self.model.hidden_updates or self.model.replay
                or frozen_fingerprint(self.model)!=self.initial):raise ValueError('Frozen/GPU invariant violated')

    def propose(self,task_id,scope,task,observations,cutoff,allowed):
        task=validate_task(task)
        if (not isinstance(allowed,(tuple,list)) or len(allowed)<2 or len(set(allowed))!=len(allowed)
                or any(a not in ACTION_ORDER for a in allowed)):raise ValueError('Need explicit multi-choice capability mask')
        sequence=encode_task_observations(task,observations,cutoff)
        self.journal.context(task_id,scope,task)
        self.assert_frozen();before=self.save_blob(self.model.dumps())
        identifier=uuid.uuid4().hex
        attempt=self.charge('forward',self.model.seed,'frozen_proposal:'+identifier)
        probabilities,f,raw,scale,indices=self.model.forward(sequence,track=False)
        self.model.xp.cuda.get_current_stream().synchronize()
        self.assert_frozen();after=self.save_blob(self.model.dumps())
        choice=max((ACTION_ORDER.index(a) for a in allowed),key=lambda i:float(probabilities[i]))
        measurement=self.save_arrays(identifier,{'sequence':sequence,'probabilities':probabilities,
            'features':f,'raw':raw,'scale':scale,'indices':indices})
        decision={'id':identifier,'task_id':task_id,'scope':scope,'public_task':task,
            'observations':copy.deepcopy(observations),'cutoff':cutoff,'created':time.time(),
            'adapter':ADAPTER,'macros':MACROS,'action_order':list(ACTION_ORDER),'allowed':list(allowed),
            'proposed_action':ACTION_ORDER[choice],'selector':'neural_policy','mode':'frozen',
            'seed':self.model.seed,'device':'gpu:0','input_sha256':hashlib.sha256(sequence.tobytes()).hexdigest(),
            'probabilities':dict(zip(ACTION_ORDER,map(float,probabilities))),
            'before_model_sha256':before,'after_model_sha256':after,'measurement':measurement,
            'forward_attempt':attempt['attempt'],
            'learning_authorized':False}
        self.journal.record_decision(decision);self.pending[identifier]=copy.deepcopy(decision)
        return copy.deepcopy(decision)

    def settle(self,receipt,expected_decision_id):
        self.journal.validate_receipt(receipt,expected_decision_id);self.pending.pop(receipt['decision_id'],None)

    def dumps(self):
        self.assert_frozen();payload=self.model.dumps()
        meta={'format':FORMAT,'adapter':ADAPTER,'macros':MACROS,'actions':list(ACTION_ORDER),
            'seed':self.model.seed,'mode':'frozen','model_sha256':hashlib.sha256(payload).hexdigest(),
            'frozen_fingerprint':self.initial,'journal_id':self.journal.identity,'pending':self.pending}
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as z:
            for name,raw in [('metadata.json',canonical(meta)),('local-alif.npz',payload)]:
                z.writestr(zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0)),raw)
        return stream.getvalue()

    @classmethod
    def loads(cls,raw,journal,charge,save_blob,save_arrays):
        if not isinstance(raw,bytes) or len(raw)>16_000_000:raise ValueError('Snapshot bound')
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                infos=z.infolist()
                if (len(infos)!=2 or {i.filename for i in infos}!={'metadata.json','local-alif.npz'}
                        or sum(i.file_size for i in infos)>16_000_000
                        or any(i.compress_type!=zipfile.ZIP_STORED for i in infos)
                        or z.getinfo('metadata.json').file_size>2_000_000):raise ValueError('Snapshot members')
                meta=json.loads(z.read('metadata.json'));payload=z.read('local-alif.npz')
        except (zipfile.BadZipFile,KeyError,UnicodeDecodeError) as exc:raise ValueError('Snapshot archive') from exc
        fields={'format','adapter','macros','actions','seed','mode','model_sha256','frozen_fingerprint','journal_id','pending'}
        if (set(meta)!=fields or meta['format']!=FORMAT or meta['adapter']!=ADAPTER or meta['macros']!=MACROS
                or meta['actions']!=list(ACTION_ORDER) or meta['mode']!='frozen'
                or type(meta['seed']) is not int or meta['seed'] not in (260113,260137)
                or meta['journal_id']!=journal.identity or not isinstance(meta['pending'],dict)
                or len(meta['pending'])>16 or hashlib.sha256(payload).hexdigest()!=meta['model_sha256']):
            raise ValueError('Snapshot interface/identity mismatch')
        for identifier,decision in meta['pending'].items():
            if identifier!=decision.get('id'):raise ValueError('Pending decision identity')
            journal.check_decision(decision)
        # Data-only CPU deserialization before placing validated arrays on GPU.
        model,controller=LocalALIF.loads(payload,device='cpu')
        if controller or model.mode!='frozen' or model.seed!=meta['seed']:
            raise ValueError('Inner model identity mismatch')
        if frozen_fingerprint(model)!=meta['frozen_fingerprint']:raise ValueError('Frozen fingerprint mismatch')
        journal.pin_frozen_model(model.seed,meta['frozen_fingerprint'])
        instance=cls.__new__(cls);instance.model=model.to_device('gpu',0);instance.journal=journal
        instance.charge=charge;instance.save_blob=save_blob;instance.save_arrays=save_arrays
        instance.pending=meta['pending'];instance.initial=meta['frozen_fingerprint'];instance.assert_frozen()
        return instance
