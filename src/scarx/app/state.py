"""One SQLite transaction commits experience, full state and active pointer.

Snapshots contain the existing learner (including plasticity-v1), replay and
orchestration RNG. A failed update restores the original in-memory objects;
an interrupted process recovers the last SQLite-committed generation.
"""
from contextlib import contextmanager
import copy
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import uuid
import zipfile
import numpy as np
import psutil
from ..neural.core import Config
from ..learning.lifelong import LifelongLearner
from ..learning.replay import ReplayBuffer
from ..memory.store import EpisodeStore
from ..runtime.resources import ResourceMonitor
from .tasks import TASKS,normalize_sample,expected_label,encode,examples
from .locking import DirectoryLease


class Workbench:
    @property
    def pilots(self):
        with self.lock:
            if not hasattr(self,'_pilots'):
                from .pilots import PilotLab
                self._pilots=PilotLab(self)
            return self._pilots

    def __init__(self,directory,seed=17):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()
        self.closed=False
        self.lease=DirectoryLease(self.directory/'session.lock')
        try: self._initialize(seed)
        except BaseException:
            if hasattr(self,'monitor'): self.monitor.__exit__(None,None,None)
            if hasattr(self,'store'): self.store.close()
            self.lease.close()
            raise

    def _initialize(self,seed):
        self.store=EpisodeStore(self.directory/'scarx.sqlite',check_same_thread=False)
        self.db=self.store.db
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS mvp_checkpoints(id TEXT PRIMARY KEY,created TEXT NOT NULL,
            action TEXT NOT NULL,updates INTEGER NOT NULL,sha256 TEXT NOT NULL,payload BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS mvp_active(singleton INTEGER PRIMARY KEY CHECK(singleton=1),id TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS mvp_history(id INTEGER PRIMARY KEY,created TEXT NOT NULL,kind TEXT NOT NULL,
            checkpoint TEXT NOT NULL,details TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS mvp_samples(sample_hash TEXT PRIMARY KEY,task TEXT NOT NULL);
        ''')
        self.monitor=ResourceMonitor(interval=.5).__enter__()
        self.process=psutil.Process(); self.process.cpu_percent(None)
        self.started=time.monotonic(); self.limit_status={'enforced':False,'reason':'Chạy qua service/test; launcher báo giới hạn OS riêng.'}
        self.busy=False
        row=self.db.execute('SELECT id FROM mvp_active WHERE singleton=1').fetchone()
        if row:
            self.active=row[0]; self._unpack(self._payload(self.active))
        else:
            self.learner=LifelongLearner(Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed),'score_ema')
            self.replay=ReplayBuffer(128,12,8,'reservoir',seed+100)
            self.rng=np.random.default_rng(seed+200); self.epoch=0
            with self.db: self._commit('init',{'seed':seed,'trained':False})

    @staticmethod
    def _sample_hash(sample):
        return hashlib.sha256(json.dumps(sample,sort_keys=True).encode()).hexdigest()

    def _pack(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            z.writestr(zipfile.ZipInfo('learner.zip'),self.learner.dumps()); z.writestr(zipfile.ZipInfo('replay.npz'),self.replay.dumps())
            z.writestr(zipfile.ZipInfo('session.json'),json.dumps({'format':'scarx-mvp-v1','rng':self.rng.bit_generator.state,'epoch':self.epoch}))
        result=stream.getvalue()
        if len(result)>2_000_000: raise MemoryError('Checkpoint MVP vượt giới hạn 2 MB.')
        return result

    def _unpack(self,payload):
        if len(payload)>2_000_000: raise ValueError('Checkpoint quá lớn.')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'learner.zip','replay.npz','session.json'} or len(z.namelist())!=3 or sum(i.file_size for i in z.infolist())>2_000_000:
                raise ValueError('Checkpoint MVP không hợp lệ.')
            meta=json.loads(z.read('session.json'))
            if meta.get('format')!='scarx-mvp-v1': raise ValueError('Sai định dạng session.')
            learner_payload=z.read('learner.zip')
            # Reject incompatible dimensions before allocating a model from metadata.
            with zipfile.ZipFile(io.BytesIO(learner_payload)) as inner:
                if sum(i.file_size for i in inner.infolist())>2_000_000: raise ValueError('Learner quá lớn.')
                model_payload=inner.read('model.npz')
            with zipfile.ZipFile(io.BytesIO(model_payload)) as model_zip:
                if model_zip.getinfo('metadata.npy').file_size>65536 or sum(i.file_size for i in model_zip.infolist())>2_000_000: raise ValueError('Model quá lớn.')
            with np.load(io.BytesIO(model_payload),allow_pickle=False) as arrays:
                config=json.loads(str(arrays['metadata']))['config']
            if tuple(config.get(k) for k in ('inputs','hidden','outputs','fan_in','plastic_rows'))!=(8,48,2,8,12):
                raise ValueError('Kích thước checkpoint không thuộc MVP V0.1.')
            learner=LifelongLearner.loads(learner_payload)
            replay=ReplayBuffer.loads(z.read('replay.npz'))
        c=learner.model.config
        if (c.inputs,c.hidden,c.outputs,c.fan_in,c.plastic_rows)!=(8,48,2,8,12) or replay.x.shape!=(128,12,8):
            raise ValueError('Kích thước checkpoint không thuộc MVP V0.1.')
        rng=np.random.default_rng(); rng.bit_generator.state=meta['rng']
        if type(meta.get('epoch')) is not int or meta['epoch']<0: raise ValueError('Session counter không hợp lệ.')
        self.learner,self.replay,self.rng,self.epoch=learner,replay,rng,meta['epoch']

    def _payload(self,checkpoint):
        row=self.db.execute('SELECT sha256,payload FROM mvp_checkpoints WHERE id=?',(checkpoint,)).fetchone()
        if row is None: raise ValueError('Không tìm thấy checkpoint.')
        if hashlib.sha256(row[1]).hexdigest()!=row[0]: raise ValueError('Checksum checkpoint sai; không tự tạo model trắng.')
        return row[1]

    def _event(self,kind,details,checkpoint=None):
        self.db.execute('INSERT INTO mvp_history(created,kind,checkpoint,details) VALUES(?,?,?,?)',
                        (datetime.now(timezone.utc).isoformat(),kind,checkpoint or self.active,json.dumps(details,ensure_ascii=False,allow_nan=False)))

    def _commit(self,action,details):
        if self.db.execute('SELECT count(*) FROM mvp_checkpoints').fetchone()[0]>=2000:
            raise ValueError('Đã có 2.000 checkpoint; hãy sao lưu/quản lý dữ liệu trước khi học thêm. Không tự xóa lịch sử.')
        payload=self._pack(); checkpoint=uuid.uuid4().hex
        self.db.execute('INSERT INTO mvp_checkpoints VALUES(?,?,?,?,?,?)',
            (checkpoint,datetime.now(timezone.utc).isoformat(),action,self.learner.model.updates,hashlib.sha256(payload).hexdigest(),payload))
        self.db.execute('INSERT OR REPLACE INTO mvp_active VALUES(1,?)',(checkpoint,))
        self._event(action,details,checkpoint)
        self.active=checkpoint
        return checkpoint

    @contextmanager
    def transaction(self):
        self.monitor.check()
        if max(self.monitor.peak_rss,self.monitor.peak_commit)>1_700_000_000:
            raise MemoryError('Không đủ RAM dự phòng cho update/rollback.')
        original=(self.learner,self.replay,self.rng,self.epoch,self.active)
        self.learner=copy.deepcopy(self.learner); self.replay=copy.deepcopy(self.replay); self.rng=copy.deepcopy(self.rng)
        self.busy=True
        try:
            with self.db: yield
        except BaseException:
            self.learner,self.replay,self.rng,self.epoch,self.active=original
            raise
        finally:
            self.busy=False

    def _probabilities(self,samples,seeds=(73001,73003,73007,73009)):
        x=np.asarray([encode(s) for s in samples],np.float32)
        probabilities=[]
        for seed in seeds:
            f=self.learner.model.features_many(x,seed)
            logits=f@self.learner.model.w_out.T+self.learner.model.bias
            logits-=logits.max(1,keepdims=True)
            p=np.exp(logits); probabilities.append(p/p.sum(1,keepdims=True))
        return np.mean(probabilities,axis=0)

    def predict(self,data,automatic=False):
        sample=normalize_sample(data)
        with self.lock:
            p=self._probabilities([sample])[0]
            passes=4
            if automatic and float(p.max())<.8:
                extra=self._probabilities([sample],seeds=tuple(range(73011,73035,2)))[0]
                p=(p*4+extra*12)/16; passes=16
            model=copy.deepcopy(self.learner.model); model.rng=np.random.default_rng(73001); model.reset_state()
            spikes=[]
            for x in encode(sample): model.step(x,track=False); spikes.append(model.spikes.astype(int).tolist())
            result={'prediction':int(np.argmax(p)),'probabilities':p.tolist(),'expected':expected_label(sample),
                    'sample':sample,'spikes':spikes,'checkpoint':self.active,'retrieval_used':False,
                    'prediction_source':'PlasticCore ALIF → trace → neural readout; mean of 4 fixed stochastic passes',
                    'expected_source':'Quy tắc của bài có cấu trúc; chỉ kiểm nhãn/chấm điểm, không thay dự đoán.'}
            result['compute']={'automatic':bool(automatic),'passes':passes,'learned_reasoning':False,
                               'policy':'4 passes; extend to 16 if maximum softmax probability < 0.8. Stochastic averaging, not chain-of-thought.'}
            with self.db:
                self.store.observe(json.dumps({'sample':sample,'prediction':result['prediction']},ensure_ascii=False),
                    'scarx-mvp:neural-prediction',kind='generated',verified=False,commit=False)
                self._event('predict',{'sample':sample,'prediction':result['prediction']})
            return result

    def _evaluate(self):
        results={}
        for index,task in enumerate(TASKS):
            samples=examples(task,128,np.random.default_rng(800001+index))
            p=self._probabilities(samples); correct=np.argmax(p,axis=1)==[expected_label(s) for s in samples]
            # Audit exact sample membership, not semantic novelty.
            seen=sum(self.db.execute('SELECT 1 FROM mvp_samples WHERE sample_hash=?',(self._sample_hash(s),)).fetchone() is not None for s in samples)
            results[task]={'accuracy':float(correct.mean()),'correct':int(correct.sum()),'count':len(samples),'seen_exact':seen,
                           'scope':'Biến thể cường độ/nhiễu mới của 4 tín hiệu; không phải khái niệm hay kỹ năng ngôn ngữ mới.'}
        return results

    def evaluate(self):
        with self.lock:
            result={'scores':self._evaluate(),'checkpoint':self.active,'retrieval_used':False,'weights_updated':False}
            with self.db: self._event('evaluate',result)
            return result

    def teach(self,data):
        manual='signal' in data
        task=data.get('task')
        if task not in TASKS: raise ValueError('Chọn tác vụ A hoặc B.')
        if manual:
            sample=normalize_sample(data); target=data.get('label')
            if type(target) is not int or target not in (0,1): raise ValueError('Chọn nhãn đúng 0 hoặc 1.')
            if data.get('confirmation') is not True: raise ValueError('Cần xác nhận nhãn trước khi dạy.')
            if target!=expected_label(sample): raise ValueError('Nhãn mâu thuẫn quy tắc tác vụ; trải nghiệm này chưa được dùng để học.')
            repeats=data.get('repeats',32)
            if type(repeats) is not int or not 1<=repeats<=128: raise ValueError('Số lần lặp từ 1 đến 128.')
        else:
            count=data.get('count',64); epochs=data.get('epochs',4)
            if type(epochs) is not int or not 1<=epochs<=8: raise ValueError('Số vòng học từ 1 đến 8.')
            if type(count) is not int or not 4<=count<=128 or count%4 or count*epochs>1024: raise ValueError('Batch học tối đa 128 ví dụ, tổng 1.024 update.')
        with self.lock:
            before=self._evaluate(); initial={n:getattr(self.learner.model,n).copy() for n in ('w_in','w_rec','w_out','bias')}
            initial_updates=self.learner.model.updates; start=time.monotonic()
            with self.transaction():
                samples=[sample] if manual else examples(task,count,self.rng)
                schedule=samples*repeats if manual else samples*epochs
                if data.get('replay',True) is not False:
                    old=self.replay.sample(min(32,self.replay.size))
                else: old=[]
                training=[(encode(s),expected_label(s)) for s in schedule]+old
                self.rng.shuffle(training)
                for i,(x,y) in enumerate(training):
                    self.learner.update(x,int(y))
                    if i%32==0:
                        self.monitor.check()
                        if time.monotonic()-start>15: raise TimeoutError('Đợt học vượt 15 giây; đã rollback.')
                if any(not np.isfinite(a).all() for a in self.learner.model.arrays().values()): raise ValueError('State không hữu hạn; đã rollback.')
                for s in samples:
                    self.replay.add(encode(s),expected_label(s),int(s['task']=='B'))
                    self.store.observe(json.dumps({'sample':s,'label':expected_label(s),'evidence':'builtin-rule-v1','operator_confirmed':manual},ensure_ascii=False),
                        'scarx-mvp:builtin-rule-v1',kind='correction',verified=True,commit=False)
                    self.db.execute('INSERT OR IGNORE INTO mvp_samples VALUES(?,?)',(self._sample_hash(s),task))
                after=self._evaluate(); changes={n:float(np.linalg.norm(getattr(self.learner.model,n)-a)) for n,a in initial.items()}
                self.epoch+=1
                result={'task':task,'before':before,'after':after,'weight_changes':changes,
                        'updates':self.learner.model.updates-initial_updates,'new_examples':len(samples),'replayed':len(old),
                        'seconds':time.monotonic()-start,'label_evidence':'Quy tắc builtin-rule-v1 kiểm tra độc lập; cờ verified không được dùng làm bằng chứng.',
                        'retrieval_used':False,'note':'Thay đổi hidden weights chưa chứng minh hidden representation tốt hơn; readout cũng được học.'}
                result['checkpoint']=self._commit('teach',result)
            return result

    def checkpoint(self):
        with self.lock,self.transaction():
            return {'checkpoint':self._commit('checkpoint',{'manual':True}),'updates':self.learner.model.updates}

    def restore(self,checkpoint):
        if not isinstance(checkpoint,str) or len(checkpoint)!=32: raise ValueError('Checkpoint ID không hợp lệ.')
        with self.lock:
            payload=self._payload(checkpoint)
            with self.transaction():
                self._unpack(payload)
                new=self._commit('restore',{'restored_from':checkpoint})
            return {'checkpoint':new,'restored_from':checkpoint,'updates':self.learner.model.updates,'scores':self._evaluate()}

    def import_checkpoint(self,payload):
        with self.lock,self.transaction():
            self._unpack(payload)
            checkpoint=self._commit('import',{'bundle_sha256':hashlib.sha256(payload).hexdigest(),'note':'Imported neural/replay state; prior active generation remains recoverable.'})
            return {'checkpoint':checkpoint,'updates':self.learner.model.updates}

    def history(self):
        with self.lock:
            rows=self.db.execute('SELECT id,created,kind,checkpoint,details FROM mvp_history ORDER BY id DESC LIMIT 40').fetchall()
            return [{'id':r[0],'created':r[1],'kind':r[2],'checkpoint':r[3],'details':json.loads(r[4])} for r in rows]

    def checkpoints(self):
        with self.lock:
            rows=self.db.execute('SELECT id,created,action,updates,sha256,length(payload) FROM mvp_checkpoints ORDER BY rowid DESC LIMIT 100').fetchall()
            return [{'id':r[0],'created':r[1],'action':r[2],'updates':r[3],'sha256':r[4],'bytes':r[5],'active':r[0]==self.active} for r in rows]

    def memory(self,query=''):
        with self.lock:
            if query: return self.store.search(query[:200],limit=20)
            self.db.row_factory=None
            return [{'id':r[0],'text':r[1],'source':r[2],'verified':bool(r[3]),'kind':r[4],'created':r[5]} for r in self.db.execute(
                'SELECT id,substr(text,1,4096),source,verified,kind,created FROM episodes ORDER BY id DESC LIMIT 40')]

    def resources(self):
        self.monitor.sample(); info=self.process.memory_info()
        return {'rss_bytes':info.rss,'commit_bytes':getattr(info,'private',info.vms),'cpu_percent':self.process.cpu_percent(None),
                'peak_rss_bytes':self.monitor.peak_rss,'peak_commit_bytes':self.monitor.peak_commit,'limit_bytes':2_000_000_000,
                'os_limit':self.limit_status,'cpu_percent_note':'CPU tiến trình: 100% tương đương một logical core.',
                'uptime_seconds':time.monotonic()-self.started}

    def status(self):
        with self.lock:
            m=self.learner.model
            return {'version':'0.1.0','lab_version':'0.2-experimental','checkpoint':self.active,'updates':m.updates,'teaching_batches':self.epoch,
                    'neurons':m.config.hidden,'synapses':int(m.w_in.size+m.w_rec.size+m.w_out.size),
                    'model_array_bytes':m.memory_bytes(),'replay_size':self.replay.size,'data_directory':str(self.directory.resolve()),
                    'mode':self.learner.mode,'tasks':TASKS,'resources':self.resources(),
                    'pilot_cores_loaded':list(self._pilots.loaded) if hasattr(self,'_pilots') else [],
                    'capabilities':'Tác vụ tín hiệu có cấu trúc. Chưa hiểu ngôn ngữ tự nhiên; chưa giải quyết continual learning.'}

    def close(self):
        with self.lock:
            if not self.closed:
                self.monitor.__exit__(None,None,None); self.store.close(); self.lease.close(); self.closed=True
