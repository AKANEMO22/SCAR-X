"""P0-only output-local protection. Never imported by serving/neural/learning.

Same G1 ALIF/hidden rule; float64 counterfactual output arithmetic, no reverse
neural traversal/autograd. Registry labels are test oracles, not fact checking.
"""
import copy,hashlib,io,json,os,tempfile,zipfile
from pathlib import Path
import numpy as np
from ..neural.unified import UnifiedCore
from ..neural.language import tokenize
from ..runtime.atomic_file import replace_with_retry

P0_FIXTURES={f'p0-{word}':{'prompt':f'P0 fixture {word}','answer':word+'.'} for word in ('alpha','beta','gamma')}
MAX_PAYLOAD=4_000_000


def _lse(values):
    values=np.asarray(values,np.float64)
    if not values.size:return -np.inf
    maximum=float(np.max(values))
    return maximum+float(np.log(np.exp(values-maximum).sum()))


def counterfactual_benefit(weight,features,bias,mask,target):
    """Signed L(remove contribution)-L(original), including masked classes.

    log(1-p) comes from other logits directly, so a probability rounded to1
    cannot erase the competing tail. No exp of an unbounded positive argument.
    """
    w=np.asarray(weight,np.float64);f=np.asarray(features,np.float64);b=np.asarray(bias,np.float64);mask=np.asarray(mask)
    if w.ndim!=2 or not 1<=w.shape[0]<=266 or not 1<=w.shape[1]<=48 or f.shape!=(w.shape[1],) or b.shape!=(w.shape[0],) or mask.dtype!=np.bool_ or mask.shape!=b.shape or not mask.any() or type(target) is not int or not 0<=target<len(b) or not mask[target]:raise ValueError('Bounded output arithmetic / valid masked target required')
    if not all(np.isfinite(a).all() for a in (w,f,b)):raise ValueError('Nonfinite readout input')
    if np.max(np.abs(w))>20 or np.max(np.abs(b))>1000 or np.max(np.abs(f))>1:raise ValueError('P0 arithmetic bounds')
    ids=np.flatnonzero(mask);z=w[ids]@f+b[ids];normalizer=_lse(z);out=np.zeros_like(w)
    if len(ids)==1:return out
    for index,k in enumerate(ids):
        a=w[k]*f;log_other=_lse(np.delete(z,index))-normalizer;log_p=z[index]-normalizer
        out[k]=np.logaddexp(log_other,log_p-a)+(a if k==target else 0.)
    if not np.isfinite(out).all():raise ValueError('Nonfinite counterfactual result')
    return out


class AppliedBudgetError(ValueError):
    def __init__(self,radius,norms):
        super().__init__('Float32 applied deltas do not satisfy preregistered matched norm')
        self.radius=float(radius);self.norms=[float(n) for n in norms]


def matched_candidates(weights,candidates,weight_limit=20.,relative_tolerance=1e-5):
    """Pure calculation: return ALL valid candidates or reject without mutation.

    Candidates are already box-clipped feasible endpoints. Scaling down follows
    their convex rays; actual post-cast deltas still need explicit verification.
    """
    if len(weights)!=len(candidates) or not 2<=len(weights)<=7 or not np.isfinite(weight_limit) or not 0<weight_limit<=20 or relative_tolerance!=1e-5:raise ValueError('Fixed matched group/bounds/tolerance required')
    pairs=[]
    for w,c in zip(weights,candidates):
        if not isinstance(w,np.ndarray) or not isinstance(c,np.ndarray) or w.dtype!=np.float32 or c.dtype!=np.float32 or w.shape!=c.shape or w.ndim!=2 or not 1<=w.shape[0]<=266 or not 1<=w.shape[1]<=48 or not np.isfinite(w).all() or not np.isfinite(c).all() or np.max(np.abs(w))>weight_limit or np.max(np.abs(c))>weight_limit:raise ValueError('Finite same-shape float32 feasible proposals required')
        pairs.append((w,c.astype(np.float64)-w.astype(np.float64)))
    norms=[float(np.linalg.norm(d)) for _,d in pairs];radius=.5*min(norms)
    result=[w.copy() if not radius else (w.astype(np.float64)+d*(radius/n)).astype(np.float32) for (w,d),n in zip(pairs,norms)]
    actual=[float(np.linalg.norm(c.astype(np.float64)-w.astype(np.float64))) for w,c in zip(weights,result)]
    if radius and max(abs(n-radius)/radius for n in actual)>relative_tolerance:raise AppliedBudgetError(radius,actual)
    return result,{'radius':radius,'actual_norms':actual,'zero_radius':radius==0.,'valid':True,'relative_tolerance':relative_tolerance}


def fixture_hash(identifier):
    return hashlib.sha256(json.dumps(P0_FIXTURES[identifier],sort_keys=True).encode()).hexdigest()


class UsefulnessPrototype(UnifiedCore):
    """Bounded fresh48ALIF P0 fixture harness; no chat or pilot endpoint."""
    decay=.995

    def __init__(self,seed=42201,beta=4.):
        if type(seed) is not int or not 0<=seed<2**30 or type(beta) not in (float,int) or beta not in (0.,4.):raise ValueError('Only fixed P0 seed/beta allowed')
        vocabulary=['<eos>','<bos>']+sorted({t for row in P0_FIXTURES.values() for t in tokenize(row['answer'])})
        super().__init__(vocabulary,seed=seed)
        self.beta=float(beta);self.omega=np.zeros_like(self.core.w_out)
        self.fixture_rng=np.random.default_rng(seed+42200000);self.signal_rng=np.random.default_rng(seed+42210000)
        self.controller={'pending':None,'segments':0,'endpoints':0,'replay_cursor':0,'last_loss':None}
        self.replay=[];self._endpoint_ready=False

    def train_pair(self,*args,**kwargs):raise ValueError('P0 requires registry oracle, not a verified flag or arbitrary teaching')

    def select_capacity(self,requested):
        if requested!=48:raise ValueError('Fixed48ALIF P0; no model growth')

    def expand_vocabulary(self,*args,**kwargs):raise ValueError('Fixed P0 vocabulary')

    def image(self,pixels,train_label=None,mode='full'):
        if train_label is not None:raise ValueError('No image teaching in P0 fixture harness')
        return super().image(pixels)

    def draw_fixture(self):return tuple(P0_FIXTURES)[int(self.fixture_rng.integers(len(P0_FIXTURES)))]

    def permuted_g(self):
        g=1/(1+self.beta*self.omega)
        return g.ravel()[self.signal_rng.permutation(g.size)].reshape(g.shape)

    def begin_fixture(self,identifier):
        if not isinstance(identifier,str) or identifier not in P0_FIXTURES or self.controller['pending'] is not None:raise ValueError('Known source fixture / no pending segment required')
        before=self.dumps()
        try:
            self.boundary();c=self.core;c.elig_in.fill(0);c.elig_rec.fill(0);c.pre.fill(0);c.post.fill(0)
            start=(c.updates*c.config.plastic_rows)%self.active_count;c.rows[:]=(np.arange(12)+start)%self.active_count
            self.controller['pending']={'fixture':identifier,'source_sha256':fixture_hash(identifier),'position':0,'previous':1,'loss_sum':0.}
        except BaseException:self.rollback(before);raise

    def learn(self,target,modality,position=0,mode='full'):
        pending=self.controller['pending']
        if not self._endpoint_ready or pending is None or modality!='language' or mode!='full' or position!=pending['position']:raise ValueError('Only the current verified P0 endpoint may update')
        fixture=P0_FIXTURES[pending['fixture']];targets=[self.ids[t] for t in tokenize(fixture['answer'])]+[0]
        if target!=targets[position] or pending['source_sha256']!=fixture_hash(pending['fixture']):raise ValueError('P0 label/provenance mismatch')
        c=self.core;p=self.probabilities(modality,position);old=c.w_out.copy();error=-p.copy();error[target]+=1
        # Use old importance for this update; current correct prediction can
        # influence only subsequent updates. Output error remains baseline G1.
        delta=c.config.readout_lr*error[:,None]*c.features[None,:]
        g=1/(1+self.beta*self.omega)
        candidate=old+np.clip(delta*g,-c.config.update_clip,c.config.update_clip)
        np.clip(candidate,-c.config.weight_clip,c.config.weight_clip,out=candidate)
        new_omega=self.decay*self.omega
        if int(np.argmax(p))==target:
            # Active classes can underflow to probability0 in float32; the
            # output layout, never p>0, defines the counterfactual softmax set.
            mask=np.zeros(c.config.outputs,bool);mask[:len(self.vocabulary)]=True;mask[1]=False
            if position==0:
                mask[0]=False
                for index,token in enumerate(self.vocabulary[2:],2):
                    if not any(ch.isalnum() for ch in token):mask[index]=False
            benefit=counterfactual_benefit(old,c.features,c.bias,mask,int(target))
            new_omega=np.maximum(new_omega,np.clip(benefit,0,1).astype(np.float32))
        loss=UnifiedCore.learn(self,target,modality,position,mode)
        c.w_out[:]=candidate;self.omega[:]=new_omega
        if not np.isfinite(self.omega).all() or np.any(self.omega<0) or np.any(self.omega>1):raise ValueError('Importance invariant')
        return loss

    def advance(self):
        if self.controller['pending'] is None:raise ValueError('No pending supervised P0 segment')
        before=self.dumps()
        try:
            pending=self.controller['pending'];fixture=P0_FIXTURES[pending['fixture']]
            targets=[self.ids[t] for t in tokenize(fixture['answer'])]+[0];position=pending['position'];target=targets[position]
            self.word_step(pending['previous'],self.context(fixture['prompt']),position,True,fixture['prompt'])
            self._endpoint_ready=True
            try:loss=self.learn(target,'language',position)
            finally:self._endpoint_ready=False
            pending['loss_sum']+=loss;pending['position']+=1;pending['previous']=target;self.controller['endpoints']+=1
            if pending['position']==len(targets):
                self.controller['last_loss']=pending['loss_sum']/len(targets);self.controller['segments']+=1
                self.replay.append({'fixture':pending['fixture'],'source_sha256':pending['source_sha256']});self.replay=self.replay[-32:]
                self.controller['pending']=None
            return loss
        except BaseException:self._endpoint_ready=False;self.rollback(before);raise

    def teach_fixture(self,identifier,commit=None):
        before=self.dumps()
        try:
            self.begin_fixture(identifier)
            while self.controller['pending'] is not None:self.advance()
            if commit is not None:commit(self)
            return self.controller['last_loss']
        except BaseException:self.rollback(before);raise

    def base_dumps(self):return UnifiedCore.dumps(self)

    def dumps(self):
        if self._endpoint_ready:raise ValueError('Snapshot only at token boundaries')
        neural=self.base_dumps();stream=io.BytesIO();np.savez_compressed(stream,omega=self.omega);state=stream.getvalue()
        meta={'format':'scarx-p0-readout-usefulness-v1','beta':self.beta,'decay':self.decay,'fixtures':P0_FIXTURES,
              'controller':self.controller,'replay':self.replay,'fixture_rng':self.fixture_rng.bit_generator.state,'signal_rng':self.signal_rng.bit_generator.state,
              'sha256':{'model.zip':hashlib.sha256(neural).hexdigest(),'state.npz':hashlib.sha256(state).hexdigest()}}
        result=io.BytesIO()
        with zipfile.ZipFile(result,'w') as archive:
            for name,payload in (('model.zip',neural),('state.npz',state),('metadata.json',json.dumps(meta,allow_nan=False).encode())):archive.writestr(zipfile.ZipInfo(name),payload)
        raw=result.getvalue()
        if len(raw)>MAX_PAYLOAD:raise ValueError('P0 checkpoint bound')
        return raw

    @classmethod
    def loads(cls,payload):
        if not isinstance(payload,bytes) or len(payload)>MAX_PAYLOAD:raise ValueError('Bounded P0 payload required')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if len(z.namelist())!=3 or set(z.namelist())!={'model.zip','state.npz','metadata.json'} or sum(i.file_size for i in z.infolist())>MAX_PAYLOAD:raise ValueError('P0 format/size')
            neural=z.read('model.zip');state=z.read('state.npz');meta=json.loads(z.read('metadata.json'))
        if set(meta)!={'format','beta','decay','fixtures','controller','replay','fixture_rng','signal_rng','sha256'} or meta.get('format')!='scarx-p0-readout-usefulness-v1' or meta.get('fixtures')!=P0_FIXTURES or meta.get('decay')!=.995 or meta.get('sha256')!={'model.zip':hashlib.sha256(neural).hexdigest(),'state.npz':hashlib.sha256(state).hexdigest()}:raise ValueError('P0 checksum/schema')
        base=UnifiedCore.loads(neural);m=cls(base.core.config.seed,meta['beta'])
        if base.core.config!=m.core.config or base.active_count!=48 or base.encoding!={'kind':'legacy'}:raise ValueError('P0 topology/config')
        with zipfile.ZipFile(io.BytesIO(state)) as z:
            if z.namelist()!=['omega.npy'] or sum(i.file_size for i in z.infolist())>100_000:raise ValueError('Bounded importance array required')
        with np.load(io.BytesIO(state),allow_pickle=False) as arrays:omega=arrays['omega'].copy()
        if omega.dtype!=np.float32 or omega.shape!=m.omega.shape or not np.isfinite(omega).all() or np.any(omega<0) or np.any(omega>1):raise ValueError('Invalid importance')
        control=meta['controller'];replay=meta['replay']
        if not isinstance(control,dict) or set(control)!={'pending','segments','endpoints','replay_cursor','last_loss'} or any(type(control[k]) is not int or not 0<=control[k]<=1_000_000 for k in ('segments','endpoints','replay_cursor')):raise ValueError('Controller schema')
        if control['last_loss'] is not None and (type(control['last_loss']) not in (float,int) or not np.isfinite(control['last_loss']) or control['last_loss']<0):raise ValueError('Controller loss')
        pending=control['pending']
        if pending is not None:
            if not isinstance(pending,dict) or set(pending)!={'fixture','source_sha256','position','previous','loss_sum'} or pending['fixture'] not in P0_FIXTURES or pending['source_sha256']!=fixture_hash(pending['fixture']):raise ValueError('Pending provenance')
            sequence=[base.ids[t] for t in tokenize(P0_FIXTURES[pending['fixture']]['answer'])]+[0];pos=pending['position']
            if type(pos) is not int or not 0<=pos<len(sequence) or pending['previous']!=(sequence[pos-1] if pos else 1) or type(pending['loss_sum']) not in (float,int) or not np.isfinite(pending['loss_sum']) or pending['loss_sum']<0:raise ValueError('Pending tokenstate')
        if not isinstance(replay,list) or len(replay)>32 or any(not isinstance(r,dict) or set(r)!={'fixture','source_sha256'} or r['fixture'] not in P0_FIXTURES or r['source_sha256']!=fixture_hash(r['fixture']) for r in replay):raise ValueError('Replay schema')
        if control['endpoints']!=base.core.updates or control['endpoints']!=3*control['segments']+(pending['position'] if pending else 0) or len(replay)!=min(control['segments'],32) or control['replay_cursor']>control['segments']:raise ValueError('Controller/neural/replay counter mismatch')
        fr=np.random.default_rng();fr.bit_generator.state=meta['fixture_rng'];sr=np.random.default_rng();sr.bit_generator.state=meta['signal_rng']
        m.__dict__.update(base.__dict__);m.omega=omega;m.controller=copy.deepcopy(control);m.replay=copy.deepcopy(replay);m.fixture_rng=fr;m.signal_rng=sr
        return m

    def rollback(self,payload):
        restored=self.loads(payload);core=self.core;hook=self.on_tick
        core.__dict__.clear();core.__dict__.update(restored.core.__dict__)
        self.__dict__.clear();self.__dict__.update(restored.__dict__);self.core=core;self.on_tick=hook

    def save(self,path):
        payload=self.dumps();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(dir=path.parent,prefix='p0-checkpoint-',suffix='.tmp')
        try:
            with os.fdopen(fd,'wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
            replace_with_retry(name,path)
        finally:
            if os.path.exists(name):os.unlink(name)

    @classmethod
    def load(cls,path):
        path=Path(path)
        if path.stat().st_size>MAX_PAYLOAD:raise ValueError('P0 checkpoint bound')
        return cls.loads(path.read_bytes())
