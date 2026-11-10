"""Small proposal30 word interface, same ALIF/eligibility; never serving."""
import copy,hashlib,io,json,zipfile
import numpy as np
from ..neural.core import Config
from ..neural.unified import UnifiedCore
from ..neural.language import tokenize,render
from .readout_eligibility import ReadoutEligibility


class DirectWords:
    def __init__(self,vocabulary,seed,condition='correct'):
        if not isinstance(vocabulary,list) or vocabulary[:2]!=['<eos>','<bos>'] or not 3<=len(vocabulary)<=256 or len(set(vocabulary))!=len(vocabulary) or any(not isinstance(t,str) or not 1<=len(t)<=64 for t in vocabulary):raise ValueError('Bounded unique wordvocabulary')
        self.vocabulary=list(vocabulary);self.ids={t:i for i,t in enumerate(vocabulary)}
        self.learner=ReadoutEligibility(Config(inputs=len(vocabulary)+198,outputs=len(vocabulary)+10,hidden=48,fan_in=8,plastic_rows=12,seed=seed,
            leak=.65,threshold=.5,adaptation_strength=.02,feature_decay=.25,anchor_strength=0),condition)
        self.learner.core.w_in-=.15;self.learner.core.stable_in[:]=self.learner.core.w_in
        self.context=np.zeros(128,np.float32);self.prepared=False;self.previous=1;self.position=0;self.responses_seen=0
        self.sampler_rng=np.random.default_rng(seed+36400000);self.calibration_rng=np.random.default_rng(seed+36300000)

    @property
    def core(self):return self.learner.core
    @property
    def head(self):return self.learner.head
    @property
    def ticks(self):return self.learner.total_ticks

    def _context(self,prompt):
        # Call the existing encoder method without allocating another neuralcore.
        adapter=object.__new__(UnifiedCore);adapter.encoding={'kind':'legacy'}
        return UnifiedCore.context(adapter,prompt)

    def mask(self,position):
        mask=np.zeros(self.core.config.outputs,bool);mask[:len(self.vocabulary)]=True;mask[1]=False
        if position==0:
            mask[0]=False
            for i,t in enumerate(self.vocabulary[2:],2):
                if not any(c.isalnum() for c in t):mask[i]=False
        return mask

    def prepare(self,prompt,track=False):
        context=self._context(prompt)
        self.learner.begin(track);self.context[:]=context
        for _ in range(12):self.learner.step(np.zeros(self.core.config.inputs,np.float32))
        self.prepared=True;self.previous=1;self.position=0

    def word(self,previous,position):
        if not self.prepared or not self.learner.segment_active or type(previous) is not int or not 0<=previous<len(self.vocabulary) or type(position) is not int or not 0<=position<8:raise ValueError('Prepared bounded wordstate')
        x=np.zeros(self.core.config.inputs,np.float32);x[previous]=.8;x[len(self.vocabulary):len(self.vocabulary)+128]=self.context*1.6
        x[-6:]=[1.,0.,.2,position/24,.2*np.sin(position),.2*np.cos(position)]
        for _ in range(3):self.learner.step(x)
        self.learner.output_mask[:]=self.mask(position);self.previous=previous;self.position=position
        return self.core.features.copy()

    def target_tokens(self,target):
        if not isinstance(target,str):raise ValueError('Explicit supervisedtarget')
        words=tokenize(target)
        if not 1<=len(words)<=7 or any(t not in self.ids for t in words):raise ValueError('Target wordbudget/vocabulary')
        return [self.ids[t] for t in words]+[0]

    def calibrate(self,prompts):
        if self.head.frozen or not prompts:raise ValueError('UnlabeledTRAINcalibration only')
        while self.head.count<128:
            prompt=prompts[int(self.calibration_rng.integers(len(prompts)))];length=int(self.calibration_rng.integers(3))
            prefix=[int(t) for t in self.calibration_rng.integers(2,len(self.vocabulary),length)]
            self.prepare(prompt);f=self.word(1,0)
            for i,t in enumerate(prefix,1):f=self.word(t,i)
            self.head.observe(f);self.learner.segment_active=False
        self.head.freeze()

    def learn(self,prompt,target):
        tokens=self.target_tokens(target)
        if not self.head.frozen:raise ValueError('Calibrate first')
        snapshot=copy.deepcopy(self.__dict__)
        try:
            self.prepare(prompt,True);previous=1;loss=[]
            for i,token in enumerate(tokens):self.word(previous,i);loss.append(self.learner.accumulate(token));previous=token
            self.learner.finish();self.responses_seen+=1;return {'loss':float(np.mean(loss)),'tokens':len(tokens)}
        except BaseException:self.__dict__.clear();self.__dict__.update(snapshot);raise

    def generate(self,prompt,max_words=8):
        if not self.head.frozen or type(max_words) is not int or not 1<=max_words<=8:raise ValueError('Calibrated boundedgeneration')
        self.prepare(prompt);previous=1;out=[];stopped=False
        for position in range(max_words):
            f=self.word(previous,position);token=int(self.head.probabilities(f,self.learner.output_mask).argmax())
            if token==0:stopped=True;break
            out.append(self.vocabulary[token]);previous=token
        self.learner.segment_active=False
        return {'text':render(out),'stopped':stopped,'scope':'Proposal30research neuralwordhead, no target/intent/LLMproxy at inference; not main/generalchat.'}

    def teacher_metrics(self,prompt,target):
        tokens=self.target_tokens(target);self.prepare(prompt);previous=1;losses=[];correct=[]
        for position,token in enumerate(tokens):
            p=self.head.probabilities(self.word(previous,position),self.learner.output_mask);losses.append(-np.log(max(float(p[token]),1e-12)));correct.append(int(p.argmax())==token);previous=token
        self.learner.segment_active=False;return {'nll':float(np.mean(losses)),'token_accuracy':float(np.mean(correct)),'firstword_correct':correct[0]}

    def memory_bytes(self):return self.learner.memory_bytes()+self.context.nbytes

    def dumps(self):
        raw=self.learner.dumps();state=io.BytesIO();np.savez(state,context=self.context);state=state.getvalue()
        meta={'format':'scarx-direct-word-bridge-v1','vocabulary':self.vocabulary,'prepared':self.prepared,'previous':self.previous,'position':self.position,'responses_seen':self.responses_seen,
            'sampler_rng':self.sampler_rng.bit_generator.state,'calibration_rng':self.calibration_rng.bit_generator.state,'learner_sha256':hashlib.sha256(raw).hexdigest(),'state_sha256':hashlib.sha256(state).hexdigest()}
        meta['metadata_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True,ensure_ascii=False).encode()).hexdigest();out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:z.writestr(zipfile.ZipInfo('learner.eligibility'),raw);z.writestr(zipfile.ZipInfo('state.npz'),state);z.writestr(zipfile.ZipInfo('metadata.json'),json.dumps(meta,ensure_ascii=False))
        result=out.getvalue()
        if len(result)>5_000_000:raise MemoryError('Wordcheckpoint exceeds5MB')
        return result

    @classmethod
    def loads(cls,payload):
        if len(payload)>5_000_000:raise ValueError('Wordcheckpoint bound')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'learner.eligibility','state.npz','metadata.json'} or len(z.namelist())!=3 or sum(i.file_size for i in z.infolist())>5_000_000 or z.getinfo('metadata.json').file_size>65536:raise ValueError('Wordcheckpoint members')
            raw=z.read('learner.eligibility');state=z.read('state.npz');meta=json.loads(z.read('metadata.json'))
        checksum=meta.pop('metadata_sha256')
        if meta.get('format')!='scarx-direct-word-bridge-v1' or hashlib.sha256(json.dumps(meta,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=checksum or hashlib.sha256(raw).hexdigest()!=meta['learner_sha256'] or hashlib.sha256(state).hexdigest()!=meta['state_sha256']:raise ValueError('Wordcheckpoint checksum')
        learned=ReadoutEligibility.loads(raw);m=cls(meta['vocabulary'],learned.core.config.seed,learned.condition)
        if learned.core.config!=m.core.config:raise ValueError('Wordfixedconfig')
        m.learner=learned
        with zipfile.ZipFile(io.BytesIO(state)) as z:
            if z.namelist()!=['context.npy'] or z.getinfo('context.npy').file_size>2048:raise ValueError('Wordcontext bound')
        with np.load(io.BytesIO(state),allow_pickle=False) as z:
            a=z['context']
            if a.shape!=(128,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Wordcontext array')
            m.context[:]=a
        if type(meta['prepared']) is not bool or type(meta['previous']) is not int or not 0<=meta['previous']<len(m.vocabulary) or type(meta['position']) is not int or not 0<=meta['position']<8 or type(meta['responses_seen']) is not int or meta['responses_seen']<0 or meta['responses_seen']!=learned.segments_learned:raise ValueError('Wordstate counters')
        for name in ('prepared','previous','position','responses_seen'):setattr(m,name,meta[name])
        m.sampler_rng.bit_generator.state=meta['sampler_rng'];m.calibration_rng.bit_generator.state=meta['calibration_rng']
        return m
