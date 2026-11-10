"""Research-only label-conditioned covariance association, no backward.

Observed answers enter ONLY a separate experience phase. Prediction always
resets neuron working state and never receives the current target as input.
Serving UnifiedCore/SpikingCore implementations and checkpoints are untouched.
"""
import io,json,hashlib,zipfile
import numpy as np
from ..neural.core import SpikingCore,Config
from ..neural.language import tokenize,render
from .fixed_word_codes import validate,context as encode_context

CONDITIONS=('frozen','paired','cross_family','erase_trace')


class ObservedAssociation:
    def __init__(self,vocabulary,seed,condition,encoding):
        if condition not in CONDITIONS or vocabulary[:2]!=['<eos>','<bos>'] or not 3<=len(vocabulary)<=256 or len(set(vocabulary))!=len(vocabulary) or any(not isinstance(t,str) or not 1<=len(t)<=64 for t in vocabulary):raise ValueError('Association vocabulary/condition')
        self.vocabulary=list(vocabulary);self.ids={t:i for i,t in enumerate(vocabulary)};self.condition=condition;self.encoding=validate(encoding)
        if self.encoding['kind']!='known-digest128-v1':raise ValueError('Preregistered denseinput only')
        self.context_start=len(vocabulary);self.context_end=self.context_start+128
        self.core=SpikingCore(Config(inputs=len(vocabulary)+198,outputs=len(vocabulary)+10,hidden=48,fan_in=8,plastic_rows=12,
            seed=seed,leak=.65,threshold=.5,adaptation_strength=.02,feature_decay=.25,readout_lr=.08,hidden_lr=.02,anchor_strength=0))
        self.core.w_in-=.15;self.core.stable_in[:]=self.core.w_in
        self.mean_query=np.zeros(128,np.float32);self.mean_post=np.zeros(48,np.float32);self.question_trace=np.zeros(128,np.float32)
        self.slot_clock=0;self.hidden_transactions=0;self.experiences=0;self.observations=0;self.ticks=0
        self.sampler_rng=np.random.default_rng(seed+11200000)
        self.output_mask=self._mask(0)

    def _tokens(self,target):
        if not isinstance(target,str):raise ValueError('Observed/target text required')
        words=tokenize(target)
        if not words or len(words)>6 or any(t not in self.ids for t in words):raise ValueError('Bounded7slots inclEOS/vocabulary')
        return [self.ids[t] for t in words]+[0]

    def _tick(self,x):self.core.step(x,False);self.ticks+=1

    def _reset_prediction(self):
        self.core.reset_state()
        for _ in range(12):self._tick(np.zeros(self.core.config.inputs,np.float32))

    def _word_input(self,previous,context,position):
        x=np.zeros(self.core.config.inputs,np.float32);x[previous]=.8
        x[self.context_start:self.context_end]=1.6*context
        x[-6:]=[1.,0.,.2,position/24,.2*np.sin(position),.2*np.cos(position)]
        return x

    def _word_step(self,previous,context,position):
        x=self._word_input(previous,context,position)
        for _ in range(3):self._tick(x)
        return self.probabilities(position)

    def _mask(self,position):
        mask=np.zeros(self.core.config.outputs,bool);mask[:len(self.vocabulary)]=True;mask[1]=False
        if position==0:
            mask[0]=False
            for i,t in enumerate(self.vocabulary[2:],2):
                if not any(c.isalnum() for c in t):mask[i]=False
        return mask

    def probabilities(self,position):
        logits=self.core.w_out@self.core.features+self.core.bias;mask=self._mask(position);self.output_mask[:]=mask
        p=np.zeros(len(logits),np.float32);values=np.exp(logits[mask]-logits[mask].max());p[mask]=values/values.sum();return p

    def _associate_slot(self,post):
        if post.shape!=(48,) or not np.isfinite(post).all():raise ValueError('Local postactivity')
        rows=np.arange(12)+(self.slot_clock%4)*12;self.core.rows[:]=rows
        if self.condition in ('paired','cross_family'):
            delta=np.clip(.02*(post[rows]-self.mean_post[rows])[:,None]*self.question_trace[None,:],-.025,.025)
            weight=self.core.w_in[rows,self.context_start:self.context_end]
            new=np.clip(weight+delta,-3,3)
            if np.any(new!=weight):self.hidden_transactions+=1
            self.core.w_in[rows,self.context_start:self.context_end]=new
        self.mean_post[:]=.99*self.mean_post+.01*post;self.slot_clock+=1

    def observe(self,prompt,observed):
        context=encode_context(self.encoding,prompt);tokens=self._tokens(observed)
        self._reset_prediction();x=self._word_input(1,context,0);self.question_trace.fill(0)
        for _ in range(3):
            self._tick(x);self.question_trace[:]=.9*self.question_trace+(1.6*context-self.mean_query)
        if self.condition=='erase_trace':self.question_trace.fill(0)
        for position in range(7):
            x=self._word_input(tokens[position],np.zeros(128,np.float32),position) if position<len(tokens) else np.zeros(self.core.config.inputs,np.float32)
            for _ in range(3):self._tick(x);self.question_trace*=.9
            self._associate_slot(self.core.features.copy())
        self.mean_query[:]=.99*self.mean_query+.01*(1.6*context);self.observations+=1

    def learn_prediction(self,prompt,target):
        context=encode_context(self.encoding,prompt);tokens=self._tokens(target)
        self._reset_prediction();previous=1;losses=[]
        for position,token in enumerate(tokens):
            p=self._word_step(previous,context,position);losses.append(-float(np.log(max(float(p[token]),1e-12))))
            error=-p;error[token]+=1
            self.core.w_out+=np.clip(.08*error[:,None]*self.core.features[None,:],-.025,.025)
            self.core.bias+=np.clip(.08*error,-.025,.025)
            np.clip(self.core.w_out,-3,3,out=self.core.w_out);np.clip(self.core.bias,-3,3,out=self.core.bias)
            self.core.updates+=1;previous=token
        return {'loss':float(np.mean(losses)),'tokens':len(tokens)}

    def learn(self,prompt,target,observed=None):
        # Validate both phases before ANY mutation, including means/neurondynamics.
        encode_context(self.encoding,prompt);self._tokens(target)
        if self.condition=='cross_family' and observed is None:raise ValueError('Cross-family observation required')
        observed=target if observed is None else observed;self._tokens(observed)
        self.observe(prompt,observed);r=self.learn_prediction(prompt,target);self.experiences+=1;return r

    def generate(self,prompt,max_words=12):
        if type(max_words) is not int or not 1<=max_words<=12:raise ValueError('Generation budget1..12')
        context=encode_context(self.encoding,prompt);self._reset_prediction();previous=1;words=[];stopped=False
        for position in range(max_words):
            p=self._word_step(previous,context,position);identifier=int(p.argmax())
            if identifier==0:stopped=True;break
            words.append(self.vocabulary[identifier]);previous=identifier
        return {'text':render(words),'stopped':stopped,'scope':'experimental observed-association ALIF, no goldinput/observation at inference, not serving'}

    def teacher_metrics(self,prompt,target):
        context=encode_context(self.encoding,prompt);tokens=self._tokens(target);self._reset_prediction();previous=1;correct=0;loss=0.
        for position,token in enumerate(tokens):
            p=self._word_step(previous,context,position);correct+=int(p.argmax()==token);loss-=np.log(max(float(p[token]),1e-12))
            if position==0:first=bool(p.argmax()==token)
            previous=token
        return {'token_accuracy':correct/len(tokens),'nll':float(loss/len(tokens)),'first_token_correct':first}

    def prompt_features(self,prompt):
        context=encode_context(self.encoding,prompt);self._reset_prediction();self._word_step(1,context,0)
        return self.core.features.copy()

    def memory_bytes(self):return self.core.memory_bytes()+self.mean_query.nbytes+self.mean_post.nbytes+self.question_trace.nbytes+self.output_mask.nbytes

    def dumps(self):
        raw=self.core.dumps();extra=io.BytesIO();np.savez(extra,mean_query=self.mean_query,mean_post=self.mean_post,question_trace=self.question_trace,output_mask=self.output_mask)
        extra=extra.getvalue();meta={'format':'scarx-observed-association-v1','vocabulary':self.vocabulary,'condition':self.condition,'encoding':self.encoding,
            'slot_clock':self.slot_clock,'hidden_transactions':self.hidden_transactions,'experiences':self.experiences,'observations':self.observations,'ticks':self.ticks,
            'sampler_rng':self.sampler_rng.bit_generator.state,'core_sha256':hashlib.sha256(raw).hexdigest(),'extra_sha256':hashlib.sha256(extra).hexdigest()}
        meta['metadata_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            for name,value in [('core.npz',raw),('local-state.npz',extra),('metadata.json',json.dumps(meta,ensure_ascii=False))]:z.writestr(zipfile.ZipInfo(name),value)
        return out.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>3_000_000:raise ValueError('Association archive exceeds3MB')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'core.npz','local-state.npz','metadata.json'} or len(z.namelist())!=3 or sum(i.file_size for i in z.infolist())>3_000_000 or z.getinfo('metadata.json').file_size>65536:raise ValueError('Association archive members')
            raw=z.read('core.npz');extra=z.read('local-state.npz');meta=json.loads(z.read('metadata.json'))
        checksum=meta.pop('metadata_sha256')
        if meta['format']!='scarx-observed-association-v1' or hashlib.sha256(json.dumps(meta,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=checksum or hashlib.sha256(raw).hexdigest()!=meta['core_sha256'] or hashlib.sha256(extra).hexdigest()!=meta['extra_sha256']:raise ValueError('Association checksum/format')
        m=cls(meta['vocabulary'],0,meta['condition'],meta['encoding']);core=SpikingCore.loads(raw)
        if (core.config.hidden,core.config.inputs,core.config.outputs,core.config.fan_in,core.config.plastic_rows,core.config.neuron)!=(48,m.core.config.inputs,m.core.config.outputs,8,12,'alif'):raise ValueError('Association topology')
        expected=Config(inputs=m.core.config.inputs,outputs=m.core.config.outputs,hidden=48,fan_in=8,plastic_rows=12,seed=core.config.seed,
            leak=.65,threshold=.5,adaptation_strength=.02,feature_decay=.25,readout_lr=.08,hidden_lr=.02,anchor_strength=0)
        if core.config!=expected:raise ValueError('Association fixed configuration')
        with zipfile.ZipFile(io.BytesIO(extra)) as z:
            if sum(i.file_size for i in z.infolist())>10000 or len(z.namelist())!=4:raise ValueError('Association localstate bound')
        with np.load(io.BytesIO(extra),allow_pickle=False) as z:
            if set(z.files)!={'mean_query','mean_post','question_trace','output_mask'}:raise ValueError('Association localstate fields')
            for name in z.files:
                a=z[name];template=getattr(m,name)
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all():raise ValueError('Association localstate values')
                template[:]=a
        if not any(np.array_equal(m.output_mask,m._mask(i)) for i in (0,1)):raise ValueError('Association outputmask policy')
        for name in ('slot_clock','hidden_transactions','experiences','observations','ticks'):
            v=meta[name]
            if type(v) is not int or v<0:raise ValueError('Association counter')
            setattr(m,name,v)
        if m.slot_clock!=7*m.observations or m.hidden_transactions>m.slot_clock or m.experiences>m.observations or m.hidden_transactions and m.condition in ('frozen','erase_trace'):raise ValueError('Association inconsistent counters')
        m.core=core;m.sampler_rng.bit_generator.state=meta['sampler_rng'];return m
