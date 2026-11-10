"""Experimental jointRGB/question to actual autoregressive neural words.

All hidden weights FROZEN. One ALIF core plus bounded heldvisualtrace. Only
output-local delta learns; no renderer/intent/answerlookup or error to hidden.
"""
import io,json,zipfile,hashlib
import numpy as np
from ..neural.core import SpikingCore,Config
from ..neural.input_events import InputEvents
from ..neural.language import tokenize,render
from .readout_conditioning import ConditionedReadout


class VisualWords:
    def __init__(self,vocabulary,seed,condition='full'):
        if condition not in ('full','no_image','no_question') or vocabulary[:2]!=['<eos>','<bos>'] or not 3<=len(vocabulary)<=256 or len(set(vocabulary))!=len(vocabulary) or any(not isinstance(t,str) or not 1<=len(t)<=64 for t in vocabulary):raise ValueError('Visualword vocabulary/condition')
        self.vocabulary=list(vocabulary);self.ids={t:i for i,t in enumerate(vocabulary)};self.condition=condition;self.encoder=InputEvents(len(vocabulary))
        self.core=SpikingCore(Config(inputs=len(vocabulary)+198,outputs=len(vocabulary)+10,hidden=48,fan_in=8,plastic_rows=12,seed=seed,
            leak=.65,threshold=.5,adaptation_strength=.02,feature_decay=.25,readout_lr=.08,hidden_lr=0.,anchor_strength=0))
        self.core.w_in-=.15;self.core.stable_in[:]=self.core.w_in
        self.head=ConditionedReadout(self.core.w_out,self.core.bias,'normalized');self.core.w_out=self.head.weights;self.core.bias=self.head.bias
        self.question_feature=np.zeros(128,np.float32);self.visual_feature=np.zeros(48,np.float32);self.output_mask=self.mask(0)
        self.ticks=0;self.experiences=0;self.images_seen=0;self.prepared=False
        self.sampler_rng=np.random.default_rng(seed+12100000);self.calibration_rng=np.random.default_rng(seed+12200000)

    def mask(self,position):
        mask=np.zeros(self.core.config.outputs,bool);mask[:len(self.vocabulary)]=True;mask[1]=False
        if position==0:
            mask[0]=False
            for i,t in enumerate(self.vocabulary[2:],2):
                if not any(c.isalnum() for c in t):mask[i]=False
        return mask

    def _tick(self,x):self.core.step(x,False);self.ticks+=1

    def _word_input(self,previous,position):
        x=np.zeros(self.core.config.inputs,np.float32);x[previous]=.8
        x[len(self.vocabulary):self.encoder.image_start]=1.6*self.question_feature
        x[-6:]=[1.,0.,.2,position/24,.2*np.sin(position),.2*np.cos(position)];return x

    def prepare(self,question,image,ablation=None):
        if ablation not in (None,'blank_image','blank_question'):raise ValueError('Unknown causalablation')
        if not isinstance(question,str) or not 1<=len(question.strip())<=2048:raise ValueError('Question1..2048chars')
        words=tokenize(question)
        if not 1<=len(words)<=64 or any(len(t)>64 for t in words):raise ValueError('Question1..64tokens,64chars/token')
        events=self.encoder.image(image) # validate image BEFORE state mutation
        q=sum((self.encoder.token_vector(t) for t in words),np.zeros(128,np.float32));q/=max(1.,float(np.linalg.norm(q)))
        if self.condition=='no_question' or ablation=='blank_question':q.fill(0)
        self.question_feature[:]=q;self.visual_feature.fill(0);self.core.reset_state()
        for _ in range(12):self._tick(np.zeros(self.core.config.inputs,np.float32))
        query=self._word_input(1,0)
        for _ in range(3):self._tick(query)
        features=[]
        for index,event in enumerate(events):
            x=event.copy();x[len(self.vocabulary):self.encoder.image_start]=1.6*q;x[-6]=1.
            if self.condition=='no_image' or ablation=='blank_image':x[self.encoder.image_start:self.encoder.image_start+48]=0
            self._tick(x)
            if 1<=index<=16:features.append(self.core.features.copy())
        self.visual_feature[:]=np.mean(features,axis=0);self.images_seen+=1;self.prepared=True

    def word_feature(self,previous,position):
        if not self.prepared or type(previous) is not int or not 0<=previous<len(self.vocabulary) or type(position) is not int or not 0<=position<8:raise ValueError('Bounded preparedword state')
        x=self._word_input(previous,position)
        for _ in range(3):self._tick(x)
        self.output_mask[:]=self.mask(position)
        return .5*(self.visual_feature+self.core.features)

    def calibrate(self,examples):
        if self.head.frozen:raise ValueError('Calibration frozen')
        if not examples:raise ValueError('Training-only calibrationinputs required')
        while self.head.count<128:
            row=examples[int(self.calibration_rng.integers(len(examples)))];length=int(self.calibration_rng.integers(3))
            prefix=[int(i) for i in self.calibration_rng.integers(2,len(self.vocabulary),length)]
            self.prepare(row['prompt'],row['image']);feature=self.word_feature(1,0)
            for position,token in enumerate(prefix,1):feature=self.word_feature(token,position)
            self.head.observe(feature)
        self.head.freeze()

    def target_tokens(self,target):
        if not isinstance(target,str):raise ValueError('Supervisedtarget text')
        words=tokenize(target)
        if not 1<=len(words)<=7 or any(t not in self.ids for t in words):raise ValueError('Target vocabulary/wordbudget')
        return [self.ids[t] for t in words]+[0]

    def learn(self,question,image,target):
        tokens=self.target_tokens(target)
        if not self.head.frozen:raise ValueError('Training-only calibration required')
        self.prepare(question,image);previous=1;loss=[]
        for position,token in enumerate(tokens):
            feature=self.word_feature(previous,position);loss.append(self.head.learn(feature,int(token),self.output_mask))
            self.core.updates+=1;previous=int(token)
        self.experiences+=1;return {'loss':float(np.mean(loss)),'tokens':len(tokens)}

    def generate(self,question,image,max_words=8,ablation=None):
        if type(max_words) is not int or not 1<=max_words<=8 or not self.head.frozen:raise ValueError('Calibratedwordbudget1..8')
        self.prepare(question,image,ablation);previous=1;out=[];stopped=False
        for position in range(max_words):
            p=self.head.probabilities(self.word_feature(previous,position),self.output_mask);token=int(p.argmax())
            if token==0:stopped=True;break
            out.append(self.vocabulary[token]);previous=token
        return {'text':render(out),'stopped':stopped,'scope':'ExperimentaloneALIF/frozenhidden + localwordhead; no renderer/goldinput/template.'}

    def teacher_metrics(self,question,image,target):
        tokens=self.target_tokens(target);self.prepare(question,image);previous=1;loss=0.;correct=0
        for position,token in enumerate(tokens):
            p=self.head.probabilities(self.word_feature(previous,position),self.output_mask);correct+=int(p.argmax()==token);loss-=np.log(max(float(p[token]),1e-12))
            if position==0:first=bool(p.argmax()==token)
            previous=int(token)
        return {'nll':float(loss/len(tokens)),'token_accuracy':correct/len(tokens),'first_token_correct':first}

    def memory_bytes(self):return self.core.memory_bytes()+self.question_feature.nbytes+self.visual_feature.nbytes+self.output_mask.nbytes+self.head.mean.nbytes+self.head.m2.nbytes

    def dumps(self):
        core=self.core.dumps();head=self.head.dumps();state=io.BytesIO();np.savez(state,question_feature=self.question_feature,visual_feature=self.visual_feature,output_mask=self.output_mask);state=state.getvalue()
        metadata={'format':'scarx-visual-word-readout-v1','vocabulary':self.vocabulary,'condition':self.condition,'ticks':self.ticks,'experiences':self.experiences,
            'images_seen':self.images_seen,'prepared':self.prepared,'sampler_rng':self.sampler_rng.bit_generator.state,'calibration_rng':self.calibration_rng.bit_generator.state,
            'core_sha256':hashlib.sha256(core).hexdigest(),'head_sha256':hashlib.sha256(head).hexdigest(),'state_sha256':hashlib.sha256(state).hexdigest()}
        metadata['metadata_sha256']=hashlib.sha256(json.dumps(metadata,sort_keys=True,ensure_ascii=False).encode()).hexdigest();out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            for name,value in [('core.npz',core),('head.npz',head),('state.npz',state),('metadata.json',json.dumps(metadata,ensure_ascii=False))]:z.writestr(zipfile.ZipInfo(name),value)
        return out.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>3_000_000:raise ValueError('Visualwordarchive bound')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'core.npz','head.npz','state.npz','metadata.json'} or len(z.namelist())!=4 or sum(i.file_size for i in z.infolist())>3_000_000 or z.getinfo('metadata.json').file_size>65536:raise ValueError('Visualwordmembers')
            core=z.read('core.npz');head=z.read('head.npz');state=z.read('state.npz');meta=json.loads(z.read('metadata.json'))
        sha=meta.pop('metadata_sha256')
        if meta['format']!='scarx-visual-word-readout-v1' or hashlib.sha256(json.dumps(meta,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=sha or any(hashlib.sha256(v).hexdigest()!=meta[n+'_sha256'] for n,v in [('core',core),('head',head),('state',state)]):raise ValueError('Visualword checksum')
        m=cls(meta['vocabulary'],0,meta['condition']);loaded=SpikingCore.loads(core)
        expected=Config(inputs=m.core.config.inputs,outputs=m.core.config.outputs,hidden=48,fan_in=8,plastic_rows=12,seed=loaded.config.seed,
            leak=.65,threshold=.5,adaptation_strength=.02,feature_decay=.25,readout_lr=.08,hidden_lr=0.,anchor_strength=0)
        if loaded.config!=expected:raise ValueError('Visualwordfixedconfiguration')
        for raw,limit in [(head,100000),(state,10000)]:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                if len(z.namelist())!=len(set(z.namelist())) or sum(i.file_size for i in z.infolist())>limit:raise ValueError('Visualwordinnerbounds')
        with np.load(io.BytesIO(head),allow_pickle=False) as z:
            if set(z.files)!={'metadata','weights','bias','mean','m2'}:raise ValueError('Visualwordheadfields')
            hmeta=json.loads(str(z['metadata']))
            if hmeta.get('mode')!='normalized' or type(hmeta.get('count')) is not int or not 0<=hmeta['count']<=128 or type(hmeta.get('updates')) is not int or hmeta['updates']<0 or type(hmeta.get('frozen')) is not bool or hmeta['frozen'] and hmeta['count']!=128:raise ValueError('Visualwordheadpolicy')
            for name in ('weights','bias','mean','m2'):
                a=z[name];template=getattr(m.head,name)
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all():raise ValueError('Visualwordheadarray')
            if np.any(z['m2']<0):raise ValueError('Negativecalibrationvariance')
        actual_head=ConditionedReadout.loads(head)
        np.testing.assert_array_equal(loaded.w_out,actual_head.weights);np.testing.assert_array_equal(loaded.bias,actual_head.bias)
        m.core=loaded;m.head=actual_head;m.core.w_out=m.head.weights;m.core.bias=m.head.bias
        with np.load(io.BytesIO(state),allow_pickle=False) as z:
            if set(z.files)!={'question_feature','visual_feature','output_mask'}:raise ValueError('Visualwordstatefields')
            for name in z.files:
                a=z[name];template=getattr(m,name)
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all():raise ValueError('Visualwordstatearray')
                template[:]=a
        if not any(np.array_equal(m.output_mask,m.mask(p)) for p in (0,1)):raise ValueError('Visualwordmaskpolicy')
        for name in ('ticks','experiences','images_seen'):
            v=meta[name]
            if type(v) is not int or v<0:raise ValueError('Visualwordcounter')
            setattr(m,name,v)
        if type(meta['prepared']) is not bool or loaded.updates!=m.head.updates:raise ValueError('Visualwordstatepolicy')
        m.prepared=meta['prepared'];m.sampler_rng.bit_generator.state=meta['sampler_rng'];m.calibration_rng.bit_generator.state=meta['calibration_rng'];return m
