"""Experimental local auxiliary-loss adaptation, not a DECOLLE reproduction.

Only fixed local G supplies hidden teacher error; no task W_out feedback,
autograd, reverse-time tape or cross-neuron gradient transport. EPSP/eligibility
are forward-only approximations with adaptation/reset paths detached.
"""
from dataclasses import asdict
import io,json,hashlib,zipfile
import numpy as np
from .core import SpikingCore,Config


def softmax(x):
    values=np.exp(x-x.max());return values/values.sum()


class LocalClassifierCore(SpikingCore):
    extra_names=('voltage_pre','fixed_classifier')

    def __init__(self,config):
        if config.neuron!='alif':raise ValueError('ALIF required')
        super().__init__(config)
        self.rng=np.random.default_rng(config.seed+3100000)
        self.fixed_classifier=self.rng.normal(0,1/np.sqrt(config.hidden),(config.outputs,config.hidden)).astype(np.float32)
        self.aux_updates=0

    def reset_state(self):
        super().reset_state();self.voltage_pre=np.zeros(self.config.inputs+self.config.hidden,np.float32)

    def step(self,x,track=True):
        c=self.config;x=np.asarray(x,np.float32)
        if x.shape!=(c.inputs,) or not np.isfinite(x).all():raise ValueError('Invalid input')
        old=self.spikes.copy();current=self.w_in@x+np.sum(self.w_rec*old[self.indices],axis=1)
        self.voltage=c.leak*self.voltage+current;limit=c.threshold+c.adaptation_strength*self.adaptation
        psi=.3/np.maximum(limit,.001)*np.maximum(0,1-np.abs((self.voltage-limit)/np.maximum(limit,.001)))
        self.spikes=(self.voltage>=limit).astype(np.float32);self.voltage-=self.spikes*limit
        self.adaptation=c.adaptation_decay*self.adaptation+self.spikes
        self.features=c.feature_decay*self.features+(1-c.feature_decay)*self.spikes
        if track:
            self.voltage_pre*=c.leak;self.voltage_pre[:c.inputs]+=x;self.voltage_pre[c.inputs:]+=old
            for key,presynaptic in (('in',np.broadcast_to(self.voltage_pre[:c.inputs],(c.plastic_rows,c.inputs))),
              ('rec',self.voltage_pre[c.inputs:][self.indices[self.rows]])):
                trace=getattr(self,'elig_'+key);trace*=c.feature_decay;trace+=(1-c.feature_decay)*psi[self.rows,None]*presynaptic
                np.clip(trace,-10,10,out=trace)
        return self.probabilities()

    def auxiliary_signal(self,target):
        if type(target) not in (int,np.int32,np.int64) or not 0<=target<self.config.outputs:raise ValueError('Invalid target')
        p=softmax(self.fixed_classifier@self.features);error=-p.copy();error[target]+=1
        return self.fixed_classifier.T@error,-float(np.log(max(float(p[target]),1e-12)))

    def learn_auxiliary(self,target,shuffle=False):
        if type(target) not in (int,np.int32,np.int64) or not 0<=target<self.config.outputs:raise ValueError('Invalid auxiliary target')
        if shuffle:target=int(self.rng.integers(self.config.outputs))
        signal,loss=self.auxiliary_signal(target);c=self.config
        for key in ('in','rec'):
            weight=getattr(self,'w_'+key)
            weight[self.rows]+=np.clip(c.hidden_lr*signal[self.rows,None]*getattr(self,'elig_'+key),-c.update_clip,c.update_clip)
            np.clip(weight,-c.weight_clip,c.weight_clip,out=weight)
        self.aux_updates+=1;return loss

    def arrays(self):return {**super().arrays(),**{k:getattr(self,k) for k in self.extra_names}}

    def dumps(self):
        arrays=self.arrays();meta={'format':'scarx-local-classifier-v1','config':asdict(self.config),'updates':self.updates,'aux_updates':self.aux_updates,
          'reward_mean':self.reward_mean,'rng':self.rng.bit_generator.state,'sha256':{k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in arrays.items()}}
        stream=io.BytesIO();np.savez(stream,metadata=np.array(json.dumps(meta)),**arrays);return stream.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>20000000:raise ValueError('Oversized local classifier checkpoint')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if sum(i.file_size for i in z.infolist())>20000000 or z.getinfo('metadata.npy').file_size>65536 or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Bad classifier members')
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['format']!='scarx-local-classifier-v1':raise ValueError('Classifier format')
            m=cls(Config(**meta['config']));expected=m.arrays()
            if set(z.files)!=set(expected)|{'metadata'}:raise ValueError('Classifier fields')
            for name,template in expected.items():
                a=z[name]
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all() or hashlib.sha256(a.tobytes()).hexdigest()!=meta['sha256'][name]:raise ValueError('Corrupt classifier array')
                template[:]=a
            for name in ('updates','aux_updates'):
                if type(meta[name]) is not int or meta[name]<0:raise ValueError('Bad update counter')
                setattr(m,name,meta[name])
            m.reward_mean=meta['reward_mean'];m.rng.bit_generator.state=meta['rng']
            if not np.isfinite(m.reward_mean) or np.any(m.indices<0) or np.any(m.indices>=m.config.hidden) or np.any(m.rows<0) or np.any(m.rows>=m.config.hidden):raise ValueError('Bad classifier state')
        return m
