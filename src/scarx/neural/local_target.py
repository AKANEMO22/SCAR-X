"""Experimental neuron-local target-rate matching. No global error feedback.

Same ALIF forward as SpikingCore. Forward sensitivities ignore cross-neuron
derivatives, so this is not exact e-prop or an unbiased network gradient.
NOT selected by the serving runtime; protocol p1-local-target-v1.
"""
from dataclasses import asdict
import io,json,hashlib,zipfile
import numpy as np
from .core import SpikingCore,Config


class LocalTargetCore(SpikingCore):
    extra_names=('qv_in','qa_in','qv_rec','qa_rec','target_rates')

    def __init__(self,config):
        if config.neuron!='alif':raise ValueError('Local target candidate requires ALIF')
        super().__init__(config)
        self.mod_rng=np.random.default_rng(config.seed+900000)
        self.target_rates=np.full((config.hidden,config.outputs),.02,np.float32)
        for j in range(config.hidden):
            self.target_rates[j,self.mod_rng.permutation(config.outputs)[:max(1,config.outputs//2)]]=.35

    def reset_state(self):
        super().reset_state()
        self.qv_in=np.zeros_like(self.elig_in);self.qa_in=np.zeros_like(self.elig_in)
        self.qv_rec=np.zeros_like(self.elig_rec);self.qa_rec=np.zeros_like(self.elig_rec)

    def step(self,x,track=True):
        c=self.config;x=np.asarray(x,dtype=np.float32)
        if x.shape!=(c.inputs,) or not np.isfinite(x).all():raise ValueError('Invalid input')
        old=self.spikes.copy();self.voltage=c.leak*self.voltage+self.w_in@x+np.sum(self.w_rec*old[self.indices],axis=1)
        limit=c.threshold+c.adaptation_strength*self.adaptation
        psi=.3/np.maximum(limit,.001)*np.maximum(0,1-np.abs((self.voltage-limit)/np.maximum(limit,.001)))
        self.spikes=(self.voltage>=limit).astype(np.float32);self.voltage-=self.spikes*limit
        self.adaptation=c.adaptation_decay*self.adaptation+self.spikes
        self.features=c.feature_decay*self.features+(1-c.feature_decay)*self.spikes
        if track:
            r=self.rows
            for key,pre in (('in',np.broadcast_to(x,(len(r),c.inputs))),('rec',old[self.indices[r]])):
                qv=getattr(self,'qv_'+key);qa=getattr(self,'qa_'+key);elig=getattr(self,'elig_'+key)
                qv*=c.leak;qv+=pre
                e=psi[r,None]*(qv-c.adaptation_strength*qa)
                qv-=limit[r,None]*e+c.adaptation_strength*self.spikes[r,None]*qa
                qa*=c.adaptation_decay;qa+=e
                elig*=c.feature_decay;elig+=(1-c.feature_decay)*e
                for a in (qv,qa,elig):np.clip(a,-10,10,out=a)
        return self.probabilities()

    def learn(self,target,mode='target',reward=None):
        if mode not in ('target','shuffled','feedback','feedback_shuffled','readout','frozen'):raise ValueError('Unknown local target mode')
        if type(target) not in (int,np.int32,np.int64) or not 0<=target<self.config.outputs:raise ValueError('Invalid target')
        if mode=='frozen':return super().learn(target,'frozen')
        index=int(self.mod_rng.integers(self.config.outputs)) if mode in ('shuffled','feedback_shuffled') else target
        local_error=(self.target_rates[self.rows,index]-self.target_rates[self.rows]@self.probabilities()
          if mode in ('feedback','feedback_shuffled') else self.target_rates[self.rows,index]-self.features[self.rows])
        loss=super().learn(target,'readout',reward)
        if mode in ('target','shuffled','feedback','feedback_shuffled'):
            c=self.config
            for key in ('in','rec'):
                weight=getattr(self,'w_'+key);stable=getattr(self,'stable_'+key)
                delta=c.hidden_lr*local_error[:,None]*getattr(self,'elig_'+key)-c.anchor_strength*(weight[self.rows]-stable[self.rows])
                weight[self.rows]+=np.clip(delta,-c.update_clip,c.update_clip)
                np.clip(weight,-c.weight_clip,c.weight_clip,out=weight)
        return loss

    def features_many(self,sequences,lesion=None):
        x=np.asarray(sequences,np.float32)
        if x.ndim!=3 or x.shape[2]!=self.config.inputs or not np.isfinite(x).all():raise ValueError('Invalid sequences')
        if lesion not in (None,'permute','reset'):raise ValueError('Invalid lesion')
        if lesion=='permute':
            x=x.copy();x[:,:10]=x[:,np.random.default_rng(91031).permutation(10)]
        c=self.config;v=np.zeros((len(x),c.hidden),np.float32);a=v.copy();z=v.copy();f=v.copy()
        for t in range(x.shape[1]):
            if lesion=='reset':v.fill(0);a.fill(0);z.fill(0);f.fill(0)
            v=c.leak*v+x[:,t]@self.w_in.T+np.sum(self.w_rec[None,:,:]*z[:,self.indices],axis=2)
            limit=c.threshold+c.adaptation_strength*a;z=(v>=limit).astype(np.float32)
            v-=z*limit;a=c.adaptation_decay*a+z;f=c.feature_decay*f+(1-c.feature_decay)*z
        return f

    def arrays(self):
        return {**super().arrays(),**{name:getattr(self,name) for name in self.extra_names}}

    def dumps(self):
        arrays=self.arrays();meta={'format':'scarx-local-target-v1','config':asdict(self.config),'updates':self.updates,
          'reward_mean':self.reward_mean,'rng':self.mod_rng.bit_generator.state,
          'sha256':{k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in arrays.items()}}
        stream=io.BytesIO();np.savez(stream,metadata=np.array(json.dumps(meta)),**arrays);return stream.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>20_000_000:raise ValueError('Oversized local target checkpoint')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(x.file_size for x in archive.infolist())>20_000_000 or archive.getinfo('metadata.npy').file_size>65536 or len(archive.namelist())!=len(set(archive.namelist())):raise ValueError('Invalid checkpoint members')
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['format']!='scarx-local-target-v1':raise ValueError('Incompatible format')
            m=cls(Config(**meta['config']))
            if set(z.files)!=set(m.arrays())|{'metadata'}:raise ValueError('Checkpoint fields')
            for name,template in m.arrays().items():
                a=z[name]
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all() or hashlib.sha256(a.tobytes()).hexdigest()!=meta['sha256'][name]:raise ValueError('Corrupt checkpoint array')
                template[:]=a
            m.updates=meta['updates'];m.reward_mean=meta['reward_mean'];m.mod_rng.bit_generator.state=meta['rng']
            if type(m.updates) is not int or m.updates<0 or not np.isfinite(m.reward_mean) or np.any(m.indices<0) or np.any(m.indices>=m.config.hidden) or np.any(m.rows<0) or np.any(m.rows>=m.config.hidden):raise ValueError('Invalid checkpoint state')
            if np.any((m.target_rates!=np.float32(.02))&(m.target_rates!=np.float32(.35))):raise ValueError('Invalid fixed target code')
        return m
