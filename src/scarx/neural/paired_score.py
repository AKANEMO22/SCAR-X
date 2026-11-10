"""Research-only paired local likelihood eligibility, no backward/error matrix."""
from dataclasses import asdict
import hashlib,io,json,zipfile
import numpy as np
from .core import SpikingCore,Config


class PairedScoreCore(SpikingCore):
    checkpoint_format='scarx-paired-score-v2'
    extra_names=('voltage_pre','moment_in','variance_in','moment_rec','variance_rec','row_updates','head_mean','head_scale')

    def __init__(self,config,pairing='independent',optimizer='adam',clipped=False,frozen=False,shuffle=False):
        if config.neuron!='alif' or config.hidden>128 or pairing not in ('independent','antithetic') or optimizer not in ('adam','sgd'):raise ValueError('Invalid paired score configuration')
        super().__init__(config)
        self.pairing=pairing;self.optimizer=optimizer;self.clipped=bool(clipped);self.frozen=bool(frozen);self.shuffle=bool(shuffle)
        self.rng=np.random.default_rng(config.seed+5100000);self.mod_rng=np.random.default_rng(config.seed+5200000)
        for name in ('in','rec'):
            setattr(self,'moment_'+name,np.zeros_like(getattr(self,'w_'+name)))
            setattr(self,'variance_'+name,np.zeros_like(getattr(self,'w_'+name)))
        self.row_updates=np.zeros(config.hidden,np.int64);self.hidden_updates=0;self.ticks=0
        self.head_mean=np.zeros(config.hidden,np.float32);self.head_scale=np.ones(config.hidden,np.float32);self.calibrated=False

    def reset_state(self):
        super().reset_state();self.voltage_pre=np.zeros(self.config.inputs+self.config.hidden,np.float32)

    def transform(self,f):return np.clip((f-self.head_mean)/self.head_scale,-5,5)

    def probabilities(self,features=None):
        f=self.features if features is None else features
        logits=self.transform(f)@self.w_out.T+self.bias;values=np.exp(logits-logits.max(axis=-1,keepdims=True))
        return values/values.sum(axis=-1,keepdims=True)

    def calibrate(self,features):
        f=np.asarray(features,np.float32)
        if self.calibrated or f.ndim!=2 or f.shape[1]!=self.config.hidden or len(f)<2 or not np.isfinite(f).all():raise ValueError('Invalid/frozen calibration')
        self.head_mean[:]=f.mean(0);self.head_scale[:]=np.maximum(f.std(0,ddof=1),.05);self.calibrated=True

    def step(self,x,uniform=None,track=True):
        c=self.config;x=np.asarray(x,np.float32)
        if x.shape!=(c.inputs,) or not np.isfinite(x).all():raise ValueError('Invalid input')
        u=self.rng.random(c.hidden) if uniform is None else np.asarray(uniform)
        if u.shape!=(c.hidden,) or not np.isfinite(u).all() or np.any((u<0)|(u>1)):raise ValueError('Invalid uniform noise')
        old=self.spikes.copy();self.voltage=c.leak*self.voltage+self.w_in@x+np.sum(self.w_rec*old[self.indices],axis=1)
        limit=c.threshold+c.adaptation_strength*self.adaptation;logit=(self.voltage-limit)/.4
        p=1/(1+np.exp(-np.clip(logit,-20,20)));self.spikes=(u<p).astype(np.float32)
        self.voltage-=self.spikes*limit;self.adaptation=c.adaptation_decay*self.adaptation+self.spikes
        self.features=c.feature_decay*self.features+(1-c.feature_decay)*self.spikes
        if track:
            self.voltage_pre*=c.leak;self.voltage_pre[:c.inputs]+=x;self.voltage_pre[c.inputs:]+=old
            r=self.rows;score=(self.spikes[r]-p[r])/.4*(np.abs(logit[r])<20)
            self.elig_in+=score[:,None]*self.voltage_pre[None,:c.inputs]
            self.elig_rec+=score[:,None]*self.voltage_pre[c.inputs:][self.indices[r]]
            if self.clipped:
                np.clip(self.elig_in,-10,10,out=self.elig_in);np.clip(self.elig_rec,-10,10,out=self.elig_rec)
        self.ticks+=1;return self.probabilities()

    def rollout(self,sequence,uniforms,rows):
        x=np.asarray(sequence,np.float32);u=np.asarray(uniforms)
        if x.ndim!=2 or x.shape[1]!=self.config.inputs or not 1<=len(x)<=128 or u.shape!=(len(x),self.config.hidden):raise ValueError('Invalid bounded trajectory')
        self.reset_state();self.rows[:]=rows
        for value,noise in zip(x,u):self.step(value,noise,True)
        return self.features.copy(),self.elig_in.copy(),self.elig_rec.copy()

    def update_hidden(self,signals):
        if self.frozen:return
        r=self.rows;self.row_updates[r]+=1
        sign=float(self.mod_rng.choice((-1,1))) if self.shuffle else 1.
        for name,g in zip(('in','rec'),signals):
            g=np.asarray(g,np.float32)*sign;weight=getattr(self,'w_'+name)
            if g.shape!=getattr(self,'elig_'+name).shape or not np.isfinite(g).all():raise ValueError('Invalid local signal')
            if self.optimizer=='adam':
                m=getattr(self,'moment_'+name);v=getattr(self,'variance_'+name)
                m[r]=.9*m[r]+.1*g;v[r]=.999*v[r]+.001*g*g
                estimate=(m[r]/(1-np.power(.9,self.row_updates[r,None])))/(np.sqrt(v[r]/(1-np.power(.999,self.row_updates[r,None])))+1e-8)
            else:estimate=g
            weight[r]+=np.clip(self.config.hidden_lr*estimate,-self.config.update_clip,self.config.update_clip)
            np.clip(weight,-self.config.weight_clip,self.config.weight_clip,out=weight)
        self.hidden_updates+=1

    def learn_example(self,sequence,target):
        if not self.calibrated or type(target) not in (int,np.int32,np.int64) or not 0<=target<self.config.outputs:raise ValueError('Invalid target/calibration')
        c=self.config;x=np.asarray(sequence,np.float32);features=[];losses=[]
        if x.ndim!=2 or not 1<=len(x)<=128 or x.shape[1]!=c.inputs or not np.isfinite(x).all():raise ValueError('Invalid episode')
        blocks=(c.hidden+c.plastic_rows-1)//c.plastic_rows
        for block in range(blocks):
            rows=(np.arange(c.plastic_rows)+(self.updates+block)*c.plastic_rows)%c.hidden
            u=self.rng.random((len(x),c.hidden));v=self.rng.random(u.shape)
            first=self.rollout(x,u,rows);second=self.rollout(x,1-u if self.pairing=='antithetic' else v,rows)
            rewards=np.array([np.log(max(float(self.probabilities(a[0])[target]),1e-12)) for a in (first,second)])
            advantage=rewards-self.reward_mean
            if self.clipped:advantage=np.clip(advantage,-2,2)
            signals=[.5*(advantage[0]*first[i]+advantage[1]*second[i]) for i in (1,2)]
            self.update_hidden(signals);self.reward_mean=.95*self.reward_mean+.05*float(rewards.mean())
            features.extend((first[0],second[0]));losses.extend((-rewards).tolist())
        f=np.mean(features,axis=0).astype(np.float32);p=self.probabilities(f);error=-p.copy();error[target]+=1
        self.w_out+=np.clip(c.readout_lr*error[:,None]*self.transform(f)[None,:],-c.update_clip,c.update_clip)
        self.bias+=np.clip(c.readout_lr*error,-c.update_clip,c.update_clip)
        np.clip(self.w_out,-c.weight_clip,c.weight_clip,out=self.w_out);np.clip(self.bias,-c.weight_clip,c.weight_clip,out=self.bias)
        self.updates+=1
        return {'loss':-float(np.log(max(float(p[target]),1e-12))),'trajectory_loss':float(np.mean(losses)),
            'rollouts':blocks*2,'ticks':len(x)*blocks*2,'hidden_updates':0 if self.frozen else blocks}

    def features_many(self,sequences,seed=991,lesion=None,samples=8):
        x=np.asarray(sequences,np.float32);c=self.config
        if x.ndim!=3 or not 1<=len(x)<=2048 or not 1<=x.shape[1]<=128 or x.shape[2]!=c.inputs or not np.isfinite(x).all() or not 1<=samples<=16:raise ValueError('Invalid feature batch')
        if lesion not in (None,'permute','reset'):raise ValueError('Invalid lesion')
        if lesion=='permute':x=x.copy();x[:,:-1]=x[:,np.random.default_rng(993).permutation(x.shape[1]-1)]
        output=np.zeros((len(x),c.hidden),np.float32)
        for sample in range(samples):
            rng=np.random.default_rng(seed+sample);v=np.zeros_like(output);a=v.copy();z=v.copy();f=v.copy()
            for tick in range(x.shape[1]):
                if lesion=='reset':v.fill(0);a.fill(0);z.fill(0);f.fill(0)
                v=c.leak*v+x[:,tick]@self.w_in.T+np.sum(self.w_rec[None,:,:]*z[:,self.indices],axis=2)
                limit=c.threshold+c.adaptation_strength*a;p=1/(1+np.exp(-np.clip((v-limit)/.4,-20,20)))
                z=(rng.random(z.shape)<p).astype(np.float32);v-=z*limit;a=c.adaptation_decay*a+z;f=c.feature_decay*f+(1-c.feature_decay)*z
            output+=f/samples
        return output

    def arrays(self):return {**super().arrays(),**{name:getattr(self,name) for name in self.extra_names}}

    def dumps(self):
        arrays=self.arrays();meta={'format':self.checkpoint_format,'config':asdict(self.config),
            **{n:getattr(self,n) for n in ('pairing','optimizer','clipped','frozen','shuffle','calibrated','updates','hidden_updates','ticks','reward_mean')},
            'rng':self.rng.bit_generator.state,'mod_rng':self.mod_rng.bit_generator.state,
            'sha256':{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in arrays.items()}}
        stream=io.BytesIO();np.savez(stream,metadata=np.array(json.dumps(meta)),**arrays);return stream.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>20000000:raise ValueError('Oversized paired score checkpoint')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if sum(i.file_size for i in z.infolist())>20000000 or z.getinfo('metadata.npy').file_size>65536 or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Invalid archive')
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['format']!=cls.checkpoint_format:raise ValueError('Paired score format')
            for name in ('clipped','frozen','shuffle','calibrated'):
                if type(meta[name]) is not bool:raise ValueError('Invalid policy')
            m=cls(Config(**meta['config']),meta['pairing'],meta['optimizer'],meta['clipped'],meta['frozen'],meta['shuffle'])
            if set(z.files)!=set(m.arrays())|{'metadata'}:raise ValueError('Invalid fields')
            for name,template in m.arrays().items():
                a=z[name]
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all() or hashlib.sha256(a.tobytes()).hexdigest()!=meta['sha256'][name]:raise ValueError('Corrupt array')
                template[:]=a
            for name in ('updates','hidden_updates','ticks'):
                if type(meta[name]) is not int or meta[name]<0:raise ValueError('Invalid counter')
                setattr(m,name,meta[name])
            m.calibrated=meta['calibrated'];m.reward_mean=meta['reward_mean'];m.rng.bit_generator.state=meta['rng'];m.mod_rng.bit_generator.state=meta['mod_rng']
            if not np.isfinite(m.reward_mean) or np.any(m.row_updates<0) or np.any(m.head_scale<.05) or np.any(m.variance_in<0) or np.any(m.variance_rec<0) or np.any(m.indices<0) or np.any(m.indices>=m.config.hidden) or np.any(m.rows<0) or np.any(m.rows>=m.config.hidden):raise ValueError('Invalid state')
        return m
