"""Phase 2.1: stochastic ALIF, local score traces and metaplasticity.

Hidden learning uses only presynaptic filtered activity, the postsynaptic
Bernoulli residual and a scalar return. No reverse graph, feedback matrix or
backward pass. Clipping/block updates make this a practical heuristic.
"""
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from .core import Config, SpikingCore


class PlasticCore(SpikingCore):
    extra_names = ("voltage_pre","importance_in","importance_rec","importance_out","importance_bias")

    def __init__(self,config,rule="score",consolidation="none",shuffle=False):
        if rule not in ("readout","stdp","score","score_meta","frozen") or consolidation not in ("none","ema","meta"):
            raise ValueError("Invalid local rule/consolidation")
        super().__init__(config)
        self.rule,self.consolidation,self.shuffle = rule,consolidation,shuffle
        self.rng=np.random.default_rng(config.seed+300_000)
        self.mod_rng=np.random.default_rng(config.seed+400_000)
        self.voltage_pre=np.zeros(config.inputs+config.hidden,np.float32)
        for key,weight in (("in","w_in"),("rec","w_rec"),("out","w_out"),("bias","bias")):
            setattr(self,"importance_"+key,np.zeros_like(getattr(self,weight)))

    def reset_state(self):
        super().reset_state()
        self.voltage_pre=np.zeros(self.config.inputs+self.config.hidden,np.float32)

    def step(self,x,track=True):
        c=self.config; x=np.asarray(x,dtype=np.float32)
        if x.shape!=(c.inputs,) or not np.isfinite(x).all(): raise ValueError("Invalid input")
        old=self.spikes.copy()
        self.voltage=c.leak*self.voltage+self.w_in@x+np.sum(self.w_rec*old[self.indices],axis=1)
        limit=c.threshold+c.adaptation_strength*self.adaptation
        probability=1/(1+np.exp(-np.clip((self.voltage-limit)/.4,-20,20)))
        self.spikes=(self.rng.random(c.hidden)<probability).astype(np.float32)
        self.voltage-=self.spikes*limit
        self.adaptation=c.adaptation_decay*self.adaptation+self.spikes
        self.features=c.feature_decay*self.features+(1-c.feature_decay)*self.spikes
        if track:
            r=self.rows
            self.voltage_pre*=c.leak
            self.voltage_pre[:c.inputs]+=x
            self.voltage_pre[c.inputs:]+=old
            if self.rule=="stdp":
                self.elig_in*=c.trace_decay; self.elig_rec*=c.trace_decay
                self.elig_in+=self.spikes[r,None]*self.pre[None,:c.inputs]-.5*self.post[r,None]*x[None,:]
                self.elig_rec+=self.spikes[r,None]*self.pre[c.inputs:][self.indices[r]]-.5*self.post[r,None]*old[self.indices[r]]
            else:
                score=(self.spikes[r]-probability[r])/.4
                self.elig_in+=score[:,None]*self.voltage_pre[None,:c.inputs]
                self.elig_rec+=score[:,None]*self.voltage_pre[c.inputs:][self.indices[r]]
            np.clip(self.elig_in,-10,10,out=self.elig_in); np.clip(self.elig_rec,-10,10,out=self.elig_rec)
            self.pre*=c.trace_decay; self.pre[:c.inputs]+=x; self.pre[c.inputs:]+=old
            self.post=c.trace_decay*self.post+self.spikes
        return self.probabilities()

    def _update(self,key,signal,lr,rows=None):
        name="bias" if key=="bias" else "w_"+key
        weight=getattr(self,name); stable=getattr(self,"stable_"+key); importance=getattr(self,"importance_"+key)
        select=slice(None) if rows is None else rows
        meta=self.consolidation=="meta" or self.rule=="score_meta"
        omega=importance[select]
        if meta:
            delta=lr*signal/(1+5*omega)-.005*omega*(weight[select]-stable[select])
        else:
            anchor=.0001 if self.consolidation=="ema" else 0.
            delta=lr*signal-anchor*(weight[select]-stable[select])
        weight[select]+=np.clip(delta,-self.config.update_clip,self.config.update_clip)
        np.clip(weight,-self.config.weight_clip,self.config.weight_clip,out=weight)
        # Importance is local activity/update magnitude, not a global loss gradient.
        magnitude=np.abs(signal)
        importance[select]=np.minimum(20,omega+.001*magnitude/(float(magnitude.mean())+1e-6))

    def learn(self,target,mode=None,reward=None):
        if not 0<=target<self.config.outputs or (reward is not None and not np.isfinite(reward)):
            raise ValueError("Invalid target/reward")
        p=self.probabilities(); loss=-float(np.log(max(float(p[target]),1e-9)))
        if self.rule=="frozen": return loss
        error=-p.copy(); error[target]+=1
        raw=-loss if reward is None else float(reward)
        advantage=float(np.clip(raw-self.reward_mean,-2,2))
        self.reward_mean=.95*self.reward_mean+.05*raw
        if self.shuffle: advantage*=float(self.mod_rng.choice([-1,1]))
        self._update("out",error[:,None]*self.features[None,:],self.config.readout_lr)
        self._update("bias",error,self.config.readout_lr)
        if self.rule not in ("readout","frozen"):
            for key,trace in (("in",self.elig_in),("rec",self.elig_rec)):
                self._update(key,advantage*trace,self.config.hidden_lr,self.rows)
        self.updates+=1
        return loss

    def consolidate(self):
        if self.consolidation!="none" or self.rule=="score_meta": super().consolidate()

    def features_many(self,sequences,seed=901,lesion=None):
        """Stateless batch evaluation, common random numbers, no RNG/trace mutation."""
        x=np.asarray(sequences,dtype=np.float32)
        if x.ndim!=3 or x.shape[2]!=self.config.inputs: raise ValueError("Invalid batch")
        c=self.config; b,t,_=x.shape
        v=np.zeros((b,c.hidden),np.float32); a=v.copy(); z=v.copy(); f=v.copy()
        rng=np.random.default_rng(seed)
        if lesion=="permute":
            x=x.copy(); x[:,:-1]=x[:,np.random.default_rng(55).permutation(t-1)]
        for step in range(t):
            if lesion=="reset": v.fill(0); a.fill(0); z.fill(0); f.fill(0)
            v=c.leak*v+x[:,step]@self.w_in.T+np.sum(self.w_rec[None,:,:]*z[:,self.indices],axis=2)
            limit=c.threshold+c.adaptation_strength*a
            p=1/(1+np.exp(-np.clip((v-limit)/.4,-20,20)))
            z=(rng.random((b,c.hidden))<p).astype(np.float32)
            v-=z*limit; a=c.adaptation_decay*a+z
            f=c.feature_decay*f+(1-c.feature_decay)*z
        return f

    def predict_many(self,sequences,seed=901,lesion=None):
        f=self.features_many(sequences,seed,lesion)
        return np.argmax(f@self.w_out.T+self.bias,axis=1)

    def arrays(self):
        base=super().arrays()
        base.update({k:getattr(self,k) for k in self.extra_names})
        return base

    def dumps(self):
        """Serialize every plasticity/RNG field for an enclosing transaction."""
        arrays=self.arrays()
        meta={"format":"plasticity-v1","config":asdict(self.config),"rule":self.rule,"consolidation":self.consolidation,"shuffle":self.shuffle,
              "updates":self.updates,"reward_mean":self.reward_mean,"rng":self.rng.bit_generator.state,"mod_rng":self.mod_rng.bit_generator.state,
              "sha256":{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
        stream=io.BytesIO()
        np.savez(stream,metadata=np.array(json.dumps(meta)),**arrays)
        return stream.getvalue()

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        payload=self.dumps()
        fd,tmp=tempfile.mkstemp(dir=path.parent,suffix=".tmp")
        try:
            with os.fdopen(fd,"wb") as f:
                f.write(payload); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    @classmethod
    def load(cls,path):
        path=Path(path)
        if path.stat().st_size>100_000_000: raise ValueError("Oversized checkpoint")
        return cls.loads(path.read_bytes())

    @classmethod
    def loads(cls,payload):
        if len(payload)>100_000_000: raise ValueError("Oversized checkpoint")
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if sum(i.file_size for i in z.infolist())>100_000_000 or z.getinfo("metadata.npy").file_size>65536: raise ValueError("Oversized checkpoint")
            if len(z.namelist())!=len(set(z.namelist())): raise ValueError("Duplicate checkpoint members")
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            meta=json.loads(str(z["metadata"]))
            if meta["format"]!="plasticity-v1": raise ValueError("Format")
            m=cls(Config(**meta["config"]),meta["rule"],meta["consolidation"],meta["shuffle"])
            if set(z.files)!=set(m.arrays())|{"metadata"}: raise ValueError("Fields")
            for k,template in m.arrays().items():
                a=z[k]
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all() or hashlib.sha256(a.tobytes()).hexdigest()!=meta["sha256"][k]: raise ValueError("Corrupt array")
                template[:]=a
            m.updates=int(meta["updates"]); m.reward_mean=float(meta["reward_mean"])
            m.rng.bit_generator.state=meta["rng"]; m.mod_rng.bit_generator.state=meta["mod_rng"]
            if m.updates<0 or not np.isfinite(m.reward_mean) or np.any(m.indices<0) or np.any(m.indices>=m.config.hidden) or np.any(m.rows<0) or np.any(m.rows>=m.config.hidden): raise ValueError("Invalid state")
        return m
