"""Isolated presynaptic normalization control, not deployed neural policy."""
import io,json
import numpy as np


class ConditionedReadout:
    def __init__(self,weights,bias,mode):
        if mode not in ('raw','centered','normalized'):raise ValueError('Invalid conditioning')
        self.weights=np.asarray(weights,np.float32).copy();self.bias=np.asarray(bias,np.float32).copy();self.mode=mode
        if self.weights.ndim!=2 or self.bias.shape!=(self.weights.shape[0],):raise ValueError('Invalid readout shapes')
        self.mean=np.zeros(self.weights.shape[1],np.float64);self.m2=self.mean.copy();self.count=0;self.frozen=False;self.updates=0

    def observe(self,features):
        if self.frozen:raise ValueError('Calibration already frozen')
        f=np.asarray(features,np.float64)
        if f.shape!=self.mean.shape or not np.isfinite(f).all():raise ValueError('Invalid feature')
        self.count+=1;delta=f-self.mean;self.mean+=delta/self.count;self.m2+=delta*(f-self.mean)

    def freeze(self):
        if self.count<2:raise ValueError('Need calibration samples')
        self.frozen=True

    def transform(self,features):
        if not self.frozen:raise ValueError('Freeze training-only statistics first')
        x=np.asarray(features,np.float32)
        if x.shape[-1]!=len(self.mean) or not np.isfinite(x).all():raise ValueError('Invalid features')
        if self.mode=='raw':return x
        f=x-self.mean.astype(np.float32)
        if self.mode=='normalized':f=f/np.maximum(np.sqrt(self.m2/(self.count-1)),.05).astype(np.float32)
        return np.clip(f,-5,5).astype(np.float32)

    def probabilities(self,features,mask=None):
        f=self.transform(features);logits=f@self.weights.T+self.bias
        if mask is not None:
            mask=np.asarray(mask,bool)
            if mask.shape!=(len(self.bias),) or not mask.any():raise ValueError('Invalid output mask')
            logits=np.where(mask,logits,-np.inf)
        values=np.exp(logits-logits.max(axis=-1,keepdims=True));return values/values.sum(axis=-1,keepdims=True)

    def learn(self,features,target,mask=None):
        if not 0<=target<len(self.bias):raise ValueError('Invalid target')
        if mask is not None and not mask[target]:raise ValueError('Target outside output mask')
        p=self.probabilities(features,mask);error=-p.copy();error[target]+=1
        f=self.transform(features)
        self.weights+=np.clip(.08*error[:,None]*f[None,:],-.025,.025)
        self.bias+=np.clip(.08*error,-.025,.025)
        np.clip(self.weights,-3,3,out=self.weights);np.clip(self.bias,-3,3,out=self.bias);self.updates+=1
        return -float(np.log(max(float(p[target]),1e-12)))

    def dumps(self):
        stream=io.BytesIO();np.savez(stream,metadata=np.array(json.dumps({'mode':self.mode,'count':self.count,'frozen':self.frozen,'updates':self.updates})),
          weights=self.weights,bias=self.bias,mean=self.mean,m2=self.m2);return stream.getvalue()

    @classmethod
    def loads(cls,payload):
        # Experimental in-process control only, never exposed as an untrusted
        # serving/checkpoint loader. Production codec would require hardening.
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));m=cls(z['weights'],z['bias'],meta['mode'])
            m.mean[:]=z['mean'];m.m2[:]=z['m2'];m.count=meta['count'];m.frozen=meta['frozen'];m.updates=meta['updates']
        return m
