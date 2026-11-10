"""Read-only ALIF inference views, for diagnostics rather than serving changes."""
import numpy as np


def inference_features(model,sequences,seed,temperature=.4,samples=8,dynamics=None):
    x=np.asarray(sequences,np.float32);c=model.config
    if x.ndim!=3 or not 1<=len(x)<=2048 or not 1<=x.shape[1]<=128 or x.shape[2]!=c.inputs or not np.isfinite(x).all():raise ValueError('Invalid bounded inference batch')
    if temperature not in (0.,.1,.4) or type(samples) is not int or not 1<=samples<=8:raise ValueError('Invalid inference view')
    leak,adaptation_decay,feature_decay=(c.leak,c.adaptation_decay,c.feature_decay)
    if dynamics is not None:
        if set(dynamics)!=set(('leak','adaptation_decay','feature_decay')):raise ValueError('Invalid dynamics fields')
        values=[np.asarray(dynamics[k],np.float32) for k in ('leak','adaptation_decay','feature_decay')]
        if any(a.shape!=(c.hidden,) or not np.isfinite(a).all() or np.any((a<0)|(a>=1)) for a in values):raise ValueError('Invalid decay vector')
        # Preserve the scalar reference's complement rounding (1-.92) when
        # a vector is merely the original scalar repeated across neurons.
        keys=('leak','adaptation_decay','feature_decay')
        values=[getattr(c,k) if np.all(v==np.float32(getattr(c,k))) else v for k,v in zip(keys,values)]
        leak,adaptation_decay,feature_decay=values
    effective=1 if temperature==0 else samples;output=np.zeros((len(x),c.hidden),np.float32)
    for sample in range(effective):
        rng=np.random.default_rng(seed+sample);v=np.zeros_like(output);a=v.copy();z=v.copy();f=v.copy()
        for tick in range(x.shape[1]):
            v=leak*v+x[:,tick]@model.w_in.T+np.sum(model.w_rec[None,:,:]*z[:,model.indices],axis=2)
            limit=c.threshold+c.adaptation_strength*a
            if temperature==0:z=(v>=limit).astype(np.float32)
            else:
                p=1/(1+np.exp(-np.clip((v-limit)/temperature,-20,20)));z=(rng.random(z.shape)<p).astype(np.float32)
            v-=z*limit;a=adaptation_decay*a+z;f=feature_decay*f+(1-feature_decay)*z
        output+=f/effective
    return output


TIMESCALE_VIEWS=('baseline','leak_spread','leak_mean','adaptation_spread','adaptation_mean','feature_spread','feature_mean','all_spread','all_mean','all_independent_spread')


def timescale_view(config,name,seed):
    if name not in TIMESCALE_VIEWS:raise ValueError('Unknown timescale view')
    keys=('leak','adaptation_decay','feature_decay');result={k:np.full(config.hidden,getattr(config,k),np.float32) for k in keys}
    if name=='baseline':return result
    rng=np.random.default_rng(seed+8600000);shared=rng.permutation(config.hidden)
    quantiles=(np.arange(config.hidden)+.5)/config.hidden;factors=np.exp(np.log(1/8)+quantiles*np.log(64))
    for short,key in zip(('leak','adaptation','feature'),keys):
        if not (name.startswith(short+'_') or name.startswith('all_')):continue
        tau0=-1/np.log(getattr(config,key));tau=tau0*factors
        if name.endswith('_mean'):result[key].fill(np.exp(-1/tau.mean()))
        else:
            order=rng.permutation(config.hidden) if name=='all_independent_spread' else shared
            result[key][:]=np.exp(-1/tau[order])
    return result
