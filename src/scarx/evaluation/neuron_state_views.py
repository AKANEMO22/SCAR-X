"""Read-only ALIF state-channel diagnostic, no new dynamics or source learning."""
import numpy as np

VIEWS=('filtered_spikes','last_spikes','membrane_after_reset','adaptation','relative_membrane')


def state_features(model,sequences,seed,samples=8):
    x=np.asarray(sequences,np.float32);c=model.config
    if x.ndim!=3 or not 1<=len(x)<=2048 or x.shape[2]!=c.inputs or not 1<=x.shape[1]<=128 or not np.isfinite(x).all() or type(samples) is not int or not 1<=samples<=16:raise ValueError('Invalid bounded state batch')
    shape=(len(x),c.hidden);out={k:np.zeros(shape,np.float32) for k in VIEWS}
    for sample in range(samples):
        rng=np.random.default_rng(seed+sample);v=np.zeros(shape,np.float32);a=v.copy();z=v.copy();f=v.copy()
        for tick in range(x.shape[1]):
            v=c.leak*v+x[:,tick]@model.w_in.T+np.sum(model.w_rec[None,:,:]*z[:,model.indices],axis=2)
            limit=c.threshold+c.adaptation_strength*a;p=1/(1+np.exp(-np.clip((v-limit)/.4,-20,20)))
            z=(rng.random(z.shape)<p).astype(np.float32);v-=z*limit;a=c.adaptation_decay*a+z;f=c.feature_decay*f+(1-c.feature_decay)*z
        actual={'filtered_spikes':f,'last_spikes':z,'membrane_after_reset':v,'adaptation':a,
            'relative_membrane':v/np.maximum(c.threshold+c.adaptation_strength*a,.001)}
        for key,value in actual.items():out[key]+=value/samples
    return out


def local_head(train_x,train_y,test_x,test_y,classes=4,seed=2003):
    a=np.asarray(train_x,np.float32);b=np.asarray(test_x,np.float32)
    mean=a.mean(0);scale=np.maximum(a.std(0,ddof=1),.05);a=np.clip((a-mean)/scale,-5,5);b=np.clip((b-mean)/scale,-5,5)
    w=np.zeros((classes,a.shape[1]),np.float32);bias=np.zeros(classes,np.float32);rng=np.random.default_rng(seed);losses=[]
    for _ in range(5):
        for index in rng.permutation(len(a)):
            logits=w@a[index]+bias;p=np.exp(logits-logits.max());p/=p.sum();target=int(train_y[index]);losses.append(-float(np.log(max(float(p[target]),1e-12))))
            error=-p;error[target]+=1;w+=np.clip(.08*error[:,None]*a[index][None,:],-.025,.025);bias+=np.clip(.08*error,-.025,.025)
            np.clip(w,-3,3,out=w);np.clip(bias,-3,3,out=bias)
    predictions=(b@w.T+bias).argmax(1)
    return {'accuracy':float(np.mean(predictions==test_y)),'predictions':predictions.tolist(),'updates':5*len(a),'mean_loss':float(np.mean(losses))}
