"""Own-neuron pooledimagecache sensitivity oracle, never neuralinference."""
import numpy as np


def anchored_pooled(weights,baseline_weights,presyn,observed,config):
    c=config;v=a=0.;anchors=[]
    for x,z in zip(presyn,observed):
        u=c.leak*v+float(baseline_weights@x);theta=c.threshold+c.adaptation_strength*a;anchors.append((u-theta)/.4)
        v=u-float(z)*theta;a=c.adaptation_decay*a+float(z)
    v=a=f=0.;cache=[]
    for tick,(x,z,anchor) in enumerate(zip(presyn,observed,anchors)):
        u=c.leak*v+float(weights@x);theta=c.threshold+c.adaptation_strength*a
        local=float(z)+1/(1+np.exp(-np.clip((u-theta)/.4,-20,20)))-1/(1+np.exp(-np.clip(anchor,-20,20)))
        v=u-local*theta;a=c.adaptation_decay*a+local;f=c.feature_decay*f+(1-c.feature_decay)*local
        if 16<=tick<=31:cache.append(f)
    return .5*(float(np.mean(cache))+f)


def check_pooled_sensitivities(model,question,image):
    presyn=[];spikes=[];actual=model._tick;model.training_trace=True
    def record(x):
        c=model.core.config;presyn.append(np.concatenate((np.broadcast_to(x,(48,c.inputs)),model.core.spikes[model.core.indices]),axis=1).astype(np.float64))
        actual(x);spikes.append(model.core.spikes.copy())
    model._tick=record
    try:model.prepare(question,image);model.word_feature(1,0)
    finally:del model._tick;model.training_trace=False
    presyn=np.stack(presyn);spikes=np.stack(spikes);c=model.core.config
    assert len(presyn)==36 and model.cache_count==16
    indices=list(np.linspace(0,c.inputs-1,8,dtype=int))+list(range(c.inputs,c.inputs+8));errors=[]
    for j in (0,7,19,37):
        weights=np.concatenate((model.core.w_in[j],model.core.w_rec[j])).astype(np.float64)
        for i in indices:
            plus=weights.copy();minus=weights.copy();plus[i]+=1e-5;minus[i]-=1e-5
            numeric=(anchored_pooled(plus,weights,presyn[:,j],spikes[:,j],c)-anchored_pooled(minus,weights,presyn[:,j],spikes[:,j],c))/(2e-5)
            trace=float(model.pooled_e[j,i]);np.testing.assert_allclose(trace,numeric,atol=1e-5,rtol=1e-3);errors.append(abs(trace-numeric))
    return {'synapses':len(errors),'max_absolute_error':max(errors),'cache_ticks':16,'input_ticks':36,
        'scope':'Anchoredsmooth own-neuron derivatives for SAME32 pooledfeature with fixedpresynaptictape; NOT hardspike/fullnetworkgradient.'}
