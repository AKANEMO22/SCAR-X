"""Independent forward-only anchored-surrogate finite-difference oracle.

Only engineering validation. Perturbed continuous outputs are never model
predictions or neural training; presynaptic spikes are externally held fixed.
"""
import numpy as np
from ..neural.core import SpikingCore


def fixed_presynaptic_tape(core,sequence):
    c=core.config;m=SpikingCore.loads(core.dumps());m.reset_state();presyn=[];spikes=[]
    for x in sequence:
        presyn.append(np.concatenate((np.broadcast_to(x,(c.hidden,c.inputs)),m.spikes[m.indices]),axis=1).astype(np.float64))
        m.step(x,False);spikes.append(m.spikes.copy())
    return np.stack(presyn),np.stack(spikes),m


def scalar_anchored_feature(weights,presyn,observed,baseline_weights,config):
    """Forward scalar recurrence; no sensitivity recursion from candidate."""
    c=config;v=a=0.;margins=[]
    for x,z in zip(presyn,observed):
        u=c.leak*v+float(np.dot(baseline_weights,x));theta=c.threshold+c.adaptation_strength*a
        margins.append((u-theta)/.4);v=u-float(z)*theta;a=c.adaptation_decay*a+float(z)
    v=a=f=0.
    for x,z,base in zip(presyn,observed,margins):
        u=c.leak*v+float(np.dot(weights,x));theta=c.threshold+c.adaptation_strength*a
        actual=1/(1+np.exp(-np.clip((u-theta)/.4,-20,20)));anchor=1/(1+np.exp(-np.clip(base,-20,20)))
        local=float(z)+actual-anchor;v=u-local*theta;a=c.adaptation_decay*a+local;f=c.feature_decay*f+(1-c.feature_decay)*local
    return f


def finite_difference_check(model,sequence,neurons=(0,7,19,37),epsilon=1e-5):
    presyn,spikes,baseline=fixed_presynaptic_tape(model.core,sequence);model.begin(True)
    for x in sequence:model.step(x)
    for name in ('voltage','adaptation','spikes','features'):np.testing.assert_array_equal(getattr(model.core,name),getattr(baseline,name))
    errors=[];pairs=[]
    for j in neurons:
        weights=np.concatenate((model.core.w_in[j],model.core.w_rec[j])).astype(np.float64);numeric=[]
        for i in range(len(weights)):
            plus=weights.copy();minus=weights.copy();plus[i]+=epsilon;minus[i]-=epsilon
            numeric.append((scalar_anchored_feature(plus,presyn[:,j],spikes[:,j],weights,model.core.config)-scalar_anchored_feature(minus,presyn[:,j],spikes[:,j],weights,model.core.config))/(2*epsilon))
        numeric=np.asarray(numeric);trace=model.e_f[j].astype(np.float64);np.testing.assert_allclose(trace,numeric,atol=1e-5,rtol=1e-3)
        errors.extend(np.abs(trace-numeric).tolist());pairs.append({'neuron':j,'synapses':len(weights),'max_absolute_error':float(np.max(np.abs(trace-numeric)))})
    return {'finite_difference_synapses':len(errors),'max_absolute_error':max(errors),'neuron_rows':pairs,'forward_state_exact':True,
        'oracle_scope':'Anchored smooth own-neuron surrogate, fixed presynaptic tape; NOT finite derivative of hardspikes/fullnetwork.'}
