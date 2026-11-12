"""Independent CONTROL checks: frozen parameters, no labels/learning/DEV.

Finite differences here apply to an explicitly anchored smooth surrogate
extension of a fixed hard-spike trajectory, never to the discontinuous spike.
"""
import copy
import numpy as np
from scarx.neural.core import Config,SpikingCore
from scarx.evaluation.readout_eligibility import spike_slope
from scarx.evaluation.literature_braintrace import full_surrogate_sensitivity


def anchored_trajectory(core,sequence):
    m=copy.deepcopy(core);m.reset_state();out=[];c=m.config
    for row in sequence:
        old_z=m.spikes.copy()
        theta=c.threshold+c.adaptation_strength*m.adaptation
        u=(c.leak*m.voltage+np.sum(m.w_rec*old_z[m.indices],axis=1))+m.w_in@row
        slope=spike_slope(u-theta)
        m.step(row,track=False)
        out.append((old_z.astype(float),theta.astype(float),slope.astype(float),m.spikes.astype(float)))
    return out


def anchored_feature_delta(core,sequence,perturbation):
    """Independent perturbed graph in float64 around the frozen hard trajectory.

    At each anchor z~=z_hard+psi*delta_margin. Products (z~*theta~) and
    recurrent weight*presynaptic z~ remain nonlinear. Its derivative at zero
    is the prescribed full surrogate graph, including all recurrent paths.
    """
    c=core.config;h=c.hidden;k=c.inputs+c.fan_in
    dw=np.asarray(perturbation,float).reshape(h,k);di=dw[:,:c.inputs];dr=dw[:,c.inputs:]
    dv=np.zeros(h);da=np.zeros(h);dz=np.zeros(h);df=np.zeros(h)
    for row,(oldz,theta,slope,z) in zip(sequence,anchored_trajectory(core,sequence)):
        recurrent=(core.w_rec.astype(float)*dz[core.indices]+dr*oldz[core.indices]+dr*dz[core.indices]).sum(1)
        du=c.leak*dv+di@row.astype(float)+recurrent
        next_dz=slope*(du-c.adaptation_strength*da)
        dv=du-theta*next_dz-c.adaptation_strength*z*da-c.adaptation_strength*da*next_dz
        da=c.adaptation_decay*da+next_dz
        df=c.feature_decay*df+(1-c.feature_decay)*next_dz
        dz=next_dz
    return df


def test_full_oracle_against_independent_whole_sequence_finite_differences():
    c=Config(inputs=2,hidden=3,outputs=2,fan_in=2,plastic_rows=3,seed=112)
    m=SpikingCore(c)
    # Include a self edge explicitly; full oracle must retain it.
    m.indices[:]=np.array([[0,1],[1,2],[2,0]],np.int32)
    m.w_rec[:]=np.array([[.2,-.1],[.15,.3],[-.1,.2]],np.float32)
    x=np.random.default_rng(113).uniform(-.1,1.1,(11,2)).astype(np.float32)
    p=c.hidden*(c.inputs+c.fan_in)
    analytic=full_surrogate_sensitivity(m,x)
    h=1e-6;numeric=np.empty_like(analytic)
    for j in range(p):
        perturb=np.zeros(p);perturb[j]=h
        numeric[:,j]=(anchored_feature_delta(m,x,perturb)-anchored_feature_delta(m,x,-perturb))/(2*h)
    np.testing.assert_allclose(analytic,numeric,rtol=2e-8,atol=2e-9)
    assert any(anchor[-1].sum()>0 for anchor in anchored_trajectory(m,x))
    assert np.any(np.abs(analytic[:,:4])>1e-8)


def test_full_oracle_vector_jvp_and_scalar_vjp_agree_without_training():
    c=Config(inputs=2,hidden=3,outputs=2,fan_in=2,plastic_rows=3,seed=772)
    m=SpikingCore(c);x=np.random.default_rng(773).uniform(-.1,1.2,(9,2)).astype(np.float32)
    e=full_surrogate_sensitivity(m,x);rng=np.random.default_rng(774)
    direction=rng.normal(size=e.shape[1]);signal=rng.normal(size=c.hidden);h=1e-6
    finite=(anchored_feature_delta(m,x,h*direction)-anchored_feature_delta(m,x,-h*direction))/(2*h)
    np.testing.assert_allclose(e@direction,finite,rtol=2e-8,atol=2e-9)
    assert np.isclose(signal@finite,(signal@e)@direction,rtol=2e-8,atol=2e-9)
