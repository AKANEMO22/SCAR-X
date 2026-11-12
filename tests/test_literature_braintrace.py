import copy,math
import numpy as np
import pytest
from scarx.neural.core import Config,SpikingCore
from scarx.evaluation.readout_eligibility import spike_slope
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit
from scarx.evaluation.literature_braintrace import BrainTraceDRTRL,local_blocks,full_surrogate_sensitivity,direction_metrics

def cfg():return Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=422)
def sequence():return np.random.default_rng(344).uniform(-.1,1.1,(21,8)).astype(np.float32)

def test_local_block_against_independent_anchored_smooth_finite_differences():
    c=cfg();rng=np.random.default_rng(76)
    for _ in range(12):
        v,a,f=rng.uniform(-.3,1.3,3);current=float(rng.uniform(-1,1));theta=c.threshold+c.adaptation_strength*a
        u=c.leak*v+current;margin=u-theta;hard=float(margin>=0)
        sigmoid=lambda x:1/(1+math.exp(-x/.4))
        base=sigmoid(margin);psi=base*(1-base)/.4
        # Value equals hard forward at anchor, derivative equals logistic surrogate.
        def smooth(state,inp):
            vv,aa,ff=state;th=c.threshold+c.adaptation_strength*aa;uu=c.leak*vv+inp
            z=hard+sigmoid(uu-th)-base
            return np.array([uu-z*th,c.adaptation_decay*aa+z,c.feature_decay*ff+(1-c.feature_decay)*z])
        state=np.array([v,a,f]);eps=1e-6;oracle=np.empty((3,3))
        for j in range(3):
            offset=np.eye(3)[j]*eps;oracle[:,j]=(smooth(state+offset,current)-smooth(state-offset,current))/(2*eps)
        direct=(smooth(state,current+eps)-smooth(state,current-eps))/(2*eps)
        d,b=local_blocks(c,np.array([theta]),np.array([hard]),np.array([psi]))
        np.testing.assert_allclose(d[0],oracle,rtol=3e-7,atol=1e-7)
        np.testing.assert_allclose(b[0],direct,rtol=3e-7,atol=1e-7)

def test_block_adapter_matches_own_trace_when_forward_filter_and_gate_aligned():
    m=BrainTraceDRTRL(cfg());scalar=MatchedLocalCredit(cfg(),'slow_ungated')
    for row in sequence():
        m.step(row);scalar.step(row)
        for key in ('voltage','adaptation','spikes','features'):
            np.testing.assert_array_equal(getattr(m.core,key),getattr(scalar.core,key))
        for key in ('e_v','e_a','e_f'):
            np.testing.assert_allclose(getattr(m,key),getattr(scalar,key),rtol=1e-4,atol=5e-6)
    assert m.gate_mode=='none' and np.all(m.gamma_e_vec==cfg().feature_decay)
    assert m.memory_bytes()==scalar.memory_bytes()

def test_one_forward_and_no_new_persistent_trace_arrays(monkeypatch):
    m=BrainTraceDRTRL(cfg());count=[0];original=m.core.step
    def step(*a,**kw):count[0]+=1;return original(*a,**kw)
    monkeypatch.setattr(m.core,'step',step)
    for row in sequence():m.step(row)
    assert count[0]==21
    assert np.isin(m.core.spikes,[0,1]).all()
    own=MatchedLocalCredit(cfg())
    assert {k for k,v in vars(m).items() if isinstance(v,np.ndarray)}=={k for k,v in vars(own).items() if isinstance(v,np.ndarray)}

@pytest.mark.parametrize('condition',['learned_mtsg','shuffled_signal','readout_only','zero_signal','frozen_hidden'])
def test_source_readout_optimizer_and_rng_remain_paired(condition):
    m=BrainTraceDRTRL(cfg(),condition)
    arm={'learned_mtsg':'multi_gated','shuffled_signal':'multi_gated_shuffle','readout_only':'multi_gated_readout_only',
         'zero_signal':'multi_gated_zero','frozen_hidden':'multi_gated_frozen'}[condition]
    own=MatchedLocalCredit(cfg(),arm);m.calibrated=own.calibrated=True
    assert m.learn_episode(sequence(),1)==own.learn_episode(sequence(),1)
    np.testing.assert_array_equal(m.core.w_out,own.core.w_out)
    np.testing.assert_array_equal(m.core.bias,own.core.bias)
    np.testing.assert_array_equal(m.f_mean,own.f_mean);np.testing.assert_array_equal(m.f_var,own.f_var)
    assert m.signal_rng.bit_generator.state==own.signal_rng.bit_generator.state
    if condition in ('readout_only','zero_signal','frozen_hidden'):
        np.testing.assert_array_equal(m.core.w_in,m.w_in_0)
        assert not m.hidden_updates

def test_full_surrogate_oracle_reduces_to_local_when_recurrent_weights_zero():
    m=BrainTraceDRTRL(cfg());m.core.w_rec.fill(0)
    full=full_surrogate_sensitivity(m.core,sequence())
    for row in sequence():m.step(row)
    h=cfg().hidden;k=16
    for j in range(h):np.testing.assert_allclose(full[j,j*k:(j+1)*k],m.e_f[j],rtol=1e-4,atol=4e-6)
    # With no recurrent mixing, owned rows are the only nonzero sensitivities.
    for j in range(h):
        other=full[j].copy();other[j*k:(j+1)*k]=0
        assert not np.any(other)

def test_full_oracle_retains_omitted_recurrent_paths_and_leaves_weights_untouched():
    m=BrainTraceDRTRL(cfg());before=m.core.fingerprint();blob=m.core.dumps()
    full=full_surrogate_sensitivity(m.core,sequence())
    assert before==m.core.fingerprint() and blob==m.core.dumps()
    for row in sequence():m.step(row)
    signal=np.random.default_rng(334).normal(size=48)
    local=(signal[:,None]*m.e_f).ravel();exact=signal@full
    assert not np.allclose(local,exact,rtol=1e-4,atol=1e-6)
    metrics=direction_metrics(local,exact)
    assert -1<=metrics['cosine']<=1 and 0<=metrics['sign_agreement']<=1
    assert metrics['norm_ratio']>0

def test_oracle_bounds_and_adapter_layout():
    with pytest.raises(ValueError):BrainTraceDRTRL(Config(inputs=4,outputs=2,hidden=48,fan_in=8,plastic_rows=12))
    with pytest.raises(ValueError):full_surrogate_sensitivity(SpikingCore(cfg()),np.ones((257,8),np.float32))
    with pytest.raises(ValueError):BrainTraceDRTRL(cfg(),'task_oracle')
