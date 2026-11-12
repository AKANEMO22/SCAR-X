"""Independent technical fixtures only; no task data, tuning or DEV training."""
import copy
import base64,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from scarx.neural.core import Config,SpikingCore
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.acr import AdaptiveCreditRouting
from scarx.evaluation.capacity import CapacityOwn,CapacityACR


def cfg(h=48):
    return Config(inputs=8,hidden=h,outputs=2,fan_in=8,plastic_rows=12,seed=56041)


def fixtures():
    return np.random.default_rng(56042).uniform(-.2,1.3,(5,17,8)).astype(np.float32)


def exact_common(a,b):
    assert a.config==b.config
    for name,array in a.core.arrays().items():np.testing.assert_array_equal(array,b.core.arrays()[name])
    names={name for name,value in vars(a).items() if isinstance(value,np.ndarray)}
    assert names=={name for name,value in vars(b).items() if isinstance(value,np.ndarray)}
    for name in names:np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    assert a.signal_rng.bit_generator.state==b.signal_rng.bit_generator.state
    if hasattr(a,'router_rng'):assert a.router_rng.bit_generator.state==b.router_rng.bit_generator.state
    for name in ('episodes_learned','hidden_updates','total_ticks','condition','trace_kind','gate_mode','calibrated'):
        assert getattr(a,name)==getattr(b,name)
    assert a.core.updates==b.core.updates


def test_port_changes_no_forward_trace_or_optimizer_implementations():
    assert CapacityOwn.step is MatchedLocalCredit.step
    assert CapacityOwn.learn_episode is MatchedLocalCredit.learn_episode
    assert CapacityACR.step is AdaptiveCreditRouting.step
    assert CapacityACR.learn_episode is AdaptiveCreditRouting.learn_episode
    assert CapacityACR.fixed_alpha is AdaptiveCreditRouting.fixed_alpha
    assert CapacityACR.reset_episode is AdaptiveCreditRouting.reset_episode


@pytest.mark.parametrize('h',[48,64,128])
def test_exact_within_height_initialization_and_hard_forward(h):
    models=[CapacityOwn(cfg(h)),CapacityOwn(cfg(h),'readout_only'),CapacityACR(cfg(h)),CapacityACR(cfg(h),'shuffled_signal')]
    for m in models[1:]:
        for name,array in models[0].core.arrays().items():np.testing.assert_array_equal(array,m.core.arrays()[name])
        for name in ('f_mean','f_var','syn_trace','syn_energy','w_in_0','w_rec_0','gamma_e_vec'):
            np.testing.assert_array_equal(getattr(models[0],name),getattr(m,name))
        assert m.signal_rng.bit_generator.state==models[0].signal_rng.bit_generator.state
    # Trace processing may differ, but one identical actual neural forward must
    # precede any label/update and the output spike is still hard binary.
    plain=SpikingCore(cfg(h))
    for row in fixtures()[0]:
        plain.step(row,track=False)
        for m in models:
            m.step(row)
            for name in ('voltage','adaptation','spikes','features'):
                np.testing.assert_array_equal(getattr(m.core,name),getattr(plain,name))
            assert np.isin(m.core.spikes,[0,1]).all()
    assert all(m.episodes_learned==0 and m.hidden_updates==0 for m in models)


@pytest.mark.parametrize('h',[48,64,128])
def test_analytical_memory_and_fixed_profile_are_actual_arrays(h):
    own=CapacityOwn(cfg(h));acr=CapacityACR(cfg(h))
    assert own.memory_bytes()==596*h+864
    assert acr.memory_bytes()==944*h+864
    assert sum(getattr(own.core,n).size for n in ('w_in','w_rec','w_out','bias'))==18*h+2
    np.testing.assert_array_equal(acr.alpha[:h//2],np.full(h//2,.25,np.float32))
    np.testing.assert_array_equal(acr.alpha[h//2:],np.full(h-h//2,.75,np.float32))
    assert own.core.indices.shape==(h,8)
    assert np.all(own.core.indices>=0) and np.all(own.core.indices<h)


@pytest.mark.parametrize('condition,arm',[('learned_mtsg','multi_gated'),('readout_only','multi_gated_readout_only'),('shuffled_signal','multi_gated_shuffle')])
def test_h48_reproduces_all_owned_state_at_every_fixture_update(condition,arm):
    xs=fixtures()
    own=CapacityOwn(cfg(),condition);original=MatchedLocalCredit(cfg(),arm)
    acr=CapacityACR(cfg(),condition);original_acr=AdaptiveCreditRouting(cfg(),'groups_up',condition)
    for a,b in ((own,original),(acr,original_acr)):
        a.calibrate(xs[:2]);b.calibrate(xs[:2]);exact_common(a,b)
        for i,x in enumerate(xs):
            assert a.learn_episode(x,i%2)==b.learn_episode(x,i%2)
            exact_common(a,b)


@pytest.mark.parametrize('cls',[CapacityOwn,CapacityACR])
@pytest.mark.parametrize('h',[64,128])
def test_learning_is_not_limited_to_legacy_twelve_plastic_rows(cls,h):
    m=cls(cfg(h));m.calibrated=True
    before=m.core.w_in.copy();old_rows=m.core.rows.copy()
    m.learn_episode(fixtures()[0],1)
    changed=np.any(m.core.w_in!=before,axis=1)
    assert changed.sum()>12
    assert np.any(changed[np.setdiff1d(np.arange(h),old_rows)])
    assert m.hidden_updates==1 and m.episodes_learned==1


@pytest.mark.parametrize('cls',[CapacityOwn,CapacityACR])
def test_frozen_hidden_preserves_weights_and_signal_rng_schedule(cls):
    a=cls(cfg(128));b=cls(cfg(128),'readout_only')
    a.calibrated=b.calibrated=True
    initial=b.core.w_in.copy()
    for i,x in enumerate(fixtures()):
        a.learn_episode(x,i%2);b.learn_episode(x,i%2)
    np.testing.assert_array_equal(b.core.w_in,initial)
    assert b.hidden_updates==0 and b.episodes_learned==5
    assert a.signal_rng.bit_generator.state==b.signal_rng.bit_generator.state


@pytest.mark.parametrize('cls',[CapacityOwn,CapacityACR])
@pytest.mark.parametrize('h',[48,64,128])
def test_mid_episode_roundtrip_and_next_update(cls,h):
    m=cls(cfg(h),'shuffled_signal');m.calibrated=True
    xs=fixtures();m.learn_episode(xs[0],0);m.reset_episode()
    for row in xs[1,:7]:m.step(row)
    blob=m.dumps();restored=cls.loads(blob)
    assert restored.dumps()==blob
    for row in xs[1,7:]:
        np.testing.assert_array_equal(m.step(row),restored.step(row))
    assert m.learn_episode(xs[2],1)==restored.learn_episode(xs[2],1)
    assert m.dumps()==restored.dumps()


def test_unsupported_heights_and_neuron_types_fail_before_constructing_large_core():
    for h in (49,96,256,10**8):
        with pytest.raises(ValueError):CapacityOwn(cfg(h))
        with pytest.raises(ValueError):CapacityACR(cfg(h))
    with pytest.raises(ValueError):CapacityOwn(Config(inputs=8,hidden=64,outputs=2,fan_in=8,plastic_rows=12,neuron='rate'))


@pytest.mark.parametrize('cls',[CapacityOwn,CapacityACR])
@pytest.mark.parametrize('h',[64,128])
def test_cold_process_replays_unfinished_tail_before_next_update(cls,h):
    m=cls(cfg(h),'shuffled_signal');m.calibrated=True;xs=fixtures()
    m.learn_episode(xs[0],0);m.reset_episode()
    for row in xs[1,:7]:m.step(row)
    payload=m.dumps()
    for row in xs[1,7:]:m.step(row)
    expected_tail=m.dumps();m.learn_episode(xs[2],1);expected_update=m.dumps()
    script="""import sys,base64,json,numpy as np
from scarx.evaluation.capacity import CapacityOwn,CapacityACR
cls=CapacityOwn if sys.argv[1]=='CapacityOwn' else CapacityACR
m=cls.loads(base64.b64decode(sys.stdin.buffer.readline()))
xs=np.random.default_rng(56042).uniform(-.2,1.3,(5,17,8)).astype(np.float32)
for row in xs[1,7:]:m.step(row)
tail=base64.b64encode(m.dumps()).decode('ascii')
m.learn_episode(xs[2],1)
print(json.dumps({'tail':tail,'updated':base64.b64encode(m.dumps()).decode('ascii')}))
"""
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    result=subprocess.run([sys.executable,'-c',script,cls.__name__],input=base64.b64encode(payload)+b'\n',capture_output=True,check=True,env=env)
    returned=json.loads(result.stdout)
    assert base64.b64decode(returned['tail'])==expected_tail
    assert base64.b64decode(returned['updated'])==expected_update
