import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit,Base,ARMS


def config():return Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
def fixture():
    x=np.full((5,8),.03,np.float32);x[0,[0,2]]=1.5;x[3,4]=1.5;x[4,7]=1
    return x


def test_multiscale_gated_candidate_keeps_pinned_forward_and_local_update_exact():
    original=Base(config(),'learned_mtsg');matched=MatchedLocalCredit(config(),'multi_gated')
    original.calibrated=matched.calibrated=True
    for i in range(3):
        assert original.learn_episode(fixture(),i%2)==matched.learn_episode(fixture(),i%2)
        for name in original.core.arrays():np.testing.assert_array_equal(original.core.arrays()[name],matched.core.arrays()[name])
        for name in original.state_array_names:np.testing.assert_array_equal(getattr(original,name),getattr(matched,name))


def test_signal_controls_share_trace_before_modulation():
    models=[MatchedLocalCredit(config(),arm) for arm in ARMS if arm.startswith('multi_gated')]
    for row in fixture():
        for model in models:model.step(row,True)
        for model in models[1:]:
            for name in ('e_v','e_a','e_f'):np.testing.assert_array_equal(getattr(models[0],name),getattr(model,name))
            np.testing.assert_array_equal(models[0].core.voltage,model.core.voltage)


def test_zero_frozen_readout_remain_identical_and_hidden_weights_fixed():
    models=[MatchedLocalCredit(config(),arm) for arm in ('multi_gated_zero','multi_gated_frozen','multi_gated_readout_only')]
    initial=models[0].core.w_in.copy()
    for model in models:model.calibrated=True
    for i in range(3):
        losses=[m.learn_episode(fixture(),i%2) for m in models]
        assert losses[0]==losses[1]==losses[2]
    for model in models:
        assert model.hidden_updates==0
        np.testing.assert_array_equal(initial,model.core.w_in)
        for name in model.core.arrays():np.testing.assert_array_equal(models[0].core.arrays()[name],model.core.arrays()[name])
        assert models[0].signal_rng.bit_generator.state==model.signal_rng.bit_generator.state


@pytest.mark.parametrize('arm',list(ARMS))
def test_matched_snapshot_restores_policy_and_next_local_update(arm):
    m=MatchedLocalCredit(config(),arm);m.calibrated=True;m.learn_episode(fixture(),1)
    blob=m.dumps();loaded=MatchedLocalCredit.loads(blob)
    assert loaded.dumps()==blob
    assert m.learn_episode(fixture(),0)==loaded.learn_episode(fixture(),0)
    assert m.dumps()==loaded.dumps()
