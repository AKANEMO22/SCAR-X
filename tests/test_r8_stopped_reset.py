import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit,ARMS


def cfg():return Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
def seq():
    x=np.full((5,8),.03,np.float32);x[0,[0,2]]=1.5;x[3,4]=1.5;x[4,7]=1
    return x


def test_reset_trace_intervention_keeps_neural_forward_exact_before_updates():
    a=MatchedLocalCredit(cfg());b=StoppedResetCredit(cfg())
    for x in seq():
        a.step(x,True);b.step(x,True)
        for name in ('voltage','adaptation','spikes','features'):np.testing.assert_array_equal(getattr(a.core,name),getattr(b.core,name))
    assert not np.array_equal(a.e_v,b.e_v)


def test_stopped_voltage_derivative_matches_declared_local_recurrence():
    m=StoppedResetCredit(cfg());row=seq()[0]
    previous=m.e_v.copy();presyn=np.concatenate((np.broadcast_to(row,(48,8)),m.core.spikes[m.core.indices]),axis=1)
    m.step(row,True);np.testing.assert_array_equal(m.e_v,m.config.leak*previous+presyn)


def test_stopped_signal_controls_share_identical_trace_policy():
    models=[StoppedResetCredit(cfg(),arm) for arm in ARMS]
    for row in seq():
        for model in models:model.step(row,True)
        for model in models[1:]:
            for name in ('e_v','e_a','e_f'):np.testing.assert_array_equal(getattr(models[0],name),getattr(model,name))


@pytest.mark.parametrize('arm',ARMS)
def test_stopped_checkpoint_preserves_next_local_update(arm):
    m=StoppedResetCredit(cfg(),arm);m.calibrated=True;m.learn_episode(seq(),1)
    payload=m.dumps();loaded=StoppedResetCredit.loads(payload)
    assert loaded.dumps()==payload
    assert m.learn_episode(seq(),0)==loaded.learn_episode(seq(),0)
    assert m.dumps()==loaded.dumps()


def test_source_versions_cannot_shadow_each_other():
    from scarx.evaluation.r8_matched_credit import Base
    assert '_peer_607ffca_' in Base.__module__
