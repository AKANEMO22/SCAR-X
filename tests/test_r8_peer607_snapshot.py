import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.evaluation.r8_matched_credit import peer_module
from scarx.evaluation.forensic import r8_snapshot as S


def models(kind):
    cfg=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
    if kind=='mtsg':
        cls=peer_module('r8_mtsg_eprop').MultiTimescaleSensoryGatedEProp
        return cls(cfg,'shuffled_signal',sensory_threshold=.123),cls(cfg,'shuffled_signal')
    cls=peer_module('r8_bf_cascade').BennaFusiMetaplasticCascade
    return cls(cfg,use_context_gating=True,kappa1=.003),cls(cfg,use_context_gating=True)


@pytest.mark.parametrize('kind',['mtsg','bf'])
def test_new_peer_version_full_state_and_next_update_exact(kind):
    model,template=models(kind);model.calibrated=True
    seq=np.full((4,8),.03,np.float32);seq[0,[0,2]]=1.5;seq[2,4]=1.5;seq[3,7]=1
    model.learn_episode(seq,1)
    if kind=='mtsg':model.signal_rng.permutation(48)
    else:
        assert model.n_context==8 and model.e_x.shape==(8,)
        model.e_x[:]=np.arange(8,dtype=np.float32)
    raw=S.dumps(model,{'phase':'P0'});loaded,state=S.loads(template,raw)
    assert S.dumps(loaded,state)==raw
    if kind=='bf':np.testing.assert_array_equal(loaded.e_x,model.e_x)
    assert loaded.learn_episode(seq,0)==model.learn_episode(seq,0)
    assert S.dumps(loaded)==S.dumps(model)


def test_old_and_new_peer_formats_cannot_be_mixed():
    new,_=models('mtsg')
    from scarx.evaluation.forensic.r8_snapshot_fixture import fixture_model
    old=fixture_model('mtsg')
    with pytest.raises(ValueError,match='mismatch'):S.loads(old,S.dumps(new))
