import numpy as np
from scarx.neural.core import Config
from scarx.neural.paired_score import PairedScoreCore
from scarx.evaluation.neuron_state_views import state_features,VIEWS,local_head


def test_state_channels_preserve_original_forward_and_source_rng_without_input_bypass():
    model=PairedScoreCore(Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12))
    x=np.random.default_rng(921).normal(0,.4,(7,12,8)).astype(np.float32);state=model.dumps()
    actual=state_features(model,x,971,8)
    np.testing.assert_array_equal(actual['filtered_spikes'],model.features_many(x,971,samples=8))
    assert set(actual)==set(VIEWS) and all(v.shape==(7,48) and np.isfinite(v).all() for v in actual.values())
    assert model.dumps()==state and np.linalg.norm(actual['membrane_after_reset'])>0
    f=np.tile(np.eye(4,dtype=np.float32),(8,1));labels=np.tile(np.arange(4),8)
    assert local_head(f,labels,f,labels)['accuracy']==1
