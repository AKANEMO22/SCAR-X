import numpy as np
import pytest
from scarx.neural.paired_score import PairedScoreCore
from scarx.neural.core import Config,SpikingCore
from scarx.evaluation.inference_dynamics import inference_features
from scarx.evaluation.inference_dynamics import timescale_view


def test_original_stochastic_view_and_existing_deterministic_alif_match_without_mutation():
    c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=15217)
    m=PairedScoreCore(c);x=np.random.default_rng(771).normal(0,.3,(4,12,8)).astype(np.float32);x[:,0,0]+=2
    before=m.dumps()
    np.testing.assert_array_equal(inference_features(m,x,331,.4,8),m.features_many(x,331,samples=8))
    cold=inference_features(m,x,331,0.,1);base=SpikingCore(c)
    expected=[]
    for sequence in x:base.forward(sequence);expected.append(base.features.copy())
    np.testing.assert_array_equal(cold,np.stack(expected))
    np.testing.assert_array_equal(cold,inference_features(m,x,331,0.,8));assert np.count_nonzero(cold)>0 and m.dumps()==before
    with pytest.raises(ValueError):inference_features(m,x,331,.2,8)


def test_timescale_vectors_keep_baseline_and_match_mean_tau_without_mutation():
    c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=15217);m=PairedScoreCore(c)
    x=np.random.default_rng(772).normal(0,.3,(5,12,8)).astype(np.float32);state=m.dumps()
    baseline=timescale_view(c,'baseline',15217)
    np.testing.assert_array_equal(inference_features(m,x,441,.4,8,baseline),m.features_many(x,441,samples=8))
    spread=timescale_view(c,'all_spread',15217);mean=timescale_view(c,'all_mean',15217)
    for key in spread:
        assert len(np.unique(spread[key]))==48
        assert np.mean(-1/np.log(spread[key]))==pytest.approx(float((-1/np.log(mean[key]))[0]),rel=2e-5)
    inference_features(m,x,441,.4,8,spread);assert m.dumps()==state
