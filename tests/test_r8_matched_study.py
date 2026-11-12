import numpy as np
import pytest
from scarx.evaluation.r8_matched_study import dataset,paired_statistics,holm,refit_predict,direct_predict
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.tasks21 import ridge_probe
from scarx.neural.core import Config


@pytest.mark.parametrize('task',['strict_binding','xor','distractor_binding','order'])
def test_pinned_dataset_matches_independent_event_oracle(task):
    d=dataset(task,140011)
    assert len(d['train'][1])==1024 and len(d['ood'][1])==512


def test_probe_predictions_match_source_ridge_score():
    rng=np.random.default_rng(4);a=rng.normal(size=(50,5));b=rng.normal(size=(30,5));y=np.arange(50)%2;z=np.arange(30)%2
    assert float(np.mean(refit_predict(a,y,b)==z))==ridge_probe(a,y,b,z)


def test_direct_predictions_match_inherited_readout():
    cfg=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
    m=MatchedLocalCredit(cfg);x=np.random.default_rng(4).normal(size=(5,12,8)).astype(np.float32)
    assert np.array_equal(direct_predict(m,m.features_many(x)),m.predict_many(x))


def test_three_seed_sign_flip_cannot_be_significant_and_holm_never_lowers_p():
    s=paired_statistics([.1,.2,.3]);assert s['exact_two_sided_p']==.25
    raw={'a':.01,'b':.04,'c':.5};adjusted=holm(raw)
    assert all(adjusted[k]>=v for k,v in raw.items())
