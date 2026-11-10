import numpy as np
from scarx.evaluation.score_integrity import fixture,exact,run,paths


def test_forward_local_score_matches_expected_reward_derivative_with_recurrence():
    for name in ('moderate','strong','delay'):
        value=exact(fixture(name));assert value['oracle_pass'],value
        assert value['path_count']==4096 and abs(value['probability_sum']-1)<1e-10
    assert exact(fixture('strong'))['variants']['trace_clip10']['bias_l2']>1e-3


def test_score_is_independent_of_current_reward_baseline_and_teacher_head():
    f=fixture('moderate');history=paths()
    a=run(f,spikes=history);changed={**f,'wo':-f['wo'],'target':0};b=run(changed,spikes=history,baseline=3)
    np.testing.assert_array_equal(a['score'],b['score'])
    assert np.linalg.norm(a['estimator']-b['estimator'])>0
