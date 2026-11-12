"""Numerical reference tests only; no dataset/ALIF training or P2 execution."""
import copy
import numpy as np
import pytest
from scarx.evaluation.literature_si import SynapticIntelligenceState as SI


def test_work_from_physical_optimizer_delta_matches_quadratic_taylor():
    before=np.array([1.,-2.]);gradient=before.copy();after=before-.1*gradient
    s=SI(before,xi=.1);s.observe(before,after,gradient)
    np.testing.assert_allclose(s.work,[.1,.4],atol=1e-15)
    # First-order loss decrease minus positive second-order Taylor remainder.
    observed=.5*(before@before-after@after)
    remainder=.5*np.sum((after-before)**2)
    assert np.isclose(s.work.sum(),observed+remainder)
    s.consolidate()
    np.testing.assert_allclose(s.importance,[.1/.11,.4/.14],atol=1e-14)
    assert np.array_equal(s.anchor,after) and not s.work.any()


def test_clipped_delta_is_not_replaced_by_raw_step_or_gradient():
    s=SI([1.]);s.observe([1.],[.9],[2.])
    assert np.isclose(s.work[0],.2)  # proposed SGD step could have been-2
    s.consolidate();assert np.isclose(s.importance[0],.2/.11)


def test_rectification_after_sum_or_ema_matches_pinned_source():
    s=SI([0.],xi=1);s.importance[:]=2
    s.observe([0.],[1.],[2.])  # negative contribution -2/(1+1)=-1
    summed=SI.from_state(s.state_dict());averaged=SI.from_state(s.state_dict())
    summed.consolidate('sum');averaged.consolidate(.5)
    assert summed.importance[0]==1 and averaged.importance[0]==.5
    # max(contribution,0) before addition would incorrectly retain importance2.


def test_regularizer_gradient_independent_finite_difference():
    s=SI([.1,-.2]);s.importance[:]=[2,3];w=np.array([.4,.8])
    value,g=s.regularizer(w,strength=.7)
    assert np.isclose(value,.7*(2*.3**2+3*1.**2))
    for i in range(2):
        d=np.zeros(2);d[i]=1e-6
        estimate=(s.regularizer(w+d,.7)[0]-s.regularizer(w-d,.7)[0])/(2e-6)
        assert np.isclose(estimate,g[i],rtol=1e-9,atol=1e-9)


def test_resume_has_full_state_and_no_aliases():
    s=SI(np.arange(866,dtype=float)/100,xi=.01)
    before=s.previous.copy();after=before-.001
    s.observe(before,after,np.ones(866));saved=s.state_dict();r=SI.from_state(saved)
    saved['arrays']['work'][:]=999
    assert not np.any(r.work==999)
    next_weights=after-.002;gradient=np.linspace(-1,1,866)
    s.observe(after,next_weights,gradient);r.observe(after,next_weights,gradient)
    s.consolidate();r.consolidate()
    for k in ('anchor','previous','work','importance'):np.testing.assert_array_equal(getattr(s,k),getattr(r,k))
    assert s.memory_bytes()==27712


@pytest.mark.parametrize('bad',[np.array([np.nan]),np.array([np.inf]),np.ones((2,2)),np.array([1+2j])])
def test_invalid_observation_leaves_state_unchanged(bad):
    s=SI([1.]);old=s.state_dict()
    with pytest.raises(ValueError):s.observe([1.],bad,[1.])
    for k,v in old['arrays'].items():np.testing.assert_array_equal(v,getattr(s,k))
    assert s.steps==0


def test_missing_or_corrupt_state_fails_closed():
    s=SI([1.]);saved=s.state_dict();saved['arrays'].pop('work')
    with pytest.raises(ValueError):SI.from_state(saved)
    saved=s.state_dict();saved['arrays']['importance'][0]=-1
    with pytest.raises(ValueError):SI.from_state(saved)
    with pytest.raises(ValueError):s.observe([.9],[.8],[1.])
    with pytest.raises(ValueError):s.consolidate(1.1)
