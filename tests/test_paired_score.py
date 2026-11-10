import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.neural.plasticity import PlasticCore
from scarx.neural.paired_score import PairedScoreCore


def test_fixed_uniform_stochastic_forward_and_local_score_match_original_unclipped():
    c=Config(hidden=48,outputs=2,plastic_rows=12,seed=14207);a=PlasticCore(c);b=PairedScoreCore(c)
    sequence=np.random.default_rng(31).normal(0,.2,(6,8)).astype(np.float32)
    rng=np.random.default_rng();rng.bit_generator.state=a.rng.bit_generator.state
    for value in sequence:
        noise=rng.random(48);a.step(value);b.step(value,noise)
        for name in ('spikes','voltage','adaptation','features'):np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        np.testing.assert_allclose(a.elig_in,b.elig_in,atol=1e-6)


def test_moments_only_selected_rows_and_exact_partial_state_rng_continuation():
    c=Config(hidden=48,outputs=2,plastic_rows=12,seed=14221);m=PairedScoreCore(c,'antithetic',shuffle=True)
    sequence=np.random.default_rng(32).normal(0,.2,(12,8)).astype(np.float32)
    m.calibrate(m.features_many(np.stack((sequence,sequence+.01)),samples=2))
    before=m.w_in.copy();inactive=np.setdiff1d(np.arange(48),m.rows)
    m.update_hidden([np.ones_like(m.elig_in),np.ones_like(m.elig_rec)])
    np.testing.assert_array_equal(m.w_in[inactive],before[inactive]);assert not m.moment_in[inactive].any()
    for value in sequence[:4]:m.step(value)
    clone=PairedScoreCore.loads(m.dumps())
    for value in sequence[4:]:m.step(value);clone.step(value)
    assert m.dumps()==clone.dumps()
    assert m.learn_example(sequence,1)==clone.learn_example(sequence,1)
    assert m.dumps()==clone.dumps()
    state=m.dumps();m.features_many(np.stack((sequence,sequence+.01)));assert m.dumps()==state
    with pytest.raises(ValueError):m.calibrate(np.ones((128,48)))


def test_frozen_hidden_has_same_compute_and_does_not_learn_from_labels():
    m=PairedScoreCore(Config(hidden=48,outputs=2,plastic_rows=12),frozen=True)
    seq=np.zeros((12,8),np.float32);m.calibrate(m.features_many(np.stack((seq,seq+.01)),samples=2))
    before=m.w_in.copy();result=m.learn_example(seq,1)
    assert result['rollouts']==8 and result['ticks']==96 and result['hidden_updates']==0 and m.updates==1
    np.testing.assert_array_equal(before,m.w_in);assert not m.row_updates.any()


def test_actual_local_core_trace_matches_finite_path_expected_gradient():
    from scarx.evaluation.score_integrity import fixture,paths,run,exact
    f=fixture('moderate');histories=paths();oracle=run(f,spikes=histories)
    m=PairedScoreCore(Config(inputs=2,hidden=2,outputs=2,fan_in=1,plastic_rows=2),frozen=True)
    for name,source in (('w_in','wi'),('w_rec','wr'),('w_out','wo'),('bias','bias'),('indices','indices')):getattr(m,name)[:]=f[source]
    actual=np.zeros(6)
    for history,probability in zip(histories,oracle['path_probability']):
        m.reset_state()
        for value,spike in zip(f['x'],history):m.step(value,np.where(spike,0.,1.))
        reward=np.log(float(m.probabilities()[1]));score=np.concatenate((m.elig_in.ravel(),m.elig_rec.ravel()))
        actual+=probability*(reward+.7)*score
    np.testing.assert_allclose(actual,exact(f)['finite_gradient'],atol=5e-7,rtol=5e-5)
