"""Unit tests for FactorialMTSGEProp in SCAR-X Round 9b.

Verifies:
1. Instantiation of all 24 configurations (4 decays x 3 gatings x 2 resets).
2. Strict matched control invariants (learned vs shuffled vs zero vs frozen).
3. Bit-for-bit trace equality before modulation.
4. Causal revert functionality.
5. Soft gating continuity.
6. Stopped reset vs own reset mechanics.
"""
import pytest
import numpy as np

from scarx.neural.core import Config
from scarx.neural.r9b_factorial_eprop import (
    FactorialMTSGEProp,
    DECAY_MODES,
    GATING_MODES,
    RESET_MODES,
    CONTROL_MODES,
)


def make_config(seed=42):
    return Config(
        inputs=8,
        outputs=2,
        hidden=48,
        fan_in=8,
        plastic_rows=12,
        neuron="alif",
        seed=seed,
    )


def test_all_24_configurations_instantiate():
    cfg = make_config()
    count = 0
    for decay in DECAY_MODES:
        for gating in GATING_MODES:
            for reset in RESET_MODES:
                model = FactorialMTSGEProp(
                    config=cfg,
                    decay_mode=decay,
                    gating_mode=gating,
                    reset_mode=reset,
                    control_mode="learned",
                )
                assert model.gamma_e_vec.shape == (48, 1)
                assert model.core.w_in.shape == (48, 8)
                assert model.core.w_rec.shape == (48, 8)
                count += 1
    assert count == 24


def test_matched_shuffled_control_bit_identical_traces():
    """CRITICAL INVARIANT: Matched shuffled control must have 100% IDENTICAL traces."""
    cfg = make_config(seed=123)
    seq = np.random.default_rng(99).normal(0, 0.5, (12, 8)).astype(np.float32)

    calib_seqs = np.random.default_rng(77).normal(0, 0.5, (4, 12, 8)).astype(np.float32)

    learned = FactorialMTSGEProp(
        config=cfg,
        decay_mode="heterogeneous",
        gating_mode="recurrent_only",
        reset_mode="own_reset",
        control_mode="learned",
    )
    shuffled = FactorialMTSGEProp(
        config=cfg,
        decay_mode="heterogeneous",
        gating_mode="recurrent_only",
        reset_mode="own_reset",
        control_mode="shuffled_signal",
    )

    learned.calibrate(calib_seqs)
    shuffled.calibrate(calib_seqs)

    # Initial weights must be identical
    np.testing.assert_array_equal(learned.core.w_in, shuffled.core.w_in)
    np.testing.assert_array_equal(learned.core.w_rec, shuffled.core.w_rec)
    np.testing.assert_array_equal(learned.gamma_e_vec, shuffled.gamma_e_vec)

    # Step through episode
    learned.reset_episode()
    shuffled.reset_episode()
    for row in seq:
        learned.step(row, track=True)
        shuffled.step(row, track=True)

    # Traces MUST be bit-for-bit identical before credit update!
    np.testing.assert_array_equal(learned.e_v, shuffled.e_v)
    np.testing.assert_array_equal(learned.e_a, shuffled.e_a)
    np.testing.assert_array_equal(learned.e_f, shuffled.e_f)

    # Now learn an episode
    loss_learned = learned.learn_episode(seq, target=1)
    loss_shuffled = shuffled.learn_episode(seq, target=1)

    assert np.isclose(loss_learned, loss_shuffled)
    # Signal norms and eligibility norms must match exactly!
    assert np.isclose(learned.last_step_eligibility_norm, shuffled.last_step_eligibility_norm)
    assert np.isclose(learned.last_step_signal_norm, shuffled.last_step_signal_norm)

    # Hidden weights must DIVERGE because credit assignment was permuted
    assert not np.array_equal(learned.core.w_in, shuffled.core.w_in)


def test_zero_signal_control_does_not_change_hidden_weights():
    cfg = make_config(seed=456)
    seq = np.random.default_rng(99).normal(0, 0.5, (12, 8)).astype(np.float32)
    calib = np.random.default_rng(77).normal(0, 0.5, (4, 12, 8)).astype(np.float32)

    zero_m = FactorialMTSGEProp(
        config=cfg,
        decay_mode="uniform_fast",
        gating_mode="none",
        reset_mode="stopped_reset",
        control_mode="zero_signal",
    )
    zero_m.calibrate(calib)

    w_in_orig = zero_m.core.w_in.copy()
    w_rec_orig = zero_m.core.w_rec.copy()
    w_out_orig = zero_m.core.w_out.copy()

    zero_m.learn_episode(seq, target=0)

    # Hidden weights must remain unchanged!
    np.testing.assert_array_equal(zero_m.core.w_in, w_in_orig)
    np.testing.assert_array_equal(zero_m.core.w_rec, w_rec_orig)
    # Readout must adapt
    assert not np.array_equal(zero_m.core.w_out, w_out_orig)


def test_causal_revert_restores_initial_hidden_weights():
    cfg = make_config(seed=789)
    seq = np.random.default_rng(99).normal(0, 0.5, (12, 8)).astype(np.float32)
    calib = np.random.default_rng(77).normal(0, 0.5, (4, 12, 8)).astype(np.float32)

    m = FactorialMTSGEProp(config=cfg, control_mode="learned")
    m.calibrate(calib)

    w_in_0 = m.w_in_0.copy()
    w_rec_0 = m.w_rec_0.copy()

    m.learn_episode(seq, target=1)

    assert not np.array_equal(m.core.w_in, w_in_0)
    w_out_trained = m.core.w_out.copy()

    m.revert_hidden()

    # Hidden weights restored
    np.testing.assert_array_equal(m.core.w_in, w_in_0)
    np.testing.assert_array_equal(m.core.w_rec, w_rec_0)
    # Readout preserved
    np.testing.assert_array_equal(m.core.w_out, w_out_trained)


def test_stopped_reset_vs_own_reset_difference():
    cfg = make_config(seed=999)
    seq = np.ones((12, 8), dtype=np.float32) * 1.5  # Strong input inducing spikes
    calib = np.ones((2, 12, 8), dtype=np.float32)

    own = FactorialMTSGEProp(config=cfg, reset_mode="own_reset", control_mode="learned")
    stopped = FactorialMTSGEProp(config=cfg, reset_mode="stopped_reset", control_mode="learned")

    own.calibrate(calib)
    stopped.calibrate(calib)

    own.learn_episode(seq, target=1)
    stopped.learn_episode(seq, target=1)

    # Because spikes occurred, e_v in own_reset subtracted theta*z_e, whereas stopped_reset kept u_e
    assert not np.array_equal(own.e_v, stopped.e_v)
    assert not np.array_equal(own.core.w_in, stopped.core.w_in)


def test_soft_sensory_gating_behavior():
    cfg = make_config()
    m = FactorialMTSGEProp(config=cfg, gating_mode="soft_sensory")
    g_low = m.compute_sensory_gate(np.zeros(8, dtype=np.float32))
    g_high = m.compute_sensory_gate(np.ones(8, dtype=np.float32) * 1.5)
    g_mid = m.compute_sensory_gate(np.ones(8, dtype=np.float32) * 0.6)

    assert 0.14 <= g_low <= 0.16
    assert 0.98 <= g_high <= 1.00
    assert g_low < g_mid < g_high


def test_frozen_hidden_vs_readout_only_control_difference():
    """Confirms frozen_hidden updates neither readout nor hidden, while readout_only updates readout."""
    cfg = make_config(seed=555)
    seq = np.random.default_rng(1).normal(0, 0.5, (12, 8)).astype(np.float32)
    calib = np.random.default_rng(2).normal(0, 0.5, (4, 12, 8)).astype(np.float32)

    frozen = FactorialMTSGEProp(config=cfg, control_mode="frozen_hidden")
    readout = FactorialMTSGEProp(config=cfg, control_mode="readout_only")

    frozen.calibrate(calib)
    readout.calibrate(calib)

    w_out_orig = frozen.core.w_out.copy()
    w_in_orig = frozen.core.w_in.copy()

    frozen.learn_episode(seq, target=1)
    readout.learn_episode(seq, target=1)

    # In frozen_hidden: NOTHING changes
    np.testing.assert_array_equal(frozen.core.w_in, w_in_orig)
    np.testing.assert_array_equal(frozen.core.w_out, w_out_orig)

    # In readout_only: readout changes, hidden remains frozen
    np.testing.assert_array_equal(readout.core.w_in, w_in_orig)
    assert not np.array_equal(readout.core.w_out, w_out_orig)


def test_input_validation_and_finiteness_guards():
    cfg = make_config(seed=666)
    m = FactorialMTSGEProp(config=cfg)

    # Calibrate without finite inputs should raise ValueError
    with pytest.raises(ValueError):
        m.calibrate(np.full((2, 12, 8), np.nan, dtype=np.float32))

    calib = np.random.default_rng(3).normal(0, 0.5, (4, 12, 8)).astype(np.float32)
    m.calibrate(calib)

    # Step with wrong shape
    with pytest.raises(ValueError):
        m.step(np.zeros(7, dtype=np.float32))

    # Step with NaN
    with pytest.raises(ValueError):
        m.step(np.full(8, np.nan, dtype=np.float32))

    # Learn episode with invalid target
    with pytest.raises(ValueError):
        m.learn_episode(np.zeros((12, 8), dtype=np.float32), target=5)

    # Learn episode with non-finite seq
    with pytest.raises(ValueError):
        m.learn_episode(np.full((12, 8), np.inf, dtype=np.float32), target=0)

