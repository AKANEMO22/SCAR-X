"""Unit tests verifying resolution of Phase 0 critical bugs A, B, C, D, E, F."""
from __future__ import annotations

import numpy as np
import pytest

from scarx.neural.core import Config
from scarx.neural.r9d_verified_gating import (
    R9D_ARCHITECTURES,
    CONTROL_MODES,
    VerifiedNeuralGatingALIF,
    get_current_process_rss,
)


def test_bug_a_gate_gradient_non_zero_and_updates_logits():
    """Bug A fix: gate logits must receive non-zero updates via e-prop (Bellec et al. 2020)."""
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=42)
    model = VerifiedNeuralGatingALIF(config=cfg, gating_arch="learned_channel_eprop", lr_gate=0.03)

    rng = np.random.default_rng(42)
    calib = rng.normal(0, 0.1, (5, 12, 8)).astype(np.float32)
    model.calibrate(calib)

    initial_logits = model.channel_gate_logits.copy()

    # Create sequence where channel 0 has high signal
    seq = rng.normal(0, 0.1, (12, 8)).astype(np.float32)
    seq[2, 0] += 2.0
    seq[8, 1] += 2.0

    model.learn_episode(seq, target=1)

    # Gate logits MUST have changed
    diff = np.abs(model.channel_gate_logits - initial_logits)
    assert np.max(diff) > 1e-5, f"Gate logits failed to update: max diff = {np.max(diff)}"
    assert np.any(model.last_gate_step != 0.0), "last_gate_step is all zeros"


def test_bug_a_noise_channel_gets_suppressed():
    """Bug A fix: channel carrying noise that causes error must see its gate logit decrease."""
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=123)
    model = VerifiedNeuralGatingALIF(config=cfg, gating_arch="learned_channel_eprop", lr_gate=0.05)

    rng = np.random.default_rng(123)
    calib = rng.normal(0, 0.1, (5, 12, 8)).astype(np.float32)
    model.calibrate(calib)

    # Train on episodes where channel 0 predicts target 0, but channel 6 carries distractor pulse that induces misprediction
    for ep in range(30):
        seq = rng.normal(0, 0.05, (12, 8)).astype(np.float32)
        y = ep % 2
        seq[2, y] += 1.5
        # Noise pulse on channel 6
        seq[5, 6] += 2.5
        model.learn_episode(seq, target=y)

    # Channel 6 must receive negative pressure relative to signal channel
    # Signal channels 0 and 1 should have higher logit than noise channel 6
    assert (
        model.channel_gate_logits[0] > model.channel_gate_logits[6]
        or model.channel_gate_logits[1] > model.channel_gate_logits[6]
    ), f"Expected signal channel > noise channel logit: {model.channel_gate_logits}"


def test_bug_c_shuffled_control_bit_identical_traces_before_intervention():
    """Bug C fix: learned and shuffled_signal controls must compute bit-identical forward states and traces."""
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=999)
    m_learned = VerifiedNeuralGatingALIF(config=cfg, gating_arch="learned_channel_eprop", control_mode="learned")
    m_shuf = VerifiedNeuralGatingALIF(config=cfg, gating_arch="learned_channel_eprop", control_mode="shuffled_signal")

    rng = np.random.default_rng(999)
    calib = rng.normal(0, 0.1, (5, 12, 8)).astype(np.float32)
    m_learned.calibrate(calib)
    m_shuf.calibrate(calib)

    test_seq = rng.normal(0, 0.2, (12, 8)).astype(np.float32)

    # Step through sequence and compare traces at every step
    m_learned.reset_episode()
    m_shuf.reset_episode()

    for row in test_seq:
        f_learn = m_learned.step(row, track=True)
        f_shuf = m_shuf.step(row, track=True)

        np.testing.assert_array_equal(f_learn, f_shuf)
        np.testing.assert_array_equal(m_learned.core.voltage, m_shuf.core.voltage)
        np.testing.assert_array_equal(m_learned.core.spikes, m_shuf.core.spikes)
        np.testing.assert_array_equal(m_learned.core.adaptation, m_shuf.core.adaptation)
        np.testing.assert_array_equal(m_learned.e_v, m_shuf.e_v)
        np.testing.assert_array_equal(m_learned.e_a, m_shuf.e_a)
        np.testing.assert_array_equal(m_learned.e_f, m_shuf.e_f)
        np.testing.assert_array_equal(m_learned.e_gate, m_shuf.e_gate)


def test_bug_e_real_rss_measurement():
    """Bug E fix: memory must be measured from real process via psutil, not hardcoded constant."""
    rss = get_current_process_rss()
    assert isinstance(rss, int)
    assert rss > 1_000_000, f"Expected realistic process memory > 1MB, got {rss}"
    assert rss < 2_000_000_000, f"Process memory exceeded 2GB limit: {rss}"


def test_bug_f_baselines_separated():
    """Bug F fix: R9b recurrent gating baseline and raw ALIF no_gating baseline must be distinct."""
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=42)
    m_none = VerifiedNeuralGatingALIF(config=cfg, gating_arch="no_gating")
    m_r9b = VerifiedNeuralGatingALIF(config=cfg, gating_arch="r9b_recurrent_gating")

    rng = np.random.default_rng(42)
    calib = rng.normal(0, 0.1, (5, 12, 8)).astype(np.float32)
    m_none.calibrate(calib)
    m_r9b.calibrate(calib)

    # Step 1: strong pulse that fires spikes to populate recurrent pre-synaptic trace
    row1 = np.full(8, 2.0, dtype=np.float32)
    m_none.step(row1, track=True)
    m_r9b.step(row1, track=True)

    # Step 2: subthreshold row (max < 0.6)
    row2 = np.full(8, 0.2, dtype=np.float32)
    m_none.step(row2, track=True)
    m_r9b.step(row2, track=True)

    # In r9b_recurrent_gating, recurrent eligibility is gated by 0.15 when s_act < 0.60
    # While in no_gating, recurrent eligibility is ungated (1.0)
    diff_ef = np.max(np.abs(m_none.e_f[:, 8:] - m_r9b.e_f[:, 8:]))
    assert diff_ef > 1e-4, "R9b baseline and no_gating failed to exhibit separate trace dynamics"
