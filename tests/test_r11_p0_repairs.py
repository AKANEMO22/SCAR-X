"""Unit tests for SCAR-X Round 11 P0 Technical Repairs.

Verifies:
1. Endpoint equivalence of DampedResetEPropALIF(alpha=0.0) == eprop_matched.
2. Endpoint equivalence of DampedResetEPropALIF(alpha=1.0) == own_reset.
3. Feature decay parameter verified at 0.92 in Config.
4. Paired step-by-step trace and weight update identity across multiple episodes.
"""
import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.neural.r10_independent_replication import IndependentLearningALIF
from scarx.neural.r11_repaired_eprop import DampedResetEPropALIF
from scarx.evaluation.r8_benchmark_integrity import generate_strict_binding


def test_feature_decay_is_092():
    cfg = Config()
    assert cfg.feature_decay == 0.92, f"Expected feature_decay 0.92, got {cfg.feature_decay}"


def test_endpoint_alpha_0_equals_eprop_matched_step():
    cfg = Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=123)
    x = np.random.default_rng(123).normal(0, 1.0, (20, 8)).astype(np.float32)

    m_ref = IndependentLearningALIF(cfg, method="eprop_matched")
    m_damped = DampedResetEPropALIF(cfg, alpha_reset=0.0)

    for row in x:
        f_ref = m_ref.step(row, track=True)
        f_damped = m_damped.step(row, track=True)

        np.testing.assert_array_equal(m_ref.core.voltage, m_damped.core.voltage)
        np.testing.assert_array_equal(m_ref.core.spikes, m_damped.core.spikes)
        np.testing.assert_array_equal(m_ref.core.adaptation, m_damped.core.adaptation)
        np.testing.assert_array_equal(m_ref.core.features, m_damped.core.features)

        np.testing.assert_allclose(m_ref.e_v, m_damped.e_v, atol=1e-6)
        np.testing.assert_allclose(m_ref.e_a, m_damped.e_a, atol=1e-6)
        np.testing.assert_allclose(m_ref.e_f, m_damped.e_f, atol=1e-6)


def test_endpoint_alpha_1_equals_own_reset_step():
    cfg = Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=123)
    x = np.random.default_rng(123).normal(0, 1.0, (20, 8)).astype(np.float32)

    m_ref = IndependentLearningALIF(cfg, method="own")
    m_damped = DampedResetEPropALIF(cfg, alpha_reset=1.0)

    for row in x:
        f_ref = m_ref.step(row, track=True)
        f_damped = m_damped.step(row, track=True)

        np.testing.assert_array_equal(m_ref.core.voltage, m_damped.core.voltage)
        np.testing.assert_array_equal(m_ref.core.spikes, m_damped.core.spikes)
        np.testing.assert_array_equal(m_ref.core.adaptation, m_damped.core.adaptation)
        np.testing.assert_array_equal(m_ref.core.features, m_damped.core.features)

        np.testing.assert_allclose(m_ref.e_v, m_damped.e_v, atol=1e-6)
        np.testing.assert_allclose(m_ref.e_a, m_damped.e_a, atol=1e-6)
        np.testing.assert_allclose(m_ref.e_f, m_damped.e_f, atol=1e-6)


def test_endpoint_learn_episode_parity():
    cfg = Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=999)
    train_x, train_y = generate_strict_binding(999, 16)

    # 1. Test alpha=0.0 against eprop_matched
    m_ref0 = IndependentLearningALIF(cfg, method="eprop_matched")
    m_damped0 = DampedResetEPropALIF(cfg, alpha_reset=0.0)
    m_ref0.calibrate(train_x[:4])
    m_damped0.calibrate(train_x[:4])

    for ep in range(16):
        loss_ref, _ = m_ref0.learn_episode(train_x[ep], int(train_y[ep]))
        loss_damped, _ = m_damped0.learn_episode(train_x[ep], int(train_y[ep]))

        np.testing.assert_allclose(loss_ref, loss_damped, atol=1e-5)
        np.testing.assert_allclose(m_ref0.core.w_in, m_damped0.core.w_in, atol=1e-6)
        np.testing.assert_allclose(m_ref0.core.w_rec, m_damped0.core.w_rec, atol=1e-6)
        np.testing.assert_allclose(m_ref0.core.w_out, m_damped0.core.w_out, atol=1e-6)

    # 2. Test alpha=1.0 against own_reset
    m_ref1 = IndependentLearningALIF(cfg, method="own")
    m_damped1 = DampedResetEPropALIF(cfg, alpha_reset=1.0)
    m_ref1.calibrate(train_x[:4])
    m_damped1.calibrate(train_x[:4])

    for ep in range(16):
        loss_ref, _ = m_ref1.learn_episode(train_x[ep], int(train_y[ep]))
        loss_damped, _ = m_damped1.learn_episode(train_x[ep], int(train_y[ep]))

        np.testing.assert_allclose(loss_ref, loss_damped, atol=1e-5)
        np.testing.assert_allclose(m_ref1.core.w_in, m_damped1.core.w_in, atol=1e-6)
        np.testing.assert_allclose(m_ref1.core.w_rec, m_damped1.core.w_rec, atol=1e-6)
        np.testing.assert_allclose(m_ref1.core.w_out, m_damped1.core.w_out, atol=1e-6)
