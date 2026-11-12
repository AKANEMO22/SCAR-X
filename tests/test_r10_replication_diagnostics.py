"""Unit tests for Round 10 independent replication and bottleneck analysis."""
import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.neural.r10_independent_replication import IndependentLearningALIF
from scarx.evaluation.r10_bottleneck_analysis import (
    full_surrogate_recurrent_sensitivity,
    compute_gradient_cosine,
    refit_probe,
    evaluate_model_standard,
)
from scarx.evaluation.r8_benchmark_integrity import generate_strict_binding


def test_independent_methods_initialization_and_step():
    cfg = Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=42)
    methods = ["own", "stopped", "eprop_matched", "eprop_aligned", "braintrace"]
    x = np.random.default_rng(42).normal(0, 1.0, (12, 8)).astype(np.float32)

    for method in methods:
        model = IndependentLearningALIF(cfg, method=method)
        model.calibrate(x[None, :, :])
        feat = model.step(x[0])
        assert feat.shape == (48,)
        assert np.isfinite(feat).all()
        assert model.e_f.shape == (48, 16)
        assert np.isfinite(model.e_f).all()


def test_full_surrogate_oracle_and_cosine():
    cfg = Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=123)
    model = IndependentLearningALIF(cfg, method="braintrace")
    x = np.random.default_rng(123).normal(0, 1.0, (10, 8)).astype(np.float32)
    signal = np.random.default_rng(124).normal(0, 1.0, 48)

    # Compute full sensitivity
    exact = full_surrogate_recurrent_sensitivity(model.core, x)
    assert exact.shape == (48, 48 * 16)
    assert np.isfinite(exact).all()

    # Step local model
    for row in x:
        model.step(row, track=True)

    local_dir = (signal[:, None] * model.e_f).ravel()
    full_ref = signal @ exact

    metrics = compute_gradient_cosine(local_dir, full_ref)
    assert -1.0 <= metrics["cosine"] <= 1.0
    assert 0.0 <= metrics["sign_agreement"] <= 1.0
    # BrainTrace on initial frozen fixture should have high cosine (>0.85)
    assert metrics["cosine"] > 0.85


def test_acr_endpoints_match_own_and_stopped():
    cfg = Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=77)
    x = np.random.default_rng(77).normal(0, 1.0, (12, 8)).astype(np.float32)

    # Check endpoint 0 == own
    m_own = IndependentLearningALIF(cfg, method="own")
    m_acr0 = IndependentLearningALIF(cfg, method="acr_static_0")
    for row in x:
        m_own.step(row, track=True)
        m_acr0.step(row, track=True)
    np.testing.assert_allclose(m_own.e_f, m_acr0.e_f, rtol=1e-5, atol=1e-5)

    # Check endpoint 100 == stopped
    m_stop = IndependentLearningALIF(cfg, method="stopped")
    m_acr100 = IndependentLearningALIF(cfg, method="acr_static_100")
    for row in x:
        m_stop.step(row, track=True)
        m_acr100.step(row, track=True)
    np.testing.assert_allclose(m_stop.e_f, m_acr100.e_f, rtol=1e-5, atol=1e-5)


def test_learn_episode_and_evaluation():
    cfg = Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=999)
    model = IndependentLearningALIF(cfg, method="own")
    train_x, train_y = generate_strict_binding(100, 32)
    model.calibrate(train_x[:8])

    loss, diag = model.learn_episode(train_x[0], int(train_y[0]), record_diagnostics=True)
    assert np.isfinite(loss)
    assert diag is not None
    assert "raw_direction" in diag
    assert "norm_step" in diag
    assert "step" in diag

    # Test evaluation
    data = {
        "refit": (train_x[:16], train_y[:16]),
        "id": (train_x[16:24], train_y[16:24]),
        "ood": (train_x[24:32], train_y[24:32]),
    }
    metrics = evaluate_model_standard(model, data)
    assert "id_direct" in metrics
    assert "id_probe" in metrics
    assert "direct_revert_drop" in metrics
