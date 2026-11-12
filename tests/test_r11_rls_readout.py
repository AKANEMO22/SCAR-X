"""Unit tests for Online Regularized RLS Readout (Round 11 P3)."""
import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.neural.r11_rls_readout import RecursiveLeastSquaresReadoutALIF
from scarx.evaluation.r8_benchmark_integrity import generate_strict_binding


def test_rls_initialization_and_memory():
    cfg = Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=42)
    model = RecursiveLeastSquaresReadoutALIF(cfg)

    assert model.dim == 49
    assert model.P.shape == (49, 49)
    assert model.W_ext.shape == (2, 49)
    # Memory footprint of P and W_ext: (49*49 + 2*49)*4 bytes = 9996 bytes < 10 KB
    mem_bytes = model.P.nbytes + model.W_ext.nbytes
    assert mem_bytes < 15000, f"RLS memory exceeds 15 KB: {mem_bytes} bytes"


def test_rls_learn_episode_numerical_stability():
    cfg = Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=123)
    model = RecursiveLeastSquaresReadoutALIF(cfg)

    train_x, train_y = generate_strict_binding(123, 32)
    model.calibrate(train_x[:8])

    for ep in range(32):
        loss = model.learn_episode_rls(train_x[ep], int(train_y[ep]))
        assert np.isfinite(loss)
        # Check P is symmetric
        np.testing.assert_allclose(model.P, model.P.T, atol=1e-5)
        # Check diagonal elements of P are positive
        assert np.all(np.diag(model.P) > 0)
        # Check weights are finite
        assert np.all(np.isfinite(model.W_ext))

    preds = model.predict_many(train_x[:10])
    assert preds.shape == (10,)
    assert set(preds).issubset({0, 1})
