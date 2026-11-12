"""Unit tests for SCAR-X Round 9 Unified Neural Model & Invariants."""
from __future__ import annotations

import numpy as np
import pytest

from scarx.neural.core import Config, SpikingCore
from scarx.neural.r9_unified_mechanisms import (
    R9UnifiedNeuralModel,
    compute_surrogate_slope,
    generate_timescale_vector,
)


def test_surrogate_slopes():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    for fn in ("fast_sigmoid", "exponential", "boxcar", "gaussian"):
        slope = compute_surrogate_slope(x, fn_type=fn, gamma=0.5, param=0.5)
        assert slope.shape == x.shape
        assert (slope >= 0.0).all()
        assert np.isfinite(slope).all()


def test_timescale_vectors():
    h = 48
    for dist in ("uniform_fast", "uniform_slow", "uniform_mid", "hetero_linear", "hetero_log", "hetero_bimodal"):
        vec = generate_timescale_vector(h, dist_type=dist, gamma_min=0.25, gamma_max=0.92)
        assert vec.shape == (h, 1)
        assert (vec >= 0.25).all()
        assert (vec <= 0.92).all()


def test_unified_model_forward_and_learn():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=42)
    model = R9UnifiedNeuralModel(
        cfg,
        reset_mode="subtractive_exact",
        surrogate_fn="fast_sigmoid",
        timescale_dist="hetero_linear",
        gating_mode="recurrent_only",
        use_bf_cascade=True,
    )

    # Invariants
    assert model.core.config.hidden == 48
    assert model.core.config.fan_in == 8
    assert model.core.config.inputs == 8

    # Calibrate
    cal_seqs = np.random.randn(8, 12, 8).astype(np.float32)
    model.calibrate(cal_seqs)
    assert model.calibrated

    # Step
    feat = model.step(cal_seqs[0, 0])
    assert feat.shape == (48,)
    assert np.isin(model.core.spikes, [0.0, 1.0]).all()

    # Learn
    loss = model.learn_episode(cal_seqs[0], target=1)
    assert np.isfinite(loss)
    assert model.episodes_learned == 1
    assert model.hidden_updates == 1

    # Predict
    pred = model.predict(cal_seqs[0])
    assert pred in (0, 1)
    preds = model.predict_many(cal_seqs)
    assert len(preds) == 8


def test_unified_model_serialization_and_revert():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=42)
    model = R9UnifiedNeuralModel(cfg, use_bf_cascade=True)
    cal_seqs = np.random.randn(8, 12, 8).astype(np.float32)
    model.calibrate(cal_seqs)
    for seq in cal_seqs:
        model.learn_episode(seq, target=0)

    # Serialize
    blob = model.dumps()
    assert isinstance(blob, bytes) and len(blob) > 0

    # Deserialize
    loaded = R9UnifiedNeuralModel.loads(blob)
    assert loaded.calibrated == model.calibrated
    assert loaded.episodes_learned == model.episodes_learned
    np.testing.assert_allclose(loaded.core.w_in, model.core.w_in)
    np.testing.assert_allclose(loaded.core.w_rec, model.core.w_rec)
    np.testing.assert_allclose(loaded.core.w_out, model.core.w_out)

    # Causal Revert
    model.revert_hidden_weights(revert_in=True, revert_rec=True)
    np.testing.assert_allclose(model.core.w_in, model.w_in_0)
    np.testing.assert_allclose(model.core.w_rec, model.w_rec_0)


def test_unified_model_ram_limit():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=42)
    model = R9UnifiedNeuralModel(cfg, use_bf_cascade=True, readout_mode="context_gated")
    mem = model.memory_bytes()
    assert mem < 10_000_000  # Well below 2 GB process limit
