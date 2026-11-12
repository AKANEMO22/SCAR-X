"""Unit tests verifying MultiTimescaleSensoryGatedEProp (Candidate 1)."""
import pytest
import numpy as np

from scarx.neural.core import Config
from scarx.neural.r8_mtsg_eprop import MultiTimescaleSensoryGatedEProp
from scarx.evaluation.r8_benchmark_integrity import generate_strict_binding


def test_mtsg_initialization_and_conditions():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, seed=42)
    for cond in MultiTimescaleSensoryGatedEProp.conditions:
        model = MultiTimescaleSensoryGatedEProp(cfg, condition=cond)
        assert model.condition == cond
        assert model.e_f.shape == (48, 16)
        assert model.gamma_e_vec.shape == (48, 1)

    with pytest.raises(ValueError, match="Unsupported condition"):
        MultiTimescaleSensoryGatedEProp(cfg, condition="invalid_cond")


def test_mtsg_calibration_and_learning_step():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, seed=42)
    model = MultiTimescaleSensoryGatedEProp(cfg, condition="learned_mtsg")

    x_cal, _ = generate_strict_binding(100, 16)
    model.calibrate(x_cal)
    assert model.calibrated is True

    x_tr, y_tr = generate_strict_binding(200, 8)
    initial_w_in = model.core.w_in.copy()
    initial_w_rec = model.core.w_rec.copy()

    for seq, y in zip(x_tr, y_tr):
        loss = model.learn_episode(seq, int(y))
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    # Verify hidden weights updated
    assert not np.allclose(model.core.w_in, initial_w_in)
    assert not np.allclose(model.core.w_rec, initial_w_rec)


def test_mtsg_frozen_and_zero_controls():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, seed=42)

    for cond in ("frozen_hidden", "zero_signal", "readout_only"):
        model = MultiTimescaleSensoryGatedEProp(cfg, condition=cond)
        x_cal, _ = generate_strict_binding(100, 16)
        model.calibrate(x_cal)

        w_in_start = model.core.w_in.copy()
        w_rec_start = model.core.w_rec.copy()

        x_tr, y_tr = generate_strict_binding(200, 8)
        for seq, y in zip(x_tr, y_tr):
            model.learn_episode(seq, int(y))

        # Hidden weights must remain strictly identical
        assert np.array_equal(model.core.w_in, w_in_start)
        assert np.array_equal(model.core.w_rec, w_rec_start)


def test_mtsg_serialization_roundtrip():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, seed=42)
    model = MultiTimescaleSensoryGatedEProp(cfg, condition="learned_mtsg")
    x_cal, _ = generate_strict_binding(100, 16)
    model.calibrate(x_cal)

    x_tr, y_tr = generate_strict_binding(200, 4)
    for seq, y in zip(x_tr, y_tr):
        model.learn_episode(seq, int(y))

    data_bytes = model.dumps()
    loaded = MultiTimescaleSensoryGatedEProp.loads(data_bytes)

    assert loaded.condition == model.condition
    assert loaded.episodes_learned == model.episodes_learned
    assert np.allclose(loaded.core.w_in, model.core.w_in)
    assert np.allclose(loaded.core.w_rec, model.core.w_rec)
    assert np.allclose(loaded.e_f, model.e_f)


def test_mtsg_causal_revert():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, seed=42)
    model = MultiTimescaleSensoryGatedEProp(cfg, condition="learned_mtsg")
    x_cal, _ = generate_strict_binding(100, 16)
    model.calibrate(x_cal)

    w_in_init = model.core.w_in.copy()
    w_rec_init = model.core.w_rec.copy()

    x_tr, y_tr = generate_strict_binding(200, 4)
    for seq, y in zip(x_tr, y_tr):
        model.learn_episode(seq, int(y))

    assert not np.allclose(model.core.w_in, w_in_init)
    model.revert_hidden_weights(revert_in=True, revert_rec=True)
    assert np.allclose(model.core.w_in, w_in_init)
    assert np.allclose(model.core.w_rec, w_rec_init)
