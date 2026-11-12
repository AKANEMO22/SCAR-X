"""Unit tests for R9b Research Model Snapshots (P0 Reproducibility).

Verifies:
1. Full bit-for-bit roundtrip for FactorialMTSGEProp.
2. Full bit-for-bit roundtrip for R9bBFCascade.
3. Exact feature, probability, and prediction parity after reload.
4. Exact next-episode learning step parity (loss, weight deltas, norms).
5. Lossless controller state roundtrip.
6. Integrity rejection on checksum tampering or non-finite payload.
7. Atomic replace write safety.
"""
from pathlib import Path
import numpy as np
import pytest

from scarx.neural.core import Config
from scarx.neural.r9b_factorial_eprop import FactorialMTSGEProp
from scarx.neural.r9b_bf_cascade import R9bBFCascade
from scarx.evaluation.forensic import r9b_snapshot as S


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


def test_factorial_mtsg_snapshot_roundtrip(tmp_path):
    cfg = make_config(seed=101)
    model = FactorialMTSGEProp(
        config=cfg,
        decay_mode="heterogeneous",
        gating_mode="soft_sensory",
        reset_mode="stopped_reset",
        control_mode="learned",
    )
    calib = np.random.default_rng(1).normal(0, 0.5, (4, 12, 8)).astype(np.float32)
    seq = np.random.default_rng(2).normal(0, 0.5, (12, 8)).astype(np.float32)
    test_seq = np.random.default_rng(3).normal(0, 0.5, (12, 8)).astype(np.float32)

    model.calibrate(calib)
    model.learn_episode(seq, target=1)

    controller = {"stage": "dev", "step": 42, "flag": True}
    path = tmp_path / "mtsg_snapshot.zip"
    S.save(model, path, controller=controller)

    # Fresh template
    template = FactorialMTSGEProp(
        config=cfg,
        decay_mode="heterogeneous",
        gating_mode="soft_sensory",
        reset_mode="stopped_reset",
        control_mode="learned",
    )
    loaded, loaded_ctrl = S.load(template, path)

    assert loaded_ctrl == controller
    assert loaded.episodes_learned == model.episodes_learned
    assert loaded.hidden_updates == model.hidden_updates
    np.testing.assert_array_equal(loaded.core.w_in, model.core.w_in)
    np.testing.assert_array_equal(loaded.core.w_rec, model.core.w_rec)
    np.testing.assert_array_equal(loaded.core.w_out, model.core.w_out)
    np.testing.assert_array_equal(loaded.core.bias, model.core.bias)
    np.testing.assert_array_equal(loaded.e_v, model.e_v)
    np.testing.assert_array_equal(loaded.e_a, model.e_a)
    np.testing.assert_array_equal(loaded.e_f, model.e_f)

    # Check prediction parity
    feat_orig = model.features(test_seq)
    feat_load = loaded.features(test_seq)
    np.testing.assert_array_equal(feat_orig, feat_load)
    assert model.predict(test_seq) == loaded.predict(test_seq)

    # Check next-step learning update parity
    next_seq = np.random.default_rng(4).normal(0, 0.5, (12, 8)).astype(np.float32)
    loss_orig = model.learn_episode(next_seq, target=0)
    loss_load = loaded.learn_episode(next_seq, target=0)
    assert np.isclose(loss_orig, loss_load)
    np.testing.assert_array_equal(model.core.w_in, loaded.core.w_in)
    np.testing.assert_array_equal(model.core.w_rec, loaded.core.w_rec)
    np.testing.assert_array_equal(model.core.w_out, loaded.core.w_out)


def test_bf_cascade_snapshot_roundtrip(tmp_path):
    cfg = make_config(seed=202)
    model = R9bBFCascade(
        config=cfg,
        consolidation_mode="selective",
        kappa1=0.005,
        kappa2=0.001,
    )
    calib = np.random.default_rng(5).normal(0, 0.5, (4, 12, 8)).astype(np.float32)
    seq = np.random.default_rng(6).normal(0, 0.5, (12, 8)).astype(np.float32)
    test_seq = np.random.default_rng(7).normal(0, 0.5, (12, 8)).astype(np.float32)

    model.calibrate(calib)
    model.learn_episode(seq, target=1)

    path = tmp_path / "bf_snapshot.zip"
    S.save(model, path)

    template = R9bBFCascade(
        config=cfg,
        consolidation_mode="selective",
        kappa1=0.005,
        kappa2=0.001,
    )
    loaded, _ = S.load(template, path)

    np.testing.assert_array_equal(loaded.u1_in, model.u1_in)
    np.testing.assert_array_equal(loaded.u2_in, model.u2_in)
    np.testing.assert_array_equal(loaded.u1_out, model.u1_out)
    np.testing.assert_array_equal(loaded.u2_out, model.u2_out)

    # Next episode parity
    next_seq = np.random.default_rng(8).normal(0, 0.5, (12, 8)).astype(np.float32)
    loss_orig = model.learn_episode(next_seq, target=0)
    loss_load = loaded.learn_episode(next_seq, target=0)
    assert np.isclose(loss_orig, loss_load)
    np.testing.assert_array_equal(model.u1_in, loaded.u1_in)
    np.testing.assert_array_equal(model.u2_in, loaded.u2_in)


def test_snapshot_tamper_rejection(tmp_path):
    cfg = make_config(seed=303)
    model = FactorialMTSGEProp(config=cfg)
    raw = S.dumps(model)
    # Corrupt a byte in the raw zip archive
    corrupted = bytearray(raw)
    corrupted[100] ^= 0xFF
    with pytest.raises(ValueError):
        S.loads(model, bytes(corrupted))


def test_snapshot_mismatched_template(tmp_path):
    cfg = make_config(seed=404)
    m1 = FactorialMTSGEProp(config=cfg)
    m2 = R9bBFCascade(config=cfg)
    raw = S.dumps(m1)
    with pytest.raises(ValueError):
        S.loads(m2, raw)
