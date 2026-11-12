"""Unit tests verifying BennaFusiMetaplasticCascade (Candidate 2)."""
import pytest
import numpy as np

from scarx.neural.core import Config
from scarx.neural.r8_bf_cascade import BennaFusiMetaplasticCascade
from scarx.evaluation.r8_benchmark_integrity import generate_strict_binding, generate_multikey_associative


def test_bf_cascade_initialization():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, seed=42)
    model = BennaFusiMetaplasticCascade(cfg, use_cascade=True, use_context_gating=True)

    assert model.u1_in.shape == (48, 8)
    assert model.u2_in.shape == (48, 8)
    assert model.u1_rec.shape == (48, 8)
    assert model.u2_rec.shape == (48, 8)
    assert model.w_ctx.shape == (2, 2, 48)

    # Memory must be well below 100 KB
    mem = model.memory_bytes()
    assert mem < 100_000


def test_bf_cascade_passive_diffusion():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, seed=42)
    model = BennaFusiMetaplasticCascade(cfg, use_cascade=True, kappa1=0.01, kappa2=0.005)

    x_cal, _ = generate_strict_binding(100, 16)
    model.calibrate(x_cal)

    # Artificially perturb u1 away from u2
    model.u1_in += 0.5
    diff_start = np.linalg.norm(model.u1_in - model.u2_in)

    x_tr, y_tr = generate_strict_binding(200, 4)
    for seq, y in zip(x_tr, y_tr):
        model.learn_episode(seq, int(y))

    diff_end = np.linalg.norm(model.u1_in - model.u2_in)
    # Passive diffusion should draw u1 and u2 closer together
    assert diff_end < diff_start


def test_bf_cascade_context_gating_prediction():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, seed=42)
    model = BennaFusiMetaplasticCascade(cfg, use_cascade=True, use_context_gating=True)

    x_cal, _ = generate_multikey_associative(100, 16, n_keys=2)
    model.calibrate(x_cal)

    x_tr, y_tr = generate_multikey_associative(200, 8, n_keys=2)
    for seq, y in zip(x_tr, y_tr):
        model.learn_episode(seq, int(y))

    preds = model.predict_many(x_tr)
    assert preds.shape == (8,)
    assert set(np.unique(preds)).issubset({0, 1})
