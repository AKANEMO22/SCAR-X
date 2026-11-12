"""Unit tests for the R9c SelectiveSensoryALIF gating architectures."""
from __future__ import annotations

import numpy as np
import pytest

from scarx.neural.core import Config
from scarx.neural.r9c_sensory_gating import GATING_ARCHITECTURES, SelectiveSensoryALIF


def test_instantiation_all_architectures():
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=42)
    for arch in GATING_ARCHITECTURES:
        model = SelectiveSensoryALIF(config=cfg, gating_arch=arch)
        assert model.gating_arch == arch
        assert model.core.w_in.shape == (48, 8)
        assert model.core.w_rec.shape == (48, 8)


def test_step_and_features():
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=42)
    rng = np.random.default_rng(42)
    calib = rng.normal(0, 0.1, (5, 12, 8)).astype(np.float32)
    seq = rng.normal(0, 0.1, (12, 8)).astype(np.float32)

    for arch in GATING_ARCHITECTURES:
        model = SelectiveSensoryALIF(config=cfg, gating_arch=arch)
        model.calibrate(calib)
        f = model.features(seq)
        assert f.shape == (48,)
        assert np.all(np.isfinite(f))
        pred = model.predict(seq)
        assert pred in (0, 1)


def test_learn_episode_and_gate_updates():
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=42)
    rng = np.random.default_rng(42)
    calib = rng.normal(0, 0.1, (5, 12, 8)).astype(np.float32)
    seq = rng.normal(0, 0.1, (12, 8)).astype(np.float32)

    model = SelectiveSensoryALIF(config=cfg, gating_arch="learned_channel")
    model.calibrate(calib)

    initial_logits = model.channel_gate_logits.copy()
    initial_w_in = model.core.w_in.copy()

    loss = model.learn_episode(seq, target=1)
    assert np.isfinite(loss)
    assert model.hidden_updates == 1
    assert model.episodes_learned == 1
    assert not np.array_equal(model.core.w_in, initial_w_in)

    # Test causal revert
    model.revert_hidden()
    np.testing.assert_array_equal(model.core.w_in, model.w_in_0)
