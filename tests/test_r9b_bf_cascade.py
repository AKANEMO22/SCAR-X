"""Unit tests for R9bBFCascade in SCAR-X Round 9b."""
import pytest
import numpy as np

from scarx.neural.core import Config
from scarx.neural.r9b_bf_cascade import R9bBFCascade, CONSOLIDATION_MODES


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


def test_bf_cascade_consolidation_modes():
    cfg = make_config()
    for mode in CONSOLIDATION_MODES:
        m = R9bBFCascade(config=cfg, consolidation_mode=mode)
        assert m.consolidation_mode == mode


def test_bf_cascade_readout_only_vs_hidden_only():
    cfg = make_config(seed=101)
    seq = np.random.default_rng(1).normal(0, 0.5, (12, 8)).astype(np.float32)
    calib = np.random.default_rng(2).normal(0, 0.5, (4, 12, 8)).astype(np.float32)

    ro = R9bBFCascade(config=cfg, consolidation_mode="readout_only", kappa1=0.1, kappa2=0.05)
    ho = R9bBFCascade(config=cfg, consolidation_mode="hidden_only", kappa1=0.1, kappa2=0.05)
    none = R9bBFCascade(config=cfg, consolidation_mode="none")

    ro.calibrate(calib)
    ho.calibrate(calib)
    none.calibrate(calib)

    ro.learn_episode(seq, target=1)
    ho.learn_episode(seq, target=1)
    none.learn_episode(seq, target=1)

    # In readout_only: hidden u2 reservoir must remain at initial state!
    np.testing.assert_array_equal(ro.u2_in, none.u2_in)
    # Check that in ho, hidden u2 moved while readout u2 stayed unchanged
    assert not np.array_equal(ho.u2_in, none.u2_in)
    np.testing.assert_array_equal(ho.u2_out, none.u2_out)
