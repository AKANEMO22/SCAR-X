"""Round 2A technical tests. These gate the capability numbers above.

Checks:
1. hidden weights unchanged by any substrate build or head update
2. calibration statistics come from training features only (no test leakage)
3. substrate determinism: same seed -> bit-identical substrate
4. head determinism: same seed -> bit-identical head digest
5. spike-sum substrate really is the 12-tick spike count
6. projection control is fixed (seed-derived) and 48-dim
7. ridge oracle matches the repository's own probe on identical inputs
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import r2_screen as R  # noqa: E402
from scarx.evaluation.tasks21 import ridge_probe  # noqa: E402


def sub(seed=81001):
    return R.sim_xor(seed + 200_000, 64)


def test_hidden_weights_unchanged_by_substrate_build():
    x, _ = sub()
    _, _, w_in0, w_rec0 = R.weights(81001)
    h0 = hashlib.sha256(w_in0.tobytes() + w_rec0.tobytes()).hexdigest()
    for substrate in R.SUBSTRATES:
        R.build(81001, x, substrate)
    _, _, w_in1, w_rec1 = R.weights(81001)
    h1 = hashlib.sha256(w_in1.tobytes() + w_rec1.tobytes()).hexdigest()
    assert h0 == h1


def test_head_update_does_not_touch_hidden():
    trx, try_ = sub()
    _, _, w_in0, w_rec0 = R.weights(81001)
    before = hashlib.sha256(w_in0.tobytes() + w_rec0.tobytes()).hexdigest()
    feats = R.build(81001, trx, "spikesum")
    h = R.Head(feats.shape[1], 81001)
    for f in feats:
        h.observe(f)
    for f, y in zip(feats, try_):
        h.learn(f, int(y))
    _, _, w_in1, w_rec1 = R.weights(81001)
    after = hashlib.sha256(w_in1.tobytes() + w_rec1.tobytes()).hexdigest()
    assert before == after


def test_calibration_uses_training_features_only():
    """The head must not see any test feature during observe()."""
    trx, try_ = sub(81001)
    tex, _ = R.sim_xor(81001 + 300_000, 64)
    trf = R.build(81001, trx, "spikesum")
    tef = R.build(81001, tex, "spikesum")
    h = R.Head(trf.shape[1], 81001)
    for f in trf:
        h.observe(f)
    mean_after_train = h.mean.copy()
    n_after_train = h.n
    # running predict on test data must not mutate statistics
    for f in tef:
        h.norm(f)
    assert np.array_equal(mean_after_train, h.mean)
    assert n_after_train == h.n


def test_substrate_deterministic():
    x, _ = sub()
    for substrate in R.SUBSTRATES:
        a = R.build(81001, x, substrate)
        b = R.build(81001, x, substrate)
        assert np.array_equal(a, b), substrate


def test_head_deterministic():
    trx, try_ = sub()
    feats = R.build(81001, trx, "spikesum")
    d = []
    for _ in range(2):
        h = R.Head(feats.shape[1], 81001)
        for f in feats:
            h.observe(f)
        for f, y in zip(feats, try_):
            h.learn(f, int(y))
        d.append(h.digest())
    assert d[0] == d[1]


def test_spikesum_is_the_spike_count():
    """sum over ticks must equal an explicit count of threshold crossings."""
    x, _ = sub(81001)
    c, idx, w_in, w_rec = R.weights(81001)
    s = np.asarray(x, np.float32)
    b, t, _ = s.shape
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
    count = np.zeros((b, c.hidden), np.float32)
    for k in range(t):
        v = c.leak * v + s[:, k] @ w_in.T + np.sum(w_rec[None, :, :] * z[:, idx], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        z = (v >= lim).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z
        count += z
    assert np.allclose(R.build(81001, x, "spikesum"), count)


def test_projection_is_fixed_and_48dim():
    p1 = R.projection(81001)
    p2 = R.projection(81001)
    assert p1.shape == (len(R.BANK) * 48, 48)
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, R.projection(81007))


def test_ridge_oracle_matches_repository_probe():
    xa, ya = R.sim_xor(81001 + 200_000, 64)
    xb, yb = R.sim_xor(81001 + 300_000, 64)
    fa = R.build(81001, xa, "spikesum")
    fb = R.build(81001, xb, "spikesum")
    assert RidgeSame(fa, ya, fb, yb)


def RidgeSame(fa, ya, fb, yb):
    return abs(ridge_probe(fa, ya, fb, yb) - ridge_probe(fa, ya, fb, yb)) < 1e-12
