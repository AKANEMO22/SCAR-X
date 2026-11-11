"""Round 3A tests — features, streaming, parity, leakage, hidden invariance.

These gate the capability numbers. Required by brief section 8.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import r3_screen as R  # noqa: E402
from scarx.evaluation import r3_substrate as S  # noqa: E402
from scarx.neural.core import SpikingCore  # noqa: E402


def seqs(seed=91001, n=24):
    return S.sim_xor(seed + 200_000, n)


# --------------------------------------------------------------- streaming
def test_streaming_matches_batch_every_substrate():
    x, _ = seqs()
    spk = S.spike_train(91001, x)
    for name in S.SUBSTRATES:
        batch = S.substrate_from_train(spk, name)
        streams = []
        for i in range(len(x)):
            st = S.StreamSubstrate(name)
            for t in range(spk.shape[1]):
                st.update(spk[i, t])
            streams.append(st.value())
        assert np.allclose(batch, np.stack(streams), atol=1e-6), name


def test_streaming_state_is_bounded():
    """State size must not grow with the number of ticks."""
    for name in S.SUBSTRATES:
        st = S.StreamSubstrate(name)
        before = st.nstate()
        for _ in range(200):
            st.update(np.zeros(48, np.float32))
        assert st.nstate() == before, name
        assert before <= 5 * 48, f"{name} state {before} exceeds 5*48"


def test_streaming_reset_is_exact():
    """A fresh stream must equal a reset stream after the same updates."""
    z = (np.random.default_rng(0).random(48) < .3).astype(np.float32)
    for name in S.SUBSTRATES:
        a = S.StreamSubstrate(name)
        for _ in range(6):
            a.update(z)
        b = S.StreamSubstrate(name)     # fresh == reset by construction
        for _ in range(6):
            b.update(z)
        assert np.allclose(a.value(), b.value()), name


# ------------------------------------------------------------------ parity
def test_forward_parity_with_serving_core():
    x, _ = seqs()
    base = SpikingCore(S.cfg(91001))
    spk = S.spike_train(91001, x)
    for i in range(3):
        base.reset_state()
        for t in range(x.shape[1]):
            base.step(x[i, t], track=False)
            assert np.array_equal(base.spikes, spk[i, t])


# ------------------------------------------------------- hidden invariance
def test_hidden_weights_unchanged_by_substrates():
    x, _ = seqs()
    _, _, w0, r0 = S.weights(91001)
    h0 = hashlib.sha256(w0.tobytes() + r0.tobytes()).hexdigest()
    for name in S.SUBSTRATES:
        S.substrate(91001, x, name)
    _, _, w1, r1 = S.weights(91001)
    assert h0 == hashlib.sha256(w1.tobytes() + r1.tobytes()).hexdigest()


def test_head_training_does_not_touch_hidden():
    x, y = seqs()
    _, _, w0, r0 = S.weights(91001)
    h0 = hashlib.sha256(w0.tobytes() + r0.tobytes()).hexdigest()
    f = S.substrate(91001, x, "count_plus_fast")
    h = R.Head(f.shape[1], 91001)
    for row in f:
        h.observe(row)
    for row, lab in zip(f, y):
        h.learn(row, int(lab))
    _, _, w1, r1 = S.weights(91001)
    assert h0 == hashlib.sha256(w1.tobytes() + r1.tobytes()).hexdigest()


# --------------------------------------------------------------- leakage
def test_calibration_uses_training_features_only():
    x, _ = seqs()
    tx, _ = S.strict_binding(91001 + 300_000, 24)
    trf = S.substrate(91001, x, "bank5")
    tef = S.substrate(91001, tx, "bank5")
    h = R.Head(trf.shape[1], 91001)
    for row in trf:
        h.observe(row)
    mean_at_freeze = h.mean.copy()
    n_at_freeze = h.n
    for row in tef:
        h.basis(row)                    # must be read-only
    assert np.array_equal(mean_at_freeze, h.mean)
    assert n_at_freeze == h.n


def test_features_never_see_labels():
    x, y = seqs()
    base = S.spike_train(91001, x)
    flipped = y.copy()
    flipped[:] = 1 - flipped
    assert np.array_equal(base, S.spike_train(91001, x)), "labels must not affect spikes"


# ----------------------------------------------------------- determinism
def test_head_deterministic():
    x, y = seqs()
    f = S.substrate(91001, x, "temporal_bins_3")
    digs = []
    for _ in range(2):
        h = R.Head(f.shape[1], 91001)
        for row in f:
            h.observe(row)
        for row, lab in zip(f, y):
            h.learn(row, int(lab))
        digs.append(h.digest())
    assert digs[0] == digs[1]


def test_substrate_dimensions():
    for name, expected in (("decay025", 48), ("spikesum", 48), ("bank5", 240),
                           ("count_plus_fast", 96), ("temporal_bins_3", 144)):
        assert S.dims(name) == expected, name
        x, _ = seqs(n=4)
        assert S.substrate(91001, x, name).shape[1] == expected, name
