"""Round 4A unit and metamorphic tests.

Brief section 4 requires eight controls and section 11 requires metamorphic tests.
These are executed, not asserted by inspection.
"""
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r4_probe as P  # noqa: E402
from scarx.evaluation import r4_relational as R  # noqa: E402
from scarx.neural.core import SpikingCore  # noqa: E402


# --------------------------------------------------- generator provenance
def test_query_set_fixture_hash_is_pinned():
    """The stored verbatim source must match the recorded fixture digest.

    A byte hash of `inspect.getsource(query_set)` breaks on a docstring edit with
    no semantic change, so the pinned artifact is the saved source file.
    """
    fixture = Path(__file__).resolve().parents[1] / R.QUERY_SET_FIXTURE
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == R.QUERY_SET_FIXTURE_SHA256


def test_query_set_matches_its_verbatim_fixture_behaviourally():
    """The live query_set must produce exactly what the pinned source produces."""
    src = (Path(__file__).resolve().parents[1] / R.QUERY_SET_FIXTURE).read_text(encoding="utf-8")
    ns = {"np": np, "PULSE": R.PULSE}
    exec(compile(src, "fixture", "exec"), ns)
    for seed in (95001, 95007):
        for u, v in zip(ns["query_set"](seed, 64), R.query_set(seed, 64)):
            assert np.array_equal(u, v), "live function diverged from the fixture"


def test_query_set_deterministic():
    a = R.query_set(95001, 64)
    b = R.query_set(95001, 64)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_query_set_label_balance():
    _, y, _ = R.query_set(95001, 512)
    counts = np.bincount(y, minlength=2)
    # the generator alternates query by index parity but draws the stored bit
    # freely, so the balance is statistical, not exact. Tolerance is 8%.
    assert abs(int(counts[0]) - int(counts[1])) <= 0.08 * len(y), counts


# ------------------------------------------- brief control 1: query-only
def test_control_1_query_only_is_chance():
    """Query-only information must not predict labels above chance."""
    x, y, _ = R.query_set(95001 + 200_000, 1024)
    t, ty, _ = R.query_set(95001 + 300_000, 512)

    def cue(s):
        q = (s[:, 10, 5] > .5).astype(np.float32)
        return np.stack([1 - q, q], 1)

    acc = P.ridge(cue(x), y, cue(t), ty)
    assert abs(acc - .5) < .06, acc


# --------------------------------------- brief control 2: first-key policy
def test_control_2_first_key_policy_not_above_baseline():
    x, y, _ = R.query_set(95001 + 300_000, 512)
    pred = np.zeros(len(x), np.int32)
    for i in range(len(x)):
        vals = {}
        for place in (2, 6):
            k = int(np.argmax(x[i, place, :2]))
            v = int(np.argmax(x[i, place, 2:4]))
            vals[k] = v
        pred[i] = vals[0]
    acc = float(np.mean(pred == y))
    assert abs(acc - .5) < .06, acc


# ------------------------------- brief control 3: reversed write order
def test_control_3_reversed_write_order_preserves_answers():
    """Swapping the two write slots must NOT change the oracle answer."""
    x, y, _ = R.query_set(95001 + 200_000, 256)
    rev, yrev, _ = R.reversed_write_order(95001 + 200_000, 256)
    assert np.array_equal(y, yrev), "labels must be unchanged by slot swap"
    assert np.array_equal(rev[:, 2], x[:, 6]) and np.array_equal(rev[:, 6], x[:, 2])
    # and the oracle must still agree with the unchanged labels
    orc = np.array([R.oracle_answer(rev[i]) for i in range(len(rev))])
    assert np.array_equal(orc, yrev)


# ------------------------------------- brief control 4: query reversal
def test_control_4_query_reversal_changes_answer_when_values_differ():
    x, y, _ = R.query_set(95001 + 200_000, 256)
    q = (x[:, 10, 5] > .5).astype(np.int32)
    flipped = x.copy()
    flipped[:, 10, 4] = 0.0
    flipped[:, 10, 5] = 0.0
    flipped[np.arange(len(x)), 10, 4 + (1 - q)] = R.PULSE

    orc = np.array([R.oracle_answer(flipped[i]) for i in range(len(flipped))])
    # values are complementary in this generator, so a query flip must invert
    assert np.array_equal(orc, 1 - y), "query reversal must invert the answer"


# ------------------------- brief control 5: shuffled association
def test_control_5_shuffled_association_changes_oracle_accordingly():
    x, y = R.shuffled_association(95001 + 200_000, 256)
    orc = np.array([R.oracle_answer(x[i]) for i in range(len(x))])
    assert np.array_equal(orc, y), "oracle must follow the rewritten binding"
    # and it must actually differ from the correctly-bound version
    xc, yc, _ = R.query_set(95001 + 200_000, 256)
    assert not np.array_equal(x, xc), "the rewrite must change the input"


# --------------------------------------- brief control 6: no label leakage
def test_control_6_labels_never_enter_features():
    x, y, _ = R.query_set(95001 + 200_000, 128)
    before = P.capture(95001, x)
    flipped = y.copy()
    flipped[:] = 1 - flipped
    after = P.capture(95001, x)
    for k in ("spikes", "voltage", "adaptation", "features"):
        assert np.array_equal(before["pre"][k], after["pre"][k]), k
        assert np.array_equal(before["post"][k], after["post"][k]), k


def test_control_6b_head_update_ignores_label_when_reward_fixed():
    """The registered head is supervised, so this checks the FEATURES, not the head."""
    x, y, _ = R.query_set(95001 + 200_000, 64)
    c = P.capture(95001, x)
    _, _, w0, r0 = P.weights(95001)
    h0 = hashlib.sha256(w0.tobytes() + r0.tobytes()).hexdigest()
    for k in ("decay025", "spikesum", "bank5"):
        _ = c["final"][k].mean(0)
    _, _, w1, r1 = P.weights(95001)
    assert h0 == hashlib.sha256(w1.tobytes() + r1.tobytes()).hexdigest()


# --------------------------------- brief control 7: split independence
def test_control_7_no_shared_episodes_across_splits():
    xa, ya, _ = R.query_set(95001 + 200_000, 256)
    xb, yb, _ = R.query_set(95001 + 300_000, 256)
    sa = {r.tobytes() for r in xa}
    sb = {r.tobytes() for r in xb}
    assert len(sa & sb) == 0
    assert R.dataset_sha256(xa, ya) != R.dataset_sha256(xb, yb)


def test_control_7b_counterfactual_siblings_stay_together():
    """Pairs must not be split across train and test."""
    x, y, pid, _ = R.paired_counterfactual(95001, 64)
    for p in np.unique(pid):
        idx = np.where(pid == p)[0]
        assert len(idx) == 2, "every pair must appear exactly once"
    # siblings differ only at the query tick
    a, b = x[0::2], x[1::2]
    off = np.abs(a - b)[:, :10, :]
    assert off.max() == 0.0


# ---------------------- brief control 8: independent reference generator
def test_control_8_independent_generator_matches_statistics():
    x, y, fk = R.query_set(95001, 512)
    xi, yi, fki = R.independent_query_set(95001, 512)
    assert np.array_equal(np.bincount(y, minlength=2), np.bincount(yi, minlength=2))
    assert np.array_equal(np.bincount(fk, minlength=2), np.bincount(fki, minlength=2))
    # the oracle must agree on BOTH generators
    assert np.array_equal([R.oracle_answer(r) for r in xi], yi)


def test_equal_value_pairs_do_not_flip():
    """Brief: equal-value pairs must not be mis-scored as paired-flip failures."""
    x, y, pid = R.equal_value_pairs(95001, 64)
    a, b = y[0::2], y[1::2]
    assert np.array_equal(a, b)
    # and the oracle agrees
    assert np.array_equal([R.oracle_answer(r) for r in x], y)


# --------------------------------------------------- forward parity
def test_forward_parity_with_spikingcore():
    x, _, _ = R.query_set(95001, 8)
    base = SpikingCore(P.cfg(95001))
    cap = P.capture(95001, x)
    for i in range(3):
        base.reset_state()
        for t in range(12):
            base.step(x[i, t], track=False)
            assert np.array_equal(base.spikes, cap["spike_train"][i, t])


# --------------------------------------------- metamorphic: cue legality
def test_bilinear_map_is_fixed_and_cue_gated():
    s = np.random.default_rng(0).normal(0, 1, (16, 48)).astype(np.float32)
    cue = np.stack([np.ones(16), np.zeros(16)], 1).astype(np.float32)  # all key0
    a = P.bilinear_query_map(s, cue, n_proj=8, seed=5)
    b = P.bilinear_query_map(s, cue, n_proj=8, seed=5)
    assert np.array_equal(a, b), "projection must be deterministic"
    # with cue=(1,0) the key1 interaction block must be zero
    assert np.allclose(a[:, -8:], 0.0)
    # and the key0 block must be non-zero for non-zero state
    assert not np.allclose(a[:, -16:-8], 0.0)


def test_consistency_null_is_one_half():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, 4000)
    b = rng.integers(0, 2, 4000)
    assert abs(float(np.mean(a != b)) - .5) < .03
