"""Round 5B tests — the safeguards this round depends on.

The most important tests here are the two that would have caught my own defects:

  * `test_zero_pad_shift_is_a_noop` — proves the naive temporal shift does nothing to
    a zero-initialised recurrent net. My first confirmation run was invalid for exactly
    this reason.
  * `test_first_key_is_075` — proves the shortcut ceiling, which is the whole reason
    this round cannot use 0.50 as the null.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r5_probe as P  # noqa: E402
from scarx.neural.core import SpikingCore  # noqa: E402


def data(n=256, seed=99001):
    return G.independent_query_set_v1(seed, n)


# ------------------------------------------------------- the 0.75 ceiling
def test_first_key_is_075():
    """alwayanswer-key-A must score ~0.75, not 0.50. This is the round's null."""
    x, y, *_ = data(4096)
    acc = float(np.mean(G.baseline_first_key(x) == y))
    assert abs(acc - .75) < .03, f"first-key = {acc:.4f}, expected ~0.75"


def test_majority_and_query_only_are_050():
    x, y, *_ = data(4096)
    assert abs(float(np.mean(G.baseline_majority(len(y)) == y)) - .5) < .03
    assert abs(float(np.mean(G.baseline_query_only(x) == y)) - .5) < .03


def test_knowing_both_values_without_query_is_075():
    x, y, vA, vB, q, _ = data(4096)
    pred = np.where(vA == vB, vA, 0)
    assert abs(float(np.mean(pred == y)) - .75) < .03


# ------------------------------------------------ generator independence
def test_values_are_independent():
    x, y, vA, vB, q, _ = data(8192)
    assert abs(float(np.corrcoef(vA, vB)[0, 1])) < .05
    # the R4A complementary pattern must not be forced
    assert abs(float(np.mean((vA == 0) & (vB == 1))) - .25) < .04


def test_all_four_memory_states_uniform():
    _, _, vA, vB, _, _ = data(8192)
    c = np.bincount(vA * 2 + vB, minlength=4) / len(vA)
    assert np.all(np.abs(c - .25) < .04), c


def test_query_uniform():
    _, _, _, _, q, _ = data(8192)
    c = np.bincount(q, minlength=2) / len(q)
    assert np.all(np.abs(c - .5) < .04), c


def test_oracle_matches_generator():
    x, y, *_ = data(512)
    assert np.array_equal(G.oracle_all(x), y)


# --------------------------------------------- counterfactual both cases
def test_case_a_flips_case_b_does_not():
    x, y, vA, vB, q, pid, flips = G.independent_paired_set_v1(99001, 512)
    a, b = y[0::2], y[1::2]
    same = vA[0::2] == vB[0::2]
    assert np.all(a[~same] != b[~same]), "case A must flip"
    assert np.all(a[same] == b[same]), "case B must NOT flip"
    # both cases must be substantial
    assert .35 < float((~same).mean()) < .65


def test_siblings_differ_only_at_query_tick():
    x, *_ = G.independent_paired_set_v1(99001, 256)
    ev, od = x[0::2], x[1::2]
    d = np.abs(ev - od)
    assert np.all(d[:, :G.QUERY_TICK, :] == 0)
    assert np.all(d[:, G.QUERY_TICK + 1:, :] == 0)
    assert d[:, G.QUERY_TICK, :].max() > 0


# -------------------------------------- THE TRANSFORM NO-OP GUARD
def test_zero_pad_shift_is_a_noop():
    """A zero-initialised recurrent net cannot see a zero-pad shift.

    This test exists because my first Stage-4 confirmation was invalid for exactly
    this reason: three conditions reported identical accuracy to four decimals.
    """
    x, *_ = data(64)
    z = G.shifted_zero_pad(x, 1)
    f_clean = P.state_vector(P.capture(99001, x, query_tick=10), "post")
    f_shift = P.state_vector(P.capture(99001, z, query_tick=11), "post")
    assert np.array_equal(f_clean, f_shift), (
        "zero-pad shift changed the state; if this fails the hazard has changed and "
        "the reasoning in math-analysis.md section 7 must be revisited")


def test_noise_pad_shift_is_a_real_perturbation():
    x, *_ = data(64)
    s = G.shifted(x, 1, seed=5)
    f_clean = P.state_vector(P.capture(99001, x, query_tick=10), "post")
    f_shift = P.state_vector(P.capture(99001, s, query_tick=11), "post")
    assert np.abs(f_clean - f_shift).max() > .1, "shift must actually change the state"


def test_shift_preserves_labels():
    x, y, *_ = data(256)
    for shift in (1, 2):
        if G.QUERY_TICK + shift >= G.N_TICKS:
            continue
        s = G.shifted(x, shift, seed=3)
        assert np.array_equal(G.oracle_all(s), y)


def test_reversed_writes_and_distractors_preserve_labels():
    x, y, *_ = data(256)
    assert np.array_equal(G.oracle_all(G.reversed_writes(x)), y)
    assert np.array_equal(G.oracle_all(G.with_distractors(x, 7)), y)


# ---------------------------------------------- oracle is shift-safe
def test_oracle_locates_its_markers():
    x, y, *_ = data(128)
    s = G.shifted(x, 1, seed=9)
    assert G.locate_query_tick(s[0]) == G.QUERY_TICK + 1
    assert np.array_equal(G.oracle_all(s), y)


def test_query_cue_follows_the_marker():
    x, *_ = data(128)
    base = G.query_cue(x)
    shifted = G.query_cue(G.shifted(x, 1, seed=4))
    assert np.array_equal(base, shifted), "cue content must not depend on tick index"


# ------------------------------------------------- no leakage, frozen
def test_features_are_label_independent():
    x, y, *_ = data(128)
    a = P.capture(99001, x, query_tick=10)
    b = P.capture(99001, x, query_tick=10)
    for w in ("pre", "post"):
        for k in a[w]:
            assert np.array_equal(a[w][k], b[w][k])


def test_hidden_weights_frozen_across_head_training():
    import hashlib
    x, y, *_ = data(128)
    _, _, w0, r0 = P.weights(99001)
    h0 = hashlib.sha256(w0.tobytes() + r0.tobytes()).hexdigest()
    cap = P.capture(99001, x, query_tick=10)
    P.train_head(cap["final"]["spikesum"], y)
    _, _, w1, r1 = P.weights(99001)
    assert h0 == hashlib.sha256(w1.tobytes() + r1.tobytes()).hexdigest()


def test_forward_parity_with_spikingcore():
    x, *_ = data(8)
    cap = P.capture(99001, x, query_tick=10)
    base = SpikingCore(P.cfg(99001))
    for i in range(3):
        base.reset_state()
        for t in range(G.N_TICKS):
            base.step(x[i, t], track=False)
            assert np.array_equal(base.spikes, cap["spike_train"][i, t])


# ------------------------------------------------- train/test hygiene
def test_splits_are_disjoint():
    a, ya, *_ = G.independent_query_set_v1(99001 + 200_000, 256)
    b, yb, *_ = G.independent_query_set_v1(99001 + 300_000, 256)
    assert G.dataset_sha256(a, ya) != G.dataset_sha256(b, yb)
    assert len({r.tobytes() for r in a} & {r.tobytes() for r in b}) == 0
