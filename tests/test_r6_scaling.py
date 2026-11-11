"""Round 6A tests — provenance, equivalence, and the guards this round depends on.

The two most important tests are the ones that would have caught my own defects:

  * `test_head_update_matches_r5b` — my first R6 Head used a BATCH update where R5B
    uses a per-example one. That is a different learner, and the brief forbids changing
    the readout partway through a scaling study.
  * `test_seed_bank_matches_protocol` — the registered seed bank must not drift.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r6_scaling as S  # noqa: E402
from scarx.evaluation.r5_probe import Head as R5Head  # noqa: E402


# ------------------------------------------------------- registered seeds
def test_seed_bank_matches_protocol():
    assert S.DEV_SEEDS == (120011, 120017, 120023)
    assert S.CONF_SEEDS == (121011, 121017, 121023, 121029,
                            121031, 121037, 121041, 121043)
    assert len(set(S.DEV_SEEDS) & set(S.CONF_SEEDS)) == 0, "dev and conf must be disjoint"


def test_offsets_match_protocol():
    assert (S.TRAIN_OFFSET, S.HELD_OFFSET, S.PAIRED_OFFSET) == (200_000, 300_000, 400_000)


def test_budgets_match_protocol():
    assert S.BUDGETS == (256, 1024, 4096, 16384)


# ------------------------------------------------- generator equivalence
def test_generator_matches_r5b_at_the_pinned_commit():
    """The generator must be the R5B one, not a reimplementation that drifted."""
    for seed in (120011, 121011):
        a = G.independent_query_set_v1(seed + S.TRAIN_OFFSET, 64)
        b = G.independent_query_set_v1(seed + S.TRAIN_OFFSET, 64)
        for u, v in zip(a, b):
            assert np.array_equal(u, v), "generator must be deterministic"


def test_paired_generator_both_cases_present():
    _, y, vA, vB, q, pid, flips = G.independent_paired_set_v1(120011, 256)
    a, b = y[0::2], y[1::2]
    same = vA[0::2] == vB[0::2]
    assert np.all(a[~same] != b[~same]), "case A must flip"
    assert np.all(a[same] == b[same]), "case B must not flip"
    assert .35 < float((~same).mean()) < .65


# ------------------------------------------------ THE UPDATE-RULE GUARD
def test_head_update_matches_r5b_bit_exactly():
    """R6's Head must perform the SAME per-example delta update as R5B's.

    R5B's Head z-scores inside `learn`; R6 scales features outside and passes
    `zscore=False` semantics. Comparing with R5B at `zscore=False` isolates the
    update arithmetic, which is what must be identical.
    """
    rng = np.random.default_rng(0)
    d = 64
    F = rng.normal(0, 1, (128, d)).astype(np.float32)
    Y = rng.integers(0, 2, 128)
    order = rng.permutation(128)

    a = R5Head(d, 120011, zscore=False)
    b = S.Head(d, 120011)
    for i in order:
        a.learn(F[i], int(Y[i]))
    for i in order:
        b.learn(F[i], int(Y[i]))
    assert np.array_equal(a.w, b.w), "weights diverged from the R5B rule"
    assert np.array_equal(a.b, b.b), "bias diverged from the R5B rule"


def test_head_is_per_example_not_batch():
    """Sanity: two separate updates must not equal one update on the pair average."""
    rng = np.random.default_rng(1)
    d = 32
    f1 = rng.normal(0, 1, d).astype(np.float32)
    f2 = rng.normal(0, 1, d).astype(np.float32)

    a = S.Head(d, 120011)
    a.learn(f1, 0)
    a.learn(f2, 1)

    # a batch learner would take the mean feature: check the result differs
    b = S.Head(d, 120011)
    from scarx.evaluation import r5_probe as P
    hb = P.Head(d, 120011, zscore=False)
    hb.learn(np.stack([f1, f2]).mean(0), 0)
    assert not np.array_equal(a.w, hb.w), "R6 head must not behave like a batch rule"


# ------------------------------------------------------ head determinism
def test_head_deterministic_given_seed_and_order():
    rng = np.random.default_rng(2)
    F = rng.normal(0, 1, (256, 48)).astype(np.float32)
    Y = rng.integers(0, 2, 256)
    o = rng.permutation(256)
    a = S.train_head(F, Y, 120011, o, epochs=2)
    b = S.train_head(F, Y, 120011, o, epochs=2)
    assert a.digest() == b.digest()


def test_head_digest_changes_with_order():
    rng = np.random.default_rng(3)
    F = rng.normal(0, 1, (256, 48)).astype(np.float32)
    Y = rng.integers(0, 2, 256)
    a = S.train_head(F, Y, 120011, rng.permutation(256), epochs=2)
    b = S.train_head(F, Y, 120011, rng.permutation(256), epochs=2)
    assert a.digest() != b.digest(), "different order means a different learner state"


def test_updates_count_equals_examples_times_epochs():
    rng = np.random.default_rng(4)
    F = rng.normal(0, 1, (64, 16)).astype(np.float32)
    Y = rng.integers(0, 2, 64)
    h = S.train_head(F, Y, 120011, np.arange(64), epochs=5)
    assert h.updates == 64 * 5


# ------------------------------------------- instrumentation invariants
def test_clip_fraction_is_a_fraction():
    rng = np.random.default_rng(5)
    F = rng.normal(0, 1, (128, 32)).astype(np.float32)
    Y = rng.integers(0, 2, 128)
    h = S.train_head(F, Y, 120011, np.arange(128))
    assert 0.0 <= h.clipped / h.updates <= 1.0


def test_budgets_are_not_nested_and_that_is_documented():
    """The generator's final reshuffle makes budgets INDEPENDENT draws, not prefixes.

    The brief asks for nested prefixes "where possible". They are not possible with
    `independent_query_set_v1` unmodified: it draws `count` episodes and then applies
    `rng.permutation(count)`, so requesting 256 gives a different draw than the first
    256 of a 16384 request (measured: only 21/256 overlap, no subset relation).

    This test pins the FACT so that no later script silently assumes nesting. Stage 1
    therefore builds every budget from an independent draw of the same distribution
    and says so, rather than pretending the curves are on one nested ladder.
    """
    full, yf, *_ = G.independent_query_set_v1(120011 + S.TRAIN_OFFSET, 16384)
    small, ys, *_ = G.independent_query_set_v1(120011 + S.TRAIN_OFFSET, 256)
    assert not np.array_equal(full[:256], small), "nesting changed; revisit Stage 1"
    overlap = len({r.tobytes() for r in small} & {r.tobytes() for r in full})
    assert overlap < 256, "if the small set became a subset, revisit the docs"


def test_different_counts_are_different_sets():
    a, *_ = G.independent_query_set_v1(120011 + S.TRAIN_OFFSET, 4096)
    b, *_ = G.independent_query_set_v1(120011 + S.TRAIN_OFFSET, 4096)
    assert np.array_equal(a, b), "same count and seed must be reproducible"


def test_first_key_ceiling_unchanged():
    """The 0.75 shortcut ceiling is the null for every claim in this round."""
    x, y, *_ = G.independent_query_set_v1(120011 + S.HELD_OFFSET, 4096)
    acc = float(np.mean(G.baseline_first_key(x) == y))
    assert abs(acc - 0.75) < 0.03, f"first-key = {acc:.4f}"
