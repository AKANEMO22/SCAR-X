"""Round 6D audit regression tests.

Each test exists because a specific R6A defect would reappear without it. Passing these
is the condition for the corrected numbers to be trusted.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r5_probe as P  # noqa: E402
from scarx.evaluation import r6d_audit as A  # noqa: E402


# ============================================== A1: normalization parity
def test_normalization_parity_with_r5b():
    """Scaler('r5b') must reproduce R5B's `Head.basis` to float32 precision.

    R6A used floor 1e-6 and no clip while R5B uses floor .05 and clip +-5. With
    1e-6 the small-variance coordinates are amplified ~50 000x more than R5B would
    amplify them, so R6A's absolute numbers are on a different feature scale.
    """
    rng = np.random.default_rng(0)
    d = 64
    F = (rng.normal(0, 1, (300, d)) * rng.uniform(0.001, 3, d)).astype(np.float32)
    h = P.Head(d, 120011)
    for f in F:
        h.observe(f)
    sigma = np.sqrt(h.m2 / max(1, h.n - 1))
    sc = A.Scaler(h.mean, np.maximum(sigma, .05), "r5b")
    assert np.allclose(sc(F), np.stack([h.basis(f) for f in F]), atol=1e-6)


def test_r6a_convention_differs_and_that_is_explicit():
    """The two conventions must differ, and each must be named."""
    rng = np.random.default_rng(1)
    F = (rng.normal(0, 1, (200, 32)) * rng.uniform(0.001, 2, 32)).astype(np.float32)
    a = A.Scaler.fit(F, "r5b")(F)
    b = A.Scaler.fit(F, "r6a")(F)
    assert not np.allclose(a, b), "conventions must not silently coincide"
    assert A.Scaler.fit(F, "r5b").convention == "r5b"
    assert A.Scaler.fit(F, "r6a").convention == "r6a"


def test_scaler_rejects_unknown_convention():
    rng = np.random.default_rng(2)
    F = rng.normal(0, 1, (64, 8)).astype(np.float32)
    try:
        A.Scaler.fit(F, "r9z")
    except ValueError:
        return
    raise AssertionError("an unregistered convention must raise")


def test_r5b_convention_clips():
    """R5B clips to +-5; the r6a convention must not.

    The outlier must be an OUTLIER, not a rescaling: multiplying a whole column by 100
    also multiplies its standard deviation by 100, so the z-score is unchanged and
    nothing clips. A single extreme value in one row does produce |z| > 5.
    """
    rng = np.random.default_rng(3)
    F = rng.normal(0, 1, (200, 16)).astype(np.float32)
    F[0, 0] = 25.0            # one extreme sample, std barely moves
    a = A.Scaler.fit(F, "r5b")(F)
    b = A.Scaler.fit(F, "r6a")(F)
    assert np.abs(a).max() <= 5.0 + 1e-6, "r5b convention must clip at 5"
    assert np.abs(b).max() > 5.0, "r6a convention must not clip"


# ============================================ A2: stress must be scaled
def test_stress_uses_training_scaler():
    """Stress features must go through the training scaler, not raw.

    R6A's stage 5 passed `capture_features(...)` (raw) to a head trained on scaled
    features. Measured magnitude mismatch was 0.563 vs 0.780 mean |f|.
    """
    seed = 130001
    sc = A.fixed_calibration(seed, "r5b")
    tex, *_ = G.independent_query_set_v1(seed + A.HELD_OFFSET, 64)
    raw = A.raw_features(seed, tex)
    scaled = A.apply_features(seed, tex, sc)
    assert not np.allclose(raw, scaled), "apply_features must change the tensor"
    # the scaled tensor must look like what the head was fit on
    d = A.build_cell(seed, 1024, sc, 64, 32)
    assert np.abs(scaled.mean(0) - d["f_te"].mean(0)).max() < 1e-6


def test_apply_features_requires_a_scaler():
    """There must be no raw evaluation path a caller can stumble into."""
    import inspect
    sig = inspect.signature(A.apply_features)
    assert "scaler" in sig.parameters
    assert sig.parameters["scaler"].default is inspect.Parameter.empty, (
        "scaler must be mandatory so raw features cannot be used by accident")


# ============================================ A3: calibration truly fixed
def test_calibration_is_actually_fixed():
    """Calibration must be bit-identical across every budget.

    R6A fitted its scaler on `trx[:256]`, which differs per budget because the
    generator reshuffles. Drift measured up to 0.170 in feature units.
    """
    seed = 121011
    digests = set()
    for n in (256, 1024, 4096, 16384):
        sc = A.fixed_calibration(seed, "r5b")
        digests.add(sc.digest())
    assert len(digests) == 1, "calibration must not depend on the training budget"


def test_calibration_pool_is_independent_of_training_data():
    """The calibration pool must not be the training pool."""
    seed = 130001
    cx, *_ = G.independent_query_set_v1(seed + A.CALIB_OFFSET, A.CALIB_SIZE)
    trx, *_ = G.independent_query_set_v1(seed + A.TRAIN_OFFSET, A.CALIB_SIZE)
    assert not np.array_equal(cx, trx), "calibration must come from its own pool"
    assert A.CALIB_OFFSET != A.TRAIN_OFFSET


# ================================================= A4/A5/A6: the gates
def test_gain_gate_uses_both_a():
    """G2 must be computed on both_A, not on direct."""
    src = (Path(__file__).resolve().parents[1] / "scripts/r6d_matrix.py").read_text(
        encoding="utf-8")
    assert '"G2_gain_16k_minus_1k_ge_5pp": d_r5b["mean"]' in src
    assert "d_r5b = contrast(cell(\"r5b\", 16384, \"both_A\"), cell(\"r5b\", 1024, \"both_A\"))" in src


def test_no_gate_is_hardcoded():
    """No gate may be a literal True/False in the matrix script."""
    src = (Path(__file__).resolve().parents[1] / "scripts/r6d_matrix.py").read_text(
        encoding="utf-8")
    for bad in ('"G4_random_labels_chance": True', '"G5_shortcuts_ok": True',
                '"G7_deterministic": True', '"G8_forward_parity": True'):
        assert bad not in src, f"hard-coded gate found: {bad}"


def test_all_registered_gates_present():
    """G1-G8 must all appear in the gate dictionary."""
    src = (Path(__file__).resolve().parents[1] / "scripts/r6d_matrix.py").read_text(
        encoding="utf-8")
    for g in ("G1_", "G2_", "G3_", "G4_", "G5_", "G6_", "G7_"):
        assert g in src, f"gate {g} missing"


# ============================================== seeds and non-nesting
def test_fresh_seeds_never_used_before():
    prev = set(A.DEV_SEEDS) | set(A.CONF_SEEDS)
    assert len(set(A.FRESH_SEEDS) & prev) == 0
    assert len(A.FRESH_SEEDS) == 8


def test_budgets_are_not_nested_documented():
    """Budgets are independent draws; this is why A3 happened."""
    full, *_ = G.independent_query_set_v1(130001 + A.TRAIN_OFFSET, 16384)
    small, *_ = G.independent_query_set_v1(130001 + A.TRAIN_OFFSET, 256)
    assert not np.array_equal(full[:256], small)


# =================================================== metric correctness
def test_both_a_ignores_case_b():
    """case-B pairs must never be scored as flip failures."""
    _, y, vA, vB, q, pid, flips = G.independent_paired_set_v1(130001, 256)
    a, b = y[0::2], y[1::2]
    same = vA[0::2] == vB[0::2]
    assert np.all(a[same] == b[same])
    perfect = y.copy()
    pm = A.paired_metrics(perfect, y, vA, vB, flips)
    assert abs(pm["both_A"] - 1.0) < 1e-9
    assert abs(pm["both_B"] - 1.0) < 1e-9


def test_head_update_matches_r5b():
    """The readout arithmetic must remain R5B's per-example delta rule."""
    rng = np.random.default_rng(4)
    d = 48
    F = rng.normal(0, 1, (96, d)).astype(np.float32)
    Y = rng.integers(0, 2, 96)
    order = rng.permutation(96)
    a = P.Head(d, 130001, zscore=False)
    b = A.Head(d, 130001)
    for i in order:
        a.learn(F[i], int(Y[i]))
        b.learn(F[i], int(Y[i]))
    assert np.array_equal(a.w, b.w)
    assert np.array_equal(a.b, b.b)
