"""Round 6A Stage 3 — is there a transition, or just a smooth curve? (H4)

The brief warns: do not fit a change-point model after seeing the curve and present
it as preregistered. So this stage does NOT search for a change point. It asks three
prior questions that separate a genuine transition from ordinary saturating growth:

  Q1. Is the per-step learning curve CONCAVE throughout, or does it have an
      interior inflection (accelerating then decelerating)?
  Q2. Does the apparent jump SURVIVE a change of training order and fresh seeds?
  Q3. Is the curve better explained by a simple saturating law with no interior
      transition? We fit `acc(U) = A - B·exp(-U/tau)` by grid search and report the
      residual. A transition hypothesis is only interesting if the smooth law is a
      measurably worse account.

Q1 is answered with a fine-grained update-count sweep, not with the four coarse
budgets, because a 4-point curve cannot resolve an interior inflection.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r6_scaling as S  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

QT = G.QUERY_TICK
N_HELD, N_PAIRS, CAL_SIZE = 1024, 256, 256
SWEEP = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def build(seed, n):
    trx, try_, *_ = G.independent_query_set_v1(seed + S.TRAIN_OFFSET, n)
    tex, tey, evA, evB, eq, _ = G.independent_query_set_v1(seed + S.HELD_OFFSET, N_HELD)
    ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(
        seed + S.PAIRED_OFFSET, N_PAIRS)
    f_tr = S.capture_features(seed, trx, QT)
    f_te = S.capture_features(seed, tex, QT)
    f_pt = S.capture_features(seed, ptx, QT)
    cal = f_tr[:CAL_SIZE]
    mu, sd = cal.mean(0), np.maximum(cal.std(0), 1e-6)
    f = lambda a: (a - mu) / sd           # noqa: E731
    return dict(f_tr=f(f_tr), f_te=f(f_te), f_pt=f(f_pt), y_tr=try_, y_te=tey,
                vA=evA, vB=evB, q=eq, pty=pty, pvA=pvA, pvB=pvB, pflips=flips)


def fit_saturating(us, accs):
    """acc(U) = A - B exp(-U/tau). Returns (params, rmse, predicted)."""
    best = None
    Us = np.logspace(np.log10(max(1, us.min())), np.log10(us.max() * 2), 120)
    for A in np.linspace(max(accs), 1.0, 40):
        for tau in Us:
            B = (A - accs[0]) * np.exp(us[0] / tau)
            pred = A - B * np.exp(-us / tau)
            rmse = float(np.sqrt(np.mean((pred - accs) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, A, B, tau, pred)
    rmse, A, B, tau, pred = best
    return {"A": float(A), "B": float(B), "tau": float(tau)}, rmse, pred


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r6/protocol-v1.md"
    psha = hashlib.sha256(proto.read_bytes()).hexdigest() if proto.exists() else "PROTOCOL_PENDING"
    run = new_run(root, "r6-transition-v1", {
        "protocol_sha256": psha, "seeds": list(S.DEV_SEEDS), "sweep": list(SWEEP),
        "purpose": "test whether the curve has an interior transition or is smooth growth"})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in S.DEV_SEEDS:
            for oi in (0, 1):                     # two fixed orders per size
                for n in SWEEP:
                    d = build(seed, n)
                    rng = np.random.default_rng(seed + 1000 * oi + n)
                    order = rng.permutation(n)
                    h = S.train_head(d["f_tr"], d["y_tr"], seed, order, epochs=1)
                    from scarx.evaluation.tasks21 import ridge_probe
                    ridge_acc = ridge_probe(d["f_tr"], d["y_tr"], d["f_te"], d["y_te"])
                    rep = S.full_report(h, d["f_te"], d["y_te"], d["vA"], d["vB"], d["q"],
                                        d["f_pt"], d["pty"], d["pvA"], d["pvB"],
                                        d["pflips"], d["f_tr"], d["y_tr"], ridge_acc)
                    rep.update({"seed": seed, "n_unique": n, "order": oi,
                                "total_updates": n})
                    rows.append(rep)
                    mon.check()
            print(f"  seed{seed} sweep done", flush=True)

    # aggregate per budget across seeds x orders
    agg = {}
    for n in SWEEP:
        sub = [r for r in rows if r["n_unique"] == n]
        agg[n] = {k: float(np.nanmean([r[k] for r in sub]))
                  for k in ("direct", "both_A", "train", "ridge", "acc_needB")}
        agg[n]["sd_both_A"] = float(np.nanstd([r["both_A"] for r in sub]))

    us = np.array(SWEEP, float)
    accs = np.array([agg[n]["both_A"] for n in SWEEP])

    # Q1: first and second differences on log-scale steps
    d1 = np.diff(accs)
    d2 = np.diff(d1)
    interior_accel = int(np.sum(d2 > 0.02))        # concave growth has NO positive d2

    # Q3: is a smooth saturating law enough?
    params, rmse, pred = fit_saturating(us, accs)
    max_resid = float(np.max(np.abs(pred - accs)))

    # ---- Is the apparent wiggle distinguishable from seed noise?
    # The aggregated curve has interior accelerations, but they are smaller than the
    # between-seed sd. Test per-(seed,order) curves: under a smooth concave law,
    # positive second differences should be rare (a concave function has d2 <= 0).
    # Under pure noise they occur at chance = 0.5.
    pos = tot = 0
    monotone_viol = 0
    for seed in S.DEV_SEEDS:
        for oi in (0, 1):
            seq = np.array([next(r["both_A"] for r in rows
                                 if r["seed"] == seed and r["n_unique"] == nn
                                 and r["order"] == oi) for nn in SWEEP])
            d2s = np.diff(np.diff(seq))
            pos += int(np.sum(d2s > 0))
            tot += len(d2s)
            if np.any(np.diff(seq) < 0):
                monotone_viol += 1
    frac_pos = pos / tot if tot else float("nan")
    seed_sd = float(np.mean([np.nanstd([r["both_A"] for r in rows
                                        if r["n_unique"] == nn]) for nn in SWEEP]))
    mean_abs_d2 = float(np.mean(np.abs(d2))) if len(d2) else 0.0

    summary = {
        "per_curve_positive_d2_fraction": frac_pos,
        "per_curve_positive_d2": f"{pos}/{tot}",
        "curves_with_a_dip": f"{monotone_viol}/{len(S.DEV_SEEDS)*2}",
        "mean_seed_sd": seed_sd,
        "mean_abs_second_difference": mean_abs_d2,
        "wiggle_is_noise": bool(abs(frac_pos - 0.5) < 0.12
                                or mean_abs_d2 < seed_sd),
        "per_budget": {str(n): agg[n] for n in SWEEP},
        "first_diffs": d1.tolist(),
        "second_diffs": d2.tolist(),
        "interior_accelerations": interior_accel,
        "saturating_fit": params,
        "saturating_rmse": rmse,
        "saturating_max_residual": max_resid,
        # A transition requires ALL of:
        #   - an interior acceleration in the aggregated curve,
        #   - a saturating law that misfits,
        #   - per-curve second differences that are NOT at chance (frac_pos < 0.35),
        #   - and a wiggle larger than the between-seed sd.
        "transition_claimed": bool(interior_accel >= 2 and max_resid > 0.05
                                   and frac_pos < 0.35
                                   and mean_abs_d2 > seed_sd),
    }
    write_json(run / "result.json",
               {"protocol_sha256": psha, "phase": "stage3_transition",
                "trials": rows, "summary": summary,
                "resources": mon.report(), "wall_seconds": time.time() - t0})

    print("\n" + "=" * 92)
    print("FINE SWEEP — both_A vs unique examples (3 seeds x 2 orders)")
    print("=" * 92)
    print(f"  {'n':>7} {'both_A':>8} {'sd':>7} {'direct':>8} {'train':>8} {'ridge':>8}")
    for n in SWEEP:
        a = agg[n]
        print(f"  {n:>7} {a['both_A']:8.4f} {a['sd_both_A']:7.4f} {a['direct']:8.4f} "
              f"{a['train']:8.4f} {a['ridge']:8.4f}")
    print("\n  first differences :", np.round(d1, 4).tolist())
    print("  second differences:", np.round(d2, 4).tolist())
    print(f"  interior accelerations (d2 > +0.02): {interior_accel}")
    print(f"\n  smooth saturating fit acc(U)=A-B exp(-U/tau): A={params['A']:.4f} "
          f"B={params['B']:.4f} tau={params['tau']:.1f}")
    print(f"  rmse {rmse:.4f}  max residual {max_resid:.4f}")
    print(f"  => transition claimed: {summary['transition_claimed']}")
    print(f"\nRESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
