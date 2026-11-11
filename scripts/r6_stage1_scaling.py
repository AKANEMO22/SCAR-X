"""Round 6A Stage 1 — the scaling curve and the matched-update contrast.

Three causal factors are separated by construction:

  A. unique examples   : n_unique in {256, 1024, 4096, 16384}
  B. update count      : total updates held FIXED at U while n_unique varies
  C. representation    : frozen here; learned-hidden is a separate labelled arm

The decisive arm is B. If 16384 updates over 256 unique examples matches 16384
unique examples at the same update count, the gain is repetition, not new knowledge.

Calibration: features are z-scaled using statistics from a FIXED training-only
calibration set (the first 256 unique training episodes), IDENTICAL for every budget.
This is not strictly causal online normalization and is labelled as such. R5B's
batch calibration is reported separately as a historical compatibility control.
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
N_HELD = 1024
N_PAIRS = 256
CAL_SIZE = 256          # fixed calibration prefix, identical for all budgets


def build(seed, n_unique):
    """Nested prefix: a larger budget CONTAINS the smaller one exactly."""
    trx, try_, tvA, tvB, tq, _ = G.independent_query_set_v1(
        seed + S.TRAIN_OFFSET, max(S.BUDGETS))
    trx, try_ = trx[:n_unique], try_[:n_unique]
    tvA, tvB, tq = tvA[:n_unique], tvB[:n_unique], tq[:n_unique]

    tex, tey, evA, evB, eq, _ = G.independent_query_set_v1(seed + S.HELD_OFFSET, N_HELD)
    ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(
        seed + S.PAIRED_OFFSET, N_PAIRS)

    f_tr = S.capture_features(seed, trx, QT)
    f_te = S.capture_features(seed, tex, QT)
    f_pt = S.capture_features(seed, ptx, QT)

    # fixed calibration prefix, train only, identical across budgets
    cal = f_tr[:CAL_SIZE]
    mu, sd = cal.mean(0), np.maximum(cal.std(0), 1e-6)
    f = lambda a: (a - mu) / sd           # noqa: E731
    return dict(f_tr=f(f_tr), f_te=f(f_te), f_pt=f(f_pt),
                y_tr=try_, y_te=tey, vA=evA, vB=evB, q=eq,
                pty=pty, pvA=pvA, pvB=pvB, pflips=flips,
                digests={"train": G.dataset_sha256(trx, try_),
                         "held": G.dataset_sha256(tex, tey),
                         "paired": G.dataset_sha256(ptx, pty)})


def ridge_ref(d):
    from scarx.evaluation.tasks21 import ridge_probe
    return ridge_probe(d["f_tr"], d["y_tr"], d["f_te"], d["y_te"])


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r6/protocol-v1.md"
    psha = hashlib.sha256(proto.read_bytes()).hexdigest() if proto.exists() else "PROTOCOL_PENDING"
    run = new_run(root, "r6-scaling-v1", {
        "protocol_sha256": psha, "seeds": list(S.DEV_SEEDS),
        "budgets": list(S.BUDGETS), "n_held": N_HELD, "n_pairs": N_PAIRS,
        "calibration": "fixed 256-episode training prefix, identical for all budgets",
        "head": "R5B per-example delta rule, lr .08 clip .025"})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in S.DEV_SEEDS:
            rng = np.random.default_rng(seed + 777)
            # two fixed training orders, generated once per seed
            orders = {}
            for n in S.BUDGETS:
                d = build(seed, n)
                orders[n] = (rng.permutation(n), rng.permutation(n))
                ridge_acc = ridge_ref(d)

                for oi, order in enumerate(orders[n]):
                    h = S.train_head(d["f_tr"], d["y_tr"], seed, order, epochs=1)
                    rep = S.full_report(h, d["f_te"], d["y_te"], d["vA"], d["vB"], d["q"],
                                        d["f_pt"], d["pty"], d["pvA"], d["pvB"],
                                        d["pflips"], d["f_tr"], d["y_tr"], ridge_acc)
                    rep.update({"seed": seed, "n_unique": n, "epochs": 1,
                                "order": oi, "total_updates": n,
                                "condition": "unique", "digests": d["digests"]})
                    rows.append(rep)
                    mon.check()
                print(f"  seed{seed} n={n:5d} direct {rows[-1]['direct']:.4f} "
                      f"bothA {rows[-1]['both_A']:.4f} ridge {ridge_acc:.4f}", flush=True)

    wall = time.time() - t0
    result = {"protocol_sha256": psha, "phase": "stage1_scaling",
              "notes": {
                  "nested": "every budget is a PREFIX of the 16384 set",
                  "calibration": "fixed 256-episode training prefix, not test data",
                  "trajectory": "each budget row is its OWN trained head from the same "
                                "initialisation; they are separate points, not a "
                                "checkpointed single trajectory"},
              "trials": rows, "resources": mon.report(), "wall_seconds": wall}
    write_json(run / "result.json", result)

    print("\n" + "=" * 96)
    print("SCALING CURVE — frozen hidden, 3 dev seeds, 2 orders, mean")
    print("=" * 96)
    print(f"  {'n_unique':>9} {'direct':>8} {'train':>8} {'both_A':>8} {'both_B':>8} "
          f"{'needB':>8} {'ridge':>8} {'gap':>8} {'clip%':>7}")
    for n in S.BUDGETS:
        sub = [r for r in rows if r["n_unique"] == n]
        m = lambda k: float(np.nanmean([r[k] for r in sub]))   # noqa: E731
        print(f"  {n:>9} {m('direct'):8.4f} {m('train'):8.4f} {m('both_A'):8.4f} "
              f"{m('both_B'):8.4f} {m('acc_needB'):8.4f} {m('ridge'):8.4f} "
              f"{m('ridge_gap'):8.4f} {m('clip_fraction')*100:7.2f}")
    print(f"\n  ceiling (first-key) {S.CEILING:.4f}")
    print(f"  wall {wall:.1f}s | peak commit "
          f"{mon.report()['peak_commit_or_vms_bytes']:,} B")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
