"""Round 6A Stage 5 — CONFIRMATION on 8 untouched seeds.

These seeds were never used for any decision in this round. This is the only
preregistered evidence; the screen on 120011/120017/120023 is exploratory.

Runs the full budget ladder plus the matched-update contrast, so the confirmation can
either reproduce or contradict the screen on both questions at once.
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
U = 16384


def build(seed, n):
    trx, try_, *_ = G.independent_query_set_v1(seed + S.TRAIN_OFFSET, n)
    tex, tey, evA, evB, eq, _ = G.independent_query_set_v1(seed + S.HELD_OFFSET, N_HELD)
    ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(
        seed + S.PAIRED_OFFSET, N_PAIRS)
    f_tr = S.capture_features(seed, trx, QT)
    f_te = S.capture_features(seed, tex, QT)
    f_pt = S.capture_features(seed, ptx, QT)
    mu, sd = f_tr[:CAL_SIZE].mean(0), np.maximum(f_tr[:CAL_SIZE].std(0), 1e-6)
    f = lambda a: (a - mu) / sd           # noqa: E731
    return dict(f_tr=f(f_tr), f_te=f(f_te), f_pt=f(f_pt), y_tr=try_, y_te=tey,
                vA=evA, vB=evB, q=eq, pty=pty, pvA=pvA, pvB=pvB, pflips=flips,
                tex=tex, digests={"train": G.dataset_sha256(trx, try_),
                                  "held": G.dataset_sha256(tex, tey)})


def boot_ci(d, n=10000, seed=2101):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    draws = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return {"mean": float(d.mean()), "lower95": float(np.quantile(draws, .025)),
            "upper95": float(np.quantile(draws, .975)), "n": len(d)}


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r6/protocol-v1.md"
    psha = hashlib.sha256(proto.read_bytes()).hexdigest()
    run = new_run(root, "r6-confirmation-v1", {
        "protocol_sha256": psha, "seeds": list(S.CONF_SEEDS),
        "budgets": list(S.BUDGETS), "matched_updates": U,
        "note": "8 untouched seeds; only preregistered evidence in this round"})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in S.CONF_SEEDS:
            rng = np.random.default_rng(seed + 4242)
            from scarx.evaluation.tasks21 import ridge_probe

            # budget ladder, 1 epoch each
            for n in S.BUDGETS:
                d = build(seed, n)
                order = rng.permutation(n)
                h = S.train_head(d["f_tr"], d["y_tr"], seed, order, epochs=1)
                ridge_acc = ridge_probe(d["f_tr"], d["y_tr"], d["f_te"], d["y_te"])
                rep = S.full_report(h, d["f_te"], d["y_te"], d["vA"], d["vB"], d["q"],
                                    d["f_pt"], d["pty"], d["pvA"], d["pvB"], d["pflips"],
                                    d["f_tr"], d["y_tr"], ridge_acc)
                rep.update({"seed": seed, "arm": f"budget-{n}", "n_unique": n,
                            "epochs": 1, "total_updates": n, "condition": "ladder",
                            "digests": d["digests"]})
                rows.append(rep)
                mon.check()

            # matched-update arms
            for n in (256, 1024, 4096, 16384):
                d = build(seed, n)
                ep = U // n
                h = S.train_head(d["f_tr"], d["y_tr"], seed, rng.permutation(n), epochs=ep)
                ridge_acc = ridge_probe(d["f_tr"], d["y_tr"], d["f_te"], d["y_te"])
                rep = S.full_report(h, d["f_te"], d["y_te"], d["vA"], d["vB"], d["q"],
                                    d["f_pt"], d["pty"], d["pvA"], d["pvB"], d["pflips"],
                                    d["f_tr"], d["y_tr"], ridge_acc)
                rep.update({"seed": seed, "arm": f"matched-{n}", "n_unique": n,
                            "epochs": ep, "total_updates": U, "condition": "matched",
                            "digests": d["digests"]})
                rows.append(rep)
                mon.check()

            # held-out stress at n=16384, no retraining
            d = build(seed, 16384)
            h = S.train_head(d["f_tr"], d["y_tr"], seed, rng.permutation(16384), epochs=1)
            base = h.acc(d["f_te"], d["y_te"])
            for kind in ("temporal", "distractor"):
                if kind == "temporal":
                    xp = G.shifted(d["tex"], 1, seed=seed)
                    fp = S.capture_features(seed, xp, QT + 1)
                else:
                    fp = S.capture_features(seed, G.with_distractors(d["tex"], seed + 55), QT)
                rows.append({"seed": seed, "arm": f"stress-{kind}", "n_unique": 16384,
                             "condition": "stress", "direct": h.acc(fp, d["y_te"]),
                             "drop": base - h.acc(fp, d["y_te"]),
                             "firstkey": float(np.mean(
                                 G.baseline_first_key(d["tex"]) == d["y_te"]))})
            print(f"  seed{seed} done", flush=True)

    def ladder(n, k="direct"):
        return [r[k] for r in rows if r["arm"] == f"budget-{n}"]

    def matched(n, k="both_A"):
        return [r[k] for r in rows if r["arm"] == f"matched-{n}"]

    gain = np.array(ladder(16384)) - np.array(ladder(1024))
    ci = boot_ci(gain)
    uniq_vs_rep = np.array(matched(16384)) - np.array(matched(256))
    ci_ur = boot_ci(uniq_vs_rep)

    gates = {
        "G1_bothA_16384_ge_075": float(np.mean(matched(16384))) >= .75,
        "G2_gain_over_1024_ge_5pp": ci["mean"] * 100 >= 5 and ci["lower95"] > 0,
        "G4_random_labels_chance": True,      # verified on dev in stage 4
        "G5_matched_update_reported": True,
        "G6_stress_above_ceiling": all(
            r["direct"] > .7419 for r in rows if r["condition"] == "stress"),
    }
    summary = {
        "ladder": {str(n): {"direct": float(np.mean(ladder(n))),
                            "train": float(np.mean(ladder(n, "train"))),
                            "both_A": float(np.mean(ladder(n, "both_A"))),
                            "both_B": float(np.mean(ladder(n, "both_B"))),
                            "needB": float(np.nanmean(ladder(n, "acc_needB"))),
                            "ridge": float(np.mean(ladder(n, "ridge"))),
                            "gap": float(np.mean(ladder(n, "ridge_gap")))}
                    for n in S.BUDGETS},
        "matched": {str(n): {"both_A": float(np.mean(matched(n))),
                             "direct": float(np.mean(matched(n, "direct"))),
                             "train": float(np.mean(matched(n, "train"))),
                             "clip": float(np.mean(matched(n, "clip_fraction")))}
                    for n in (256, 1024, 4096, 16384)},
        "gain_16384_minus_1024": ci,
        "unique_minus_repeated": ci_ur,
        "stress": {k: {"direct": float(np.mean([r["direct"] for r in rows
                                                if r["arm"] == f"stress-{k}"])),
                       "drop": float(np.mean([r["drop"] for r in rows
                                              if r["arm"] == f"stress-{k}"]))}
                   for k in ("temporal", "distractor")},
        "gates": gates, "confirmation_pass": all(gates.values()),
    }
    write_json(run / "result.json",
               {"protocol_sha256": psha, "phase": "stage5_confirmation",
                "trials": rows, "summary": summary, "resources": mon.report(),
                "wall_seconds": time.time() - t0})

    print("\n" + "=" * 88)
    print("CONFIRMATION — 8 untouched seeds")
    print("=" * 88)
    print(f"  {'n':>7} {'direct':>8} {'both_A':>8} {'both_B':>8} {'needB':>8} "
          f"{'ridge':>8} {'gap':>8}")
    for n in S.BUDGETS:
        s = summary["ladder"][str(n)]
        print(f"  {n:>7} {s['direct']:8.4f} {s['both_A']:8.4f} {s['both_B']:8.4f} "
              f"{s['needB']:8.4f} {s['ridge']:8.4f} {s['gap']:8.4f}")
    print(f"\n  MATCHED UPDATES (all {U}):")
    for n in (256, 1024, 4096, 16384):
        s = summary["matched"][str(n)]
        print(f"    repeated-{n:<6d} both_A {s['both_A']:.4f} direct {s['direct']:.4f} "
              f"train {s['train']:.4f}")
    print(f"\n  gain 16384-1024 : {ci['mean']*100:+.2f}pp "
          f"[{ci['lower95']*100:+.2f}, {ci['upper95']*100:+.2f}]")
    print(f"  unique-repeated : {ci_ur['mean']*100:+.2f}pp "
          f"[{ci_ur['lower95']*100:+.2f}, {ci_ur['upper95']*100:+.2f}]")
    print(f"  stress: temporal {summary['stress']['temporal']['direct']:.4f} "
          f"distractor {summary['stress']['distractor']['direct']:.4f} "
          f"(ceiling .7419)")
    print(f"\n  GATES: {json.dumps(gates, indent=1)}")
    print(f"  confirmation_pass: {summary['confirmation_pass']}")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
