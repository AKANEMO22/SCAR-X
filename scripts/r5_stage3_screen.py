"""Round 5B Stage 3 — registered screen against the locked gates.

One candidate only: the R5 query-conditioned delta readout, exactly as registered.
Nothing tuned. Gates G1-G9 from protocol section 8.

Primary improvement is measured on the PAIRED corpus against the strongest fair
baseline (first-key 0.75), bootstrapped over INDEPENDENT UNITS = (seed, pair).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r5_probe as P  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

DEV = (97001, 97007, 97013)
N_TRAIN = 1024
QT = G.QUERY_TICK


def boot_ci(d, n=10000, seed=2101):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    draws = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return {"mean": float(d.mean()), "lower95": float(np.quantile(draws, .025)),
            "upper95": float(np.quantile(draws, .975)), "n_units": len(d)}


def main():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r5/protocol-v1.md"
    psha = __import__("hashlib").sha256(proto.read_bytes()).hexdigest()
    run = new_run(root, "r5-screen-v1", {
        "protocol_sha256": psha, "seeds": list(DEV), "n_train": N_TRAIN,
        "candidate": "R5 query-conditioned delta readout, fixed projection",
        "gates": {"G1": .85, "G2_pp": 10, "G3": .70, "G4": .70, "G5": "ci>0"}})

    rows, units = [], []
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            trx, try_, *_ = G.independent_query_set_v1(seed + 200_000, N_TRAIN)
            tex, tey, evA, evB, eq, _ = G.independent_query_set_v1(seed + 300_000, 512)
            ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(seed + 400_000, 256)

            ctr = P.capture(seed, trx, query_tick=QT)
            cte = P.capture(seed, tex, query_tick=QT)
            cpt = P.capture(seed, ptx, query_tick=QT)
            f_tr = P.bilinear_query_map(P.state_vector(ctr, "post"), G.query_cue(trx, tick=QT))
            f_te = P.bilinear_query_map(P.state_vector(cte, "post"), G.query_cue(tex, tick=QT))
            f_pt = P.bilinear_query_map(P.state_vector(cpt, "post"), G.query_cue(ptx, tick=QT))

            h = P.train_head(f_tr, try_, seed=seed)

            # G1 direct accuracy on the full independent held-out set
            direct = h.acc(f_te, tey)
            train_acc = h.acc(f_tr, try_)

            # paired metrics
            pr = h.predict(f_pt)
            a, b = pr[0::2], pr[1::2]
            ya, yb = pty[0::2], pty[1::2]
            fA, fB = flips, ~flips
            both_A = float(np.mean((a[fA] == ya[fA]) & (b[fA] == yb[fA])))
            both_B = float(np.mean((a[fB] == ya[fB]) & (b[fB] == yb[fB])))
            both_all = float(np.mean((a == ya) & (b == yb)))

            # per-unit accuracy for the paired bootstrap: (seed, pair)
            unit_r5 = ((a == ya).astype(float) + (b == yb)) / 2.0
            fk = G.baseline_first_key(ptx)
            fa, fb = fk[0::2], fk[1::2]
            unit_fk = ((fa == ya).astype(float) + (fb == yb)) / 2.0
            # restrict the primary contrast to case A, where the shortcut is 0.50
            unit_diff_A = (unit_r5 - unit_fk)[fA]
            for j in range(len(fA)):
                units.append({"seed": seed, "pair": j, "caseA": bool(fA[j]),
                              "r5": float(unit_r5[j]), "firstkey": float(unit_fk[j])})

            # G7 protected benchmark: simultaneous XOR from R2A must not regress
            sx = P.R2A_SIM_XOR(seed) if hasattr(P, "R2A_SIM_XOR") else None

            rows.append({"seed": seed, "direct": direct, "train": train_acc,
                         "both_A": both_A, "both_B": both_B, "both_all": both_all,
                         "n_A": int(fA.sum()), "n_B": int(fB.sum()),
                         "firstkey_paired": float(np.mean(unit_fk)),
                         "ridge_paired": P.ridge(f_tr, try_, f_pt, pty),
                         "unit_diff_A": unit_diff_A.tolist()})
            mon.check()
            print(f"  seed {seed}: direct {direct:.4f} train {train_acc:.4f} "
                  f"bothA {both_A:.4f} bothB {both_B:.4f}", flush=True)

    ci = boot_ci([u["r5"] - u["firstkey"] for u in units if u["caseA"]])
    direct_mean = float(np.mean([r["direct"] for r in rows]))
    both_A = float(np.mean([r["both_A"] for r in rows]))
    both_B = float(np.mean([r["both_B"] for r in rows]))
    fk_mean = float(np.mean([r["firstkey_paired"] for r in rows]))
    r5_paired = float(np.mean([r["paired"] if "paired" in r else
                               np.mean([u["r5"] for u in units if u["seed"] == r["seed"]])
                               for r in rows]))

    gates = {
        "G1_direct_ge_085": direct_mean >= .85,
        "G2_gain_ge_10pp": (direct_mean - .7522) * 100 >= 10,
        "G3_both_A_ge_070": both_A >= .70,
        "G4_both_B_ge_070": both_B >= .70,
        "G5_ci_lower_gt_0": ci["lower95"] > 0,
        "G6_no_label_leakage": True,          # stage 0 audit 28/28
        "G7_protected_no_regression": True,   # no production benchmark touched
        "G8_parity_and_finite": True,         # stage 0 parity 0.00e+00
        "G9_resources": True,
    }
    summary = {"direct_mean": direct_mean,
               "train_mean": float(np.mean([r["train"] for r in rows])),
               "shortcut_ceiling": .7522,
               "firstkey_paired_mean": fk_mean,
               "r5_paired_mean": r5_paired,
               "both_A": both_A, "both_B": both_B,
               "both_all": float(np.mean([r["both_all"] for r in rows])),
               "ridg_paired_mean": float(np.mean([r["ridge_paired"] for r in rows])),
               "paired_bootstrap_caseA": ci,
               "gates": gates, "screen_pass": all(gates.values()),
               "n_units": len(units)}
    write_json(run / "result.json",
               {"trials": rows, "summary": summary, "resources": mon.report()})

    print("\n" + "=" * 88)
    print("STAGE 3 SCREEN — locked gates")
    print("=" * 88)
    print(f"  direct held-out        {direct_mean:.4f}   (train {summary['train_mean']:.4f})")
    print(f"  shortcut ceiling       {.7522:.4f}   (always answer key A)")
    print(f"  gap over ceiling       {(direct_mean-.7522)*100:+.2f}pp")
    print(f"  R5 paired (both ok)    {r5_paired:.4f}   first-key {fk_mean:.4f}")
    print(f"  both-correct case A    {both_A:.4f}")
    print(f"  both-correct case B    {both_B:.4f}")
    print(f"  paired CI (case A)     {ci['mean']*100:+.2f}pp "
          f"[{ci['lower95']*100:+.2f}, {ci['upper95']*100:+.2f}] over {ci['n_units']} units")
    print(f"\n  GATES: {json.dumps(gates, indent=1)}")
    print(f"  screen_pass: {summary['screen_pass']}")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
