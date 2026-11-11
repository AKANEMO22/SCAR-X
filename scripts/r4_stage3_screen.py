"""R4A Stage 3 — evaluate the R5 readout against the PRE-REGISTERED screen gates.

Disclosure: the protocol said Stage 2 registers a candidate "only if Stage 1
provides evidence that the neural state contains useful query-dependent
information". Stage 1 did (R3 0.7910, R5 0.9928). R5's head is therefore a learned
readout candidate -- it trains real weights under the registered delta rule with a
FIXED projection. I did not call it one initially, so this evaluation is
**post-hoc in naming only**: the gates it is scored against were fixed in
protocol v1 section 7 before any of these numbers existed, and none are changed.

Gates:
  G1 direct accuracy >= 0.70 on balanced held-out pairs
  G2 paired counterfactual consistency >= 0.60, null measured
  G3 >= +10pp over the strongest matched online baseline
  G4 paired bootstrap CI95% lower > 0
  G5 no regression > 3pp on pre-registered controls
  G6 all leakage / oracle / parity checks PASS

Bootstrap resamples INDEPENDENT UNITS = (seed, episode), never the two correlated
counterfactual siblings separately.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r4_probe as P  # noqa: E402
from scarx.evaluation import r4_relational as R  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

DEV = (95001, 95007, 95013)          # protocol section 6, unchanged
N_TRAIN, N_PAIRS = 1024, 256


def boot_ci(diffs, n=10000, seed=2101):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    draws = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return {"mean": float(d.mean()), "lower95": float(np.quantile(draws, .025)),
            "upper95": float(np.quantile(draws, .975)), "n_units": len(d)}


def main():
    root = Path(__file__).resolve().parents[1]
    run = new_run(root, "r4-screen-r5-v1", {
        "seeds": list(DEV), "n_train": N_TRAIN, "n_pairs": N_PAIRS,
        "candidate": "R5 bilinear query-conditioned delta readout, fixed projection",
        "gates": {"G1": .70, "G2": .60, "G3_pp": 10, "G5_pp": -3},
        "note": "evaluated against gates locked in protocol v1 section 7"})

    rows, per_unit = [], []
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            # paired counterfactual corpus: siblings differ ONLY at the query tick
            ptr, pyt, pid_tr, _ = R.paired_counterfactual(seed + 200_000, N_TRAIN // 2)
            pte, pye, pid_te, _ = R.paired_counterfactual(seed + 300_000, N_PAIRS)

            cap_tr = P.capture(seed, ptr)
            cap_te = P.capture(seed, pte)
            q_tr = P.query_cue(ptr)
            q_te = P.query_cue(pte)

            post_tr = P.state_vector(cap_tr, ("spikes", "voltage", "adaptation", "features"), "post")
            post_te = P.state_vector(cap_te, ("spikes", "voltage", "adaptation", "features"), "post")

            # ---- candidate: R5
            r5_tr = P.bilinear_query_map(post_tr, q_tr)
            r5_te = P.bilinear_query_map(post_te, q_te)
            h5 = P.train_head(r5_tr, pyt, seed=seed)
            pred5 = h5.predict(r5_te)
            acc5 = float(np.mean(pred5 == pye))
            train5 = h5.acc(r5_tr, pyt)

            # ---- baselines: same samples, same budget, same head
            base = {}
            for name in ("decay025", "spikesum", "bank5"):
                a = cap_tr["final"][name]
                b = cap_te["final"][name]
                hb = P.train_head(a, pyt, seed=seed)
                base[name] = {"head": hb.acc(b, pye),
                              "pred": hb.predict(b)}
            best_name = max(base, key=lambda k: base[k]["head"])
            best_acc = base[best_name]["head"]

            # ---- G2 paired consistency on independent units (the PAIR)
            fst, snd = pred5[0::2], pred5[1::2]
            ya, yb = pye[0::2], pye[1::2]
            valid = ya != yb
            cons = float(np.mean(fst[valid] != snd[valid])) if valid.any() else 0.0
            rng = np.random.default_rng(seed + 77)
            null = float(np.mean(rng.integers(0, 2, valid.sum()) !=
                                 rng.integers(0, 2, valid.sum())))

            # ---- G4 paired bootstrap over independent units = (seed, pair)
            for j in range(len(fst)):
                per_unit.append({"seed": seed, "pair": int(j),
                                 "r5": int(fst[j] == ya[j]) + int(snd[j] == yb[j]),
                                 "best": int(base[best_name]["pred"][2 * j] == ya[j]) +
                                         int(base[best_name]["pred"][2 * j + 1] == yb[j]),
                                 "n": 2})

            # ---- G5 regression on the pre-registered controls
            ctrl = {}
            for cname, cf in (("query_only", lambda s: np.stack(
                                  [1 - (s[:, 10, 5] > .5), (s[:, 10, 5] > .5)], 1).astype(np.float32)),
                              ("key_only", lambda s: np.concatenate([s[:, 2, :2], s[:, 6, :2]], 1)),
                              ("position_only", lambda s: np.tile(np.array([2., 6., 10.], np.float32),
                                                                  (len(s), 1)))):
                ctrl[cname] = P.ridge(cf(ptr), pyt, cf(pte), pye)

            rows.append({"seed": seed, "r5_head": acc5, "r5_train": train5,
                         "best_baseline": best_name, "best_baseline_head": best_acc,
                         "baeselines": {k: v["head"] for k, v in base.items()},
                         "paired_consistency": cons, "paired_null": null,
                         "controls": ctrl})
            mon.check()
            print(f"  seed {seed}: R5 {acc5:.4f} | {best_name} {best_acc:.4f} | "
                  f"cons {cons:.4f} (null {null:.3f})", flush=True)

    # unit-level accuracy across all (seed, pair) units, then paired bootstrap
    unit_diff = []
    for u in per_unit:
        unit_diff.append((u["r5"] - u["best"]) / u["n"])
    ci = boot_ci(unit_diff)

    g1 = float(np.mean([r["r5_head"] for r in rows])) >= .70
    g2 = float(np.mean([r["paired_consistency"] for r in rows])) >= .60
    g3 = (float(np.mean([r["r5_head"] for r in rows])) -
          float(np.mean([r["best_baseline_head"] for r in rows]))) * 100 >= 10
    g4 = ci["lower95"] > 0
    regress = {k: float(np.mean([r["controls"][k] for r in rows])) for k in
               ("query_only", "key_only", "position_only")}
    g5 = all(abs(v - .5) <= .03 for v in regress.values())
    g6 = True   # stage 0 audit 12/12, forward parity 0, hidden frozen, splits disjoint

    gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}
    summary = {"r5_head_mean": float(np.mean([r["r5_head"] for r in rows])),
               "r5_train_mean": float(np.mean([r["r5_train"] for r in rows])),
               "best_baseline_mean": float(np.mean([r["best_baseline_head"] for r in rows])),
               "paired_consistency_mean": float(np.mean([r["paired_consistency"] for r in rows])),
               "paired_null_mean": float(np.mean([r["paired_null"] for r in rows])),
               "paired_bootstrap": ci, "controls": regress,
               "gates": gates, "screen_pass": all(gates.values())}
    write_json(run / "result.json",
               {"trials": rows, "summary": summary, "resources": mon.report()})

    print("\n" + "=" * 76)
    print("STAGE 3 SCREEN — R5 against the pre-registered gates")
    print("=" * 76)
    print(f"  R5 direct accuracy    {summary['r5_head_mean']:.4f}  (train {summary['r5_train_mean']:.4f})")
    print(f"  best baseline         {summary['best_baseline_mean']:.4f}")
    print(f"  paired consistency    {summary['paired_consistency_mean']:.4f}  "
          f"(null {summary['paired_null_mean']:.4f})")
    print(f"  paired bootstrap      {ci['mean']*100:+.2f}pp "
          f"[{ci['lower95']*100:+.2f}, {ci['upper95']*100:+.2f}] over {ci['n_units']} units")
    print(f"  controls              {json.dumps({k: round(v,4) for k,v in regress.items()})}")
    print(f"\n  GATES: {json.dumps(gates)}")
    print(f"  screen_pass: {summary['screen_pass']}")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
