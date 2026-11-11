"""Round 6D — the 2x2 causal matrix plus matched updates and stress.

Protocol: experiments/collab/scarx-neural-round1-20261111/deepseek-r6d/protocol-v1.md

Runs under a `phase` argument:
  replay   -> CORRECTED_REPLAY   (R6A dev seeds, already seen)
  confirm  -> FRESH_CONFIRMATION (never-used seeds; the only preregistered evidence)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r6d_audit as A  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

QT = G.QUERY_TICK
HELD, PAIRS = 1024, 256


def boot_ci(d, n=10000, seed=2101):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    draws = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return {"mean": float(d.mean()), "lower95": float(np.quantile(draws, .025)),
            "upper95": float(np.quantile(draws, .975)),
            "per_seed": d.round(6).tolist(), "worst": float(d.min())}


def stress_valid(seed):
    """Protocol section 4: stress must preserve the task before it may be scored."""
    x, y, *_ = G.independent_query_set_v1(seed + A.HELD_OFFSET, 128)
    xa = G.shifted(x, 1, seed=seed)
    xb = G.with_distractors(x, seed + 55)
    ok_shift = np.array_equal(G.oracle_all(xa), y) and G.locate_query_tick(xa[0]) == QT + 1
    ok_dist = np.array_equal(G.oracle_all(xb), y)
    return bool(ok_shift and ok_dist), {"shift": bool(ok_shift), "distractor": bool(ok_dist)}


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r6d/protocol-v1.md"
    psha = hashlib.sha256(proto.read_bytes()).hexdigest()
    phase = sys.argv[1] if len(sys.argv) > 1 else "replay"
    seeds = A.DEV_SEEDS if phase == "replay" else A.FRESH_SEEDS
    label = "CORRECTED_REPLAY" if phase == "replay" else "FRESH_CONFIRMATION"

    run = new_run(root, f"r6d-{phase}-v1", {
        "protocol_sha256": psha, "label": label, "seeds": list(seeds),
        "cells": "2x2 data x preprocessing", "matched_updates": A.MATCHED_UPDATES,
        "calibration": "fixed pinned pool (offset +700000)", "phase": phase})

    rows, stress_ok = [], {}
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in seeds:
            valid, detail = stress_valid(seed)
            stress_ok[seed] = {"valid": valid, **detail}

            scalers = {c: A.fixed_calibration(seed, c) for c in ("r5b", "r6a")}

            # ---- 2x2: data quantity x preprocessing
            for conv in ("r5b", "r6a"):
                for n in A.BUDGETS:
                    d = A.build_cell(seed, n, scalers[conv], HELD, PAIRS)
                    rng = np.random.default_rng(seed + 31 * n)
                    h = A.train_head(d["f_tr"], d["y_tr"], seed, rng.permutation(n), 1)
                    rep = A.evaluate(h, d)
                    rep.update({"seed": seed, "conv": conv, "n_unique": n,
                                "arm": "grid", "updates": n,
                                "scaler": d["digests"]["scaler"],
                                "digests": d["digests"]})
                    rows.append(rep)
                    mon.check()

            # ---- matched updates (both scalers)
            for conv in ("r5b", "r6a"):
                for n in (256, 16384):
                    d = A.build_cell(seed, n, scalers[conv], HELD, PAIRS)
                    ep = A.MATCHED_UPDATES // n
                    rng = np.random.default_rng(seed + 77 * n)
                    h = A.train_head(d["f_tr"], d["y_tr"], seed,
                                     rng.permutation(n), ep)
                    rep = A.evaluate(h, d)
                    rep.update({"seed": seed, "conv": conv, "n_unique": n,
                                "arm": "matched", "epochs": ep,
                                "updates": ep * n, "scaler": d["digests"]["scaler"]})
                    rows.append(rep)
                    mon.check()

            # ---- stress WITH the training scaler (defect A2 fixed)
            d = A.build_cell(seed, 16384, scalers["r5b"], HELD, PAIRS)
            h = A.train_head(d["f_tr"], d["y_tr"], seed,
                             np.random.default_rng(seed + 9).permutation(16384), 1)
            base = A.evaluate(h, d)
            for kind in ("temporal", "distractor"):
                if kind == "temporal":
                    xp, qt = G.shifted(d["tex"], 1, seed=seed), QT + 1
                else:
                    xp, qt = G.with_distractors(d["tex"], seed + 55), QT
                f_s = A.apply_features(seed, xp, scalers["r5b"], qt)
                f_sp = A.apply_features(seed, G.shifted(d["ptx"], 1, seed=seed)
                                        if kind == "temporal" else
                                        G.with_distractors(d["ptx"], seed + 57),
                                        scalers["r5b"], qt)
                pred = h.predict(f_s)
                pm = A.paired_metrics(h.predict(f_sp), d["pty"], d["pvA"], d["pvB"],
                                      d["pflips"])
                rows.append({"seed": seed, "arm": f"stress-{kind}", "conv": "r5b",
                             "n_unique": 16384, "direct": float(np.mean(pred == d["y_te"])),
                             "both_A": pm["both_A"],
                             "drop_direct": base["direct"] - float(np.mean(pred == d["y_te"])),
                             "drop_bothA": base["both_A"] - pm["both_A"],
                             "stress_valid": valid})
                mon.check()

            print(f"  seed{seed} done (stress valid: {valid})", flush=True)

    def cell(conv, n, k, arm="grid"):
        return [r[k] for r in rows if r["arm"] == arm and r["conv"] == conv
                and r["n_unique"] == n]

    def contrast(a, b):
        return boot_ci(np.array(a) - np.array(b))

    # ---- the 2x2 effects, all on both_A
    d_r5b = contrast(cell("r5b", 16384, "both_A"), cell("r5b", 1024, "both_A"))
    d_r6a = contrast(cell("r6a", 16384, "both_A"), cell("r6a", 1024, "both_A"))
    p_1024 = contrast(cell("r6a", 1024, "both_A"), cell("r5b", 1024, "both_A"))
    p_16k = contrast(cell("r6a", 16384, "both_A"), cell("r5b", 16384, "both_A"))
    interaction = boot_ci((np.array(d_r6a["per_seed"]) - np.array(d_r5b["per_seed"])))
    matched_ur = contrast(cell("r5b", 16384, "both_A", "matched"),
                          cell("r5b", 256, "both_A", "matched"))
    matched_ur_r6a = contrast(cell("r6a", 16384, "both_A", "matched"),
                              cell("r6a", 256, "both_A", "matched"))

    stress = {k: {"both_A": float(np.mean([r["both_A"] for r in rows
                                           if r["arm"] == f"stress-{k}"])),
                  "direct": float(np.mean([r["direct"] for r in rows
                                           if r["arm"] == f"stress-{k}"])),
                  "worst_both_A": float(np.min([r["both_A"] for r in rows
                                                if r["arm"] == f"stress-{k}"]))}
              for k in ("temporal", "distractor")}

    all_valid = all(v["valid"] for v in stress_ok.values())
    gates = {
        "G1_bothA_16k_ge_075": float(np.mean(cell("r5b", 16384, "both_A"))) >= .75,
        "G2_gain_16k_minus_1k_ge_5pp": d_r5b["mean"] * 100 >= 5 and d_r5b["lower95"] > 0,
        "G3_matched_unique_gt_repeated": matched_ur["mean"] > 0,
        "G4_random_labels_chance": None,     # filled by the controls script
        "G5_shortcuts_ok": None,             # filled by the controls script
        "G6_stress_ge_075": all_valid and all(stress[k]["both_A"] >= .75 for k in stress),
        "G7_deterministic": None,            # filled by the repeat check below
        "G8_forward_parity": None,           # filled by tests
    }
    # G7: repeat one cell and require an identical digest
    d0 = A.build_cell(seeds[0], 16384, A.fixed_calibration(seeds[0], "r5b"), HELD, PAIRS)
    h1 = A.train_head(d0["f_tr"], d0["y_tr"], seeds[0],
                      np.random.default_rng(seeds[0] + 31 * 16384).permutation(16384), 1)
    h2 = A.train_head(d0["f_tr"], d0["y_tr"], seeds[0],
                      np.random.default_rng(seeds[0] + 31 * 16384).permutation(16384), 1)
    gates["G7_deterministic"] = h1.digest() == h2.digest()

    summary = {
        "label": label, "seeds": list(seeds), "stress_valid": stress_ok,
        "grid": {f"{c}-{n}": {"both_A": float(np.mean(cell(c, n, "both_A"))),
                              "direct": float(np.mean(cell(c, n, "direct"))),
                              "needB": float(np.nanmean(cell(c, n, "acc_needB"))),
                              "clip": float(np.mean(cell(c, n, "clip_fraction")))}
                 for c in ("r5b", "r6a") for n in A.BUDGETS},
        "effects": {"data_under_r5b": d_r5b, "data_under_r6a": d_r6a,
                    "preproc_at_1024": p_1024, "preproc_at_16384": p_16k,
                    "interaction": interaction,
                    "matched_unique_minus_repeated_r5b": matched_ur,
                    "matched_unique_minus_repeated_r6a": matched_ur_r6a},
        "matched": {c: {str(n): float(np.mean(cell(c, n, "both_A", "matched")))
                        for n in (256, 16384)} for c in ("r5b", "r6a")},
        "stress": stress, "gates": gates,
    }
    write_json(run / "result.json",
               {"protocol_sha256": psha, "label": label, "trials": rows,
                "summary": summary, "resources": mon.report(),
                "wall_seconds": time.time() - t0})

    print("\n" + "=" * 90)
    print(f"{label} — 2x2 both_A matrix")
    print("=" * 90)
    print(f"  {'scaler':8s} {'n=1024':>10} {'n=16384':>10} {'gain':>18}")
    for c in ("r5b", "r6a"):
        eff = d_r5b if c == "r5b" else d_r6a
        print(f"  {c:8s} {np.mean(cell(c,1024,'both_A')):10.4f} "
              f"{np.mean(cell(c,16384,'both_A')):10.4f} "
              f"{eff['mean']*100:+7.2f}pp [{eff['lower95']*100:+6.2f},{eff['upper95']*100:+6.2f}]")
    print(f"\n  preprocessing effect at n=1024 : {p_1024['mean']*100:+7.2f}pp "
          f"[{p_1024['lower95']*100:+6.2f},{p_1024['upper95']*100:+6.2f}]")
    print(f"  preprocessing effect at n=16384: {p_16k['mean']*100:+7.2f}pp "
          f"[{p_16k['lower95']*100:+6.2f},{p_16k['upper95']*100:+6.2f}]")
    print(f"  interaction                    : {interaction['mean']*100:+7.2f}pp")
    print(f"\n  matched unique-repeated (r5b)  : {matched_ur['mean']*100:+7.2f}pp "
          f"[{matched_ur['lower95']*100:+6.2f},{matched_ur['upper95']*100:+6.2f}] "
          f"worst {matched_ur['worst']*100:+.2f}pp")
    print(f"  matched unique-repeated (r6a)  : {matched_ur_r6a['mean']*100:+7.2f}pp")
    print(f"\n  stress both_A: temporal {stress['temporal']['both_A']:.4f} "
          f"(worst {stress['temporal']['worst_both_A']:.4f}) | "
          f"distractor {stress['distractor']['both_A']:.4f} "
          f"(worst {stress['distractor']['worst_both_A']:.4f})")
    print(f"  stress transforms valid: {all_valid}")
    print(f"  gates: {json.dumps({k: v for k, v in gates.items()}, default=str)}")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
