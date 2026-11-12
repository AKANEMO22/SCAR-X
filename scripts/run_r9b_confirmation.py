"""SCAR-X Round 9b 10-Seed Confirmation Runner.

Executes the frozen top candidate on 10 FRESH, UNSEEN seeds [150001..150010].
Evaluates strict_binding, xor, distractor_binding, and order.
Records per-seed metrics, 95% confidence intervals, and gate verdicts.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarx.neural.core import Config
from scarx.neural.r9b_factorial_eprop import FactorialMTSGEProp
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_distractor_binding,
    generate_xor_representation,
    generate_order_representation,
)
from scarx.evaluation.tasks21 import ridge_probe

CONFIRMATION_SEEDS = [150001, 150002, 150003, 150004, 150005, 150006, 150007, 150008, 150009, 150010]
TASKS = ["strict_binding", "xor", "distractor_binding", "order"]
N_TRAIN = 512
N_REFIT = 128
N_TEST = 256


def get_task_data(task: str, seed: int):
    if task == "strict_binding":
        train_x, train_y = generate_strict_binding(seed, N_TRAIN, ood=False)
        refit_x, refit_y = generate_strict_binding(seed + 1000, N_REFIT, ood=False)
        test_id_x, test_id_y = generate_strict_binding(seed + 2000, N_TEST, ood=False)
        test_ood_x, test_ood_y = generate_strict_binding(seed + 3000, N_TEST, ood=True)
    elif task == "distractor_binding":
        train_x, train_y = generate_distractor_binding(seed, N_TRAIN, ood=False)
        refit_x, refit_y = generate_distractor_binding(seed + 1000, N_REFIT, ood=False)
        test_id_x, test_id_y = generate_distractor_binding(seed + 2000, N_TEST, ood=False)
        test_ood_x, test_ood_y = generate_distractor_binding(seed + 3000, N_TEST, ood=True)
    elif task == "xor":
        train_x, train_y = generate_xor_representation(seed, N_TRAIN, ood=False)
        refit_x, refit_y = generate_xor_representation(seed + 1000, N_REFIT, ood=False)
        test_id_x, test_id_y = generate_xor_representation(seed + 2000, N_TEST, ood=False)
        test_ood_x, test_ood_y = generate_xor_representation(seed + 3000, N_TEST, ood=True)
    elif task == "order":
        train_x, train_y = generate_order_representation(seed, N_TRAIN, ood=False)
        refit_x, refit_y = generate_order_representation(seed + 1000, N_REFIT, ood=False)
        test_id_x, test_id_y = generate_order_representation(seed + 2000, N_TEST, ood=False)
        test_ood_x, test_ood_y = generate_order_representation(seed + 3000, N_TEST, ood=True)
    else:
        raise ValueError(f"Unknown task {task}")
    return train_x, train_y, refit_x, refit_y, test_id_x, test_id_y, test_ood_x, test_ood_y


def run_confirmation():
    print("=" * 80)
    print("SCAR-X Round 9b 10-Seed Confirmation Study")
    print(f"Locked Top Candidate: uniform_fast_none_stopped_reset")
    print(f"Fresh Seeds: {CONFIRMATION_SEEDS}")
    print(f"Tasks: {TASKS}")
    print("=" * 80)

    t_start = time.time()
    records = []

    for seed in CONFIRMATION_SEEDS:
        print(f"\nEvaluating Seed: {seed}")
        for task in TASKS:
            train_x, train_y, refit_x, refit_y, test_id_x, test_id_y, test_ood_x, test_ood_y = get_task_data(task, seed)

            cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, neuron="alif", seed=seed)

            # Candidate: uniform_fast_none_stopped_reset
            model = FactorialMTSGEProp(
                config=cfg,
                decay_mode="uniform_fast",
                gating_mode="none",
                reset_mode="stopped_reset",
                control_mode="learned",
            )
            model.calibrate(train_x[:32])

            # Frozen baseline probe
            f_clone = model.clone()
            fr_feat_refit = f_clone.features_many(refit_x)
            fr_feat_test = f_clone.features_many(test_id_x)
            frozen_probe = ridge_probe(fr_feat_refit, refit_y, fr_feat_test, test_id_y)
            frozen_direct = float(np.mean(f_clone.predict_many(test_id_x) == test_id_y))

            # Train
            for x_ep, y_ep in zip(train_x, train_y):
                model.learn_episode(x_ep, y_ep)

            direct_id = float(np.mean(model.predict_many(test_id_x) == test_id_y))
            direct_ood = float(np.mean(model.predict_many(test_ood_x) == test_ood_y))

            refit_feat = model.features_many(refit_x)
            test_id_feat = model.features_many(test_id_x)
            probe_id = ridge_probe(refit_feat, refit_y, test_id_feat, test_id_y)

            # Causal Revert
            rev_model = model.clone()
            rev_model.revert_hidden()
            rev_direct = float(np.mean(rev_model.predict_many(test_id_x) == test_id_y))
            causal_drop = direct_id - rev_direct

            # Matched Shuffled Control
            shuf_m = FactorialMTSGEProp(
                config=cfg,
                decay_mode="uniform_fast",
                gating_mode="none",
                reset_mode="stopped_reset",
                control_mode="shuffled_signal",
            )
            shuf_m.calibrate(train_x[:32])
            for x_ep, y_ep in zip(train_x, train_y):
                shuf_m.learn_episode(x_ep, y_ep)
            shuf_direct = float(np.mean(shuf_m.predict_many(test_id_x) == test_id_y))

            rec = {
                "seed": seed,
                "task": task,
                "direct_id": direct_id,
                "direct_ood": direct_ood,
                "probe_id": probe_id,
                "frozen_probe_id": frozen_probe,
                "frozen_direct_id": frozen_direct,
                "probe_gain": probe_id - frozen_probe,
                "causal_revert_direct": rev_direct,
                "causal_drop": causal_drop,
                "shuffled_direct": shuf_direct,
                "vs_shuffled": direct_id - shuf_direct,
            }
            records.append(rec)
            print(
                f"  {task:18s} | Direct: {direct_id*100:5.2f}% | Shuf: {shuf_direct*100:5.2f}% | "
                f"Probe: {probe_id*100:5.2f}% | Causal Drop: {causal_drop*100:+5.2f}pp"
            )

    dt_total = time.time() - t_start
    print(f"\n10-seed confirmation completed in {dt_total:.1f} seconds.")

    # Summary Statistics & Gates
    print("\n" + "=" * 90)
    print("CONFIRMATION SUMMARY STATISTICS (10 FRESH SEEDS)")
    print("=" * 90)
    print(f"{'Task':18s} | {'Direct ID (CI95)':22s} | {'Probe Gain':14s} | {'Causal Drop':14s} | {'vs Shuffled':14s} | {'Gate'}")
    print("-" * 90)

    task_summaries = {}
    overall_pass = True

    for t in TASKS:
        t_recs = [r for r in records if r["task"] == t]
        dirs = [r["direct_id"] * 100 for r in t_recs]
        gains = [r["probe_gain"] * 100 for r in t_recs]
        drops = [r["causal_drop"] * 100 for r in t_recs]
        vs_shufs = [r["vs_shuffled"] * 100 for r in t_recs]

        mean_dir = float(np.mean(dirs))
        std_dir = float(np.std(dirs, ddof=1))
        ci95_dir_lower = mean_dir - 2.262 * (std_dir / np.sqrt(len(dirs)))
        ci95_dir_upper = mean_dir + 2.262 * (std_dir / np.sqrt(len(dirs)))

        mean_gain = float(np.mean(gains))
        mean_drop = float(np.mean(drops))
        mean_shuf_diff = float(np.mean(vs_shufs))

        pass_direct = mean_dir >= 65.0
        pass_drop = mean_drop >= 5.0
        pass_shuf = mean_shuf_diff >= 2.0

        task_pass = pass_direct and pass_drop and pass_shuf
        if not task_pass and t in ("strict_binding", "xor", "distractor_binding"):
            overall_pass = False

        status = "PASS" if task_pass else "FAIL"
        print(
            f"{t:18s} | {mean_dir:5.2f}% [{ci95_dir_lower:5.2f}, {ci95_dir_upper:5.2f}] | "
            f"{mean_gain:+6.2f}pp      | {mean_drop:+6.2f}pp      | {mean_shuf_diff:+6.2f}pp      | {status}"
        )

        task_summaries[t] = {
            "mean_direct": mean_dir,
            "ci95_lower": ci95_dir_lower,
            "ci95_upper": ci95_dir_upper,
            "mean_probe_gain": mean_gain,
            "mean_causal_drop": mean_drop,
            "mean_vs_shuffled": mean_shuf_diff,
            "pass": task_pass,
        }

    final_verdict = "P1_CAUSAL_CONFIRMED" if overall_pass else "CONFIRMATION_FAIL"
    print("\n" + "=" * 90)
    print(f"FINAL CONFIRMATION DECISION: {final_verdict}")
    print("=" * 90)

    # Save artifact
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "r9b_confirmation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "study": "r9b_confirmation",
                "candidate": "uniform_fast_none_stopped_reset",
                "seeds": CONFIRMATION_SEEDS,
                "tasks": TASKS,
                "final_verdict": final_verdict,
                "task_summaries": task_summaries,
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"\nConfirmation artifact saved to {out_path}")
    return final_verdict, task_summaries, records


if __name__ == "__main__":
    run_confirmation()
