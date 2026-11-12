"""SCAR-X Round 9b Retention & Continual Learning Study.

Compares 5 consolidation modes under strict continual learning:
1. none: unconsolidated baseline
2. full: standard BF-Cascade (all synapses)
3. readout_only: cascade on readout synapses only
4. hidden_only: cascade on hidden synapses only
5. selective: activity-dependent cascade

Strict Continual Invariants:
- Calibrated ONLY on Task 1 training sequences (ZERO future task bank).
- No task boundary oracle cues.
- Matched training budget across all arms.
- Full 4x4 skill matrix R_{k,j} recorded per seed.
- Acquisition, Final, Mean Forgetting, and Worst Forgetting (per seed).
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
from scarx.neural.r9b_bf_cascade import R9bBFCascade, CONSOLIDATION_MODES
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_xor_representation,
    generate_distractor_binding,
    generate_order_representation,
)

DEV_SEEDS = [140011, 140017, 140023]
TASK_NAMES = ["strict_binding", "xor", "distractor_binding", "order"]
N_TRAIN = 256
N_TEST = 256


def get_all_task_data(seed: int):
    tasks_data = {}
    tasks_data["strict_binding"] = {
        "train": generate_strict_binding(seed, N_TRAIN, ood=False),
        "test": generate_strict_binding(seed + 5000, N_TEST, ood=False),
    }
    tasks_data["xor"] = {
        "train": generate_xor_representation(seed, N_TRAIN, ood=False),
        "test": generate_xor_representation(seed + 5000, N_TEST, ood=False),
    }
    tasks_data["distractor_binding"] = {
        "train": generate_distractor_binding(seed, N_TRAIN, ood=False),
        "test": generate_distractor_binding(seed + 5000, N_TEST, ood=False),
    }
    tasks_data["order"] = {
        "train": generate_order_representation(seed, N_TRAIN, ood=False),
        "test": generate_order_representation(seed + 5000, N_TEST, ood=False),
    }
    return tasks_data


def run_retention_study():
    print("=" * 80)
    print("SCAR-X Round 9b BF-Cascade Retention & Continual Learning Study")
    print(f"Arms: {CONSOLIDATION_MODES}")
    print(f"Seeds: {DEV_SEEDS}")
    print(f"Sequential Tasks: {TASK_NAMES}")
    print("=" * 80)

    results = {}
    t_start = time.time()

    for arm in CONSOLIDATION_MODES:
        results[arm] = {
            "per_seed": [],
            "mean_acquisition": 0.0,
            "mean_final": 0.0,
            "mean_forgetting": 0.0,
            "worst_forgetting_per_seed": [],
            "mean_worst_forgetting": 0.0,
            "elapsed_seconds": 0.0,
        }

        t_arm = time.time()
        print(f"\nEvaluating Consolidation Arm: [{arm}]")

        for seed in DEV_SEEDS:
            tasks_data = get_all_task_data(seed)

            cfg = Config(
                inputs=8,
                outputs=2,
                hidden=48,
                fan_in=8,
                plastic_rows=12,
                neuron="alif",
                seed=seed,
            )

            model = R9bBFCascade(config=cfg, consolidation_mode=arm)

            # Strict continual invariant: calibrate ONLY on Task 1
            t1_train_x, _ = tasks_data[TASK_NAMES[0]]["train"]
            model.calibrate(t1_train_x[:32])

            # R matrix of shape (4, 4): R[k, j] = accuracy on task j after learning task k
            r_matrix = np.zeros((len(TASK_NAMES), len(TASK_NAMES)), dtype=np.float32)

            for k, train_task in enumerate(TASK_NAMES):
                tr_x, tr_y = tasks_data[train_task]["train"]
                for x_ep, y_ep in zip(tr_x, tr_y):
                    model.learn_episode(x_ep, y_ep)

                # Evaluate all tasks so far
                for j, test_task in enumerate(TASK_NAMES):
                    te_x, te_y = tasks_data[test_task]["test"]
                    preds = model.predict_many(te_x)
                    acc = float(np.mean(preds == te_y))
                    r_matrix[k, j] = acc

            # Compute metrics
            # Acquisition: diagonal elements
            acquisitions = [float(r_matrix[k, k]) for k in range(len(TASK_NAMES))]
            mean_acq = float(np.mean(acquisitions))

            # Final accuracy: last row
            finals = [float(r_matrix[-1, j]) for j in range(len(TASK_NAMES))]
            mean_final = float(np.mean(finals))

            # Forgetting per task j < 3
            forgettings = []
            for j in range(len(TASK_NAMES) - 1):
                peak_acc = float(np.max(r_matrix[j:, j]))
                final_acc = float(r_matrix[-1, j])
                forgettings.append(max(0.0, peak_acc - final_acc))

            mean_forg = float(np.mean(forgettings))
            worst_forg = float(np.max(forgettings))

            seed_record = {
                "seed": seed,
                "r_matrix": r_matrix.tolist(),
                "acquisitions": acquisitions,
                "mean_acquisition": mean_acq,
                "finals": finals,
                "mean_final": mean_final,
                "forgettings": forgettings,
                "mean_forgetting": mean_forg,
                "worst_forgetting": worst_forg,
            }
            results[arm]["per_seed"].append(seed_record)
            results[arm]["worst_forgetting_per_seed"].append(worst_forg)

            print(
                f"  Seed {seed} | Acq: {mean_acq*100:5.2f}% | Final: {mean_final*100:5.2f}% | "
                f"Mean Forg: {mean_forg*100:5.2f}pp | Worst Forg: {worst_forg*100:5.2f}pp"
            )

        # Aggregate across seeds
        results[arm]["mean_acquisition"] = float(np.mean([s["mean_acquisition"] for s in results[arm]["per_seed"]]))
        results[arm]["mean_final"] = float(np.mean([s["mean_final"] for s in results[arm]["per_seed"]]))
        results[arm]["mean_forgetting"] = float(np.mean([s["mean_forgetting"] for s in results[arm]["per_seed"]]))
        results[arm]["mean_worst_forgetting"] = float(np.mean(results[arm]["worst_forgetting_per_seed"]))
        results[arm]["elapsed_seconds"] = float(time.time() - t_arm)

    dt_total = time.time() - t_start
    print(f"\nRetention study completed in {dt_total:.1f} seconds.")

    # Print Summary Table
    print("\n" + "=" * 80)
    print("CONTINUAL LEARNING & BF-CASCADE SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Consolidation Arm':22s} | {'Acquisition':12s} | {'Final Acc':12s} | {'Mean Forg':12s} | {'Worst Forg':12s} | {'Gate Verdict'}")
    print("-" * 85)
    for arm in CONSOLIDATION_MODES:
        acq = results[arm]["mean_acquisition"] * 100
        fin = results[arm]["mean_final"] * 100
        m_f = results[arm]["mean_forgetting"] * 100
        w_f = results[arm]["mean_worst_forgetting"] * 100

        # Check gates: Acq >= 65%, Final >= 65%, Mean Forg <= 3pp, Worst Forg <= 5pp on every seed
        pass_acq = acq >= 65.0
        pass_fin = fin >= 65.0
        pass_m_f = m_f <= 3.0
        all_seed_wf_pass = all(wf <= 0.05 for wf in results[arm]["worst_forgetting_per_seed"])

        if pass_acq and pass_fin and pass_m_f and all_seed_wf_pass:
            verdict = "PASS"
        else:
            reasons = []
            if not pass_acq: reasons.append("Acq<65%")
            if not pass_fin: reasons.append("Final<65%")
            if not pass_m_f: reasons.append("MeanF>3pp")
            if not all_seed_wf_pass: reasons.append("WorstF>5pp")
            verdict = f"FAIL ({', '.join(reasons)})"

        print(f"{arm:22s} | {acq:10.2f}% | {fin:10.2f}% | {m_f:10.2f}pp | {w_f:10.2f}pp | {verdict}")

    # Save artifact
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "r9b_retention_study_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"study": "r9b_retention_study", "results": results}, f, indent=2)
    print(f"\nRetention artifact saved to {out_path}")
    return results


if __name__ == "__main__":
    run_retention_study()
