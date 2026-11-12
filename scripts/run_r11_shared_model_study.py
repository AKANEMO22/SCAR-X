"""SCAR-X Round 11: Shared-Model Bottleneck & Readout Learning Study (P1, P2, P3, P4).

Executes:
- Arm A: Isolated task baselines (1024 episodes per task).
- Arm B: Joint learning, original budget (1024 episodes total = 256 per task).
- Arm C: Joint learning, equal per-task exposure (4096 episodes total = 1024 per task).
- P3 Readout Comparison: Online SGD Readout vs Online Regularized RLS Readout.
- P4 Sequential Stream Diagnostic: Task stream Binding -> XOR -> Distractor -> Order with continual retention tracking.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

# Restrict threading for reproducible CPU-first benchmarking
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil

from scarx.neural.core import Config
from scarx.neural.r10_independent_replication import IndependentLearningALIF
from scarx.neural.r11_rls_readout import RecursiveLeastSquaresReadoutALIF
from scarx.evaluation.r10_bottleneck_analysis import (
    evaluate_model_standard,
    refit_probe,
    compute_gradient_cosine,
)
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_distractor_binding,
    generate_xor_representation,
    generate_order_representation,
)

TASKS = ("strict_binding", "xor", "distractor_binding", "order")
TASK_GENERATORS = {
    "strict_binding": generate_strict_binding,
    "xor": generate_xor_representation,
    "distractor_binding": generate_distractor_binding,
    "order": generate_order_representation,
}
DEV_SEEDS = (140011, 140017, 140023)


def build_task_datasets(seed: int) -> dict:
    """Build datasets for all 4 tasks for a given seed."""
    out = {}
    for task_name, gen_func in TASK_GENERATORS.items():
        train_x, train_y = gen_func(seed + 200, 1024)
        refit_x, refit_y = gen_func(seed + 300, 256)
        id_x, id_y = gen_func(seed + 400, 512)
        ood_x, ood_y = gen_func(seed + 500, 512, ood=True)
        out[task_name] = {
            "train": (train_x, train_y),
            "refit": (refit_x, refit_y),
            "id": (id_x, id_y),
            "ood": (ood_x, ood_y),
        }
    return out


def evaluate_on_all_tasks(model, datasets: dict) -> dict:
    """Evaluate a single shared model on all 4 tasks."""
    results = {}
    for task_name in TASKS:
        task_data = datasets[task_name]
        m_eval = evaluate_model_standard(model, task_data)
        results[task_name] = m_eval
    return results


def run_arm_a_isolated(readout_type="sgd") -> dict:
    print(f"\n{'='*70}\nARM A: ISOLATED BASELINES (Readout: {readout_type.upper()})\n{'='*70}")
    results = {}
    for task_name in TASKS:
        results[task_name] = []
        for seed in DEV_SEEDS:
            data = build_task_datasets(seed)[task_name]
            train_x, train_y = data["train"]
            cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)

            if readout_type == "sgd":
                model = IndependentLearningALIF(cfg, method="own")
                learn_fn = model.learn_episode
            else:
                model = RecursiveLeastSquaresReadoutALIF(cfg, method="own")
                learn_fn = model.learn_episode_rls

            model.calibrate(train_x[:32])
            t0 = time.process_time()
            for ep in range(1024):
                idx = ep % len(train_x)
                learn_fn(train_x[idx], int(train_y[idx]))
            cpu_sec = time.process_time() - t0

            m_res = evaluate_model_standard(model, data)
            m_res["cpu_seconds"] = cpu_sec
            m_res["seed"] = seed
            results[task_name].append(m_res)
            print(
                f"[{task_name:18s} | Seed {seed}] Direct: {m_res['id_direct']*100:5.2f}% | "
                f"Probe: {m_res['id_probe']*100:5.2f}% | OOD: {m_res['ood_direct']*100:5.2f}%"
            )
    return results


def run_joint_regime(total_episodes: int, readout_type="sgd", arm_name="ARM_B") -> dict:
    ep_per_task = total_episodes // len(TASKS)
    print(f"\n{'='*70}\n{arm_name}: JOINT LEARNING (Total: {total_episodes}, Per-Task: {ep_per_task}, Readout: {readout_type.upper()})\n{'='*70}")
    results = {"per_seed": [], "summary_per_task": {}}

    for seed in DEV_SEEDS:
        all_data = build_task_datasets(seed)
        cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)

        if readout_type == "sgd":
            model = IndependentLearningALIF(cfg, method="own")
            learn_fn = model.learn_episode
        else:
            model = RecursiveLeastSquaresReadoutALIF(cfg, method="own")
            learn_fn = model.learn_episode_rls

        # Calibrate on mixed initial samples across all 4 tasks
        calib_seqs = np.concatenate([all_data[t]["train"][0][:8] for t in TASKS], axis=0)
        model.calibrate(calib_seqs)

        # Build interleaved training sequence
        mixed_examples = []
        for i in range(ep_per_task):
            for t_idx, t_name in enumerate(TASKS):
                train_x, train_y = all_data[t_name]["train"]
                idx = i % len(train_x)
                mixed_examples.append((train_x[idx], int(train_y[idx]), t_name))

        # Locked deterministic pseudo-random shuffle to prevent task blocks
        mix_rng = np.random.default_rng(seed + 8888)
        perm = mix_rng.permutation(len(mixed_examples))
        shuffled_train = [mixed_examples[p] for p in perm]

        t0 = time.process_time()
        for ep_idx, (seq, target, _) in enumerate(shuffled_train):
            learn_fn(seq, target)
        cpu_sec = time.process_time() - t0

        seed_eval = evaluate_on_all_tasks(model, all_data)
        seed_eval["cpu_seconds"] = cpu_sec
        seed_eval["seed"] = seed
        results["per_seed"].append(seed_eval)

        print(f"--- Seed {seed} (CPU: {cpu_sec:.2f}s) ---")
        for t_name in TASKS:
            t_res = seed_eval[t_name]
            print(
                f"  {t_name:18s} | Direct: {t_res['id_direct']*100:5.2f}% | "
                f"Probe: {t_res['id_probe']*100:5.2f}% | OOD: {t_res['ood_direct']*100:5.2f}%"
            )

    # Compute task-level averages
    for t_name in TASKS:
        directs = [s[t_name]["id_direct"] for s in results["per_seed"]]
        probes = [s[t_name]["id_probe"] for s in results["per_seed"]]
        oods = [s[t_name]["ood_direct"] for s in results["per_seed"]]
        results["summary_per_task"][t_name] = {
            "mean_direct": float(np.mean(directs)),
            "mean_probe": float(np.mean(probes)),
            "mean_ood": float(np.mean(oods)),
        }
        print(
            f"==> [{t_name:18s} MEAN] Direct: {np.mean(directs)*100:5.2f}% | "
            f"Probe: {np.mean(probes)*100:5.2f}% | OOD: {np.mean(oods)*100:5.2f}%"
        )
    return results


def run_p4_sequential_stream() -> dict:
    print(f"\n{'='*70}\nP4: SEQUENTIAL STREAM CONTINUAL LEARNING DIAGNOSTIC\nStream: Binding (1024) -> XOR (1024) -> Distractor (1024) -> Order (1024)\n{'='*70}")
    results = {}

    for seed in DEV_SEEDS:
        all_data = build_task_datasets(seed)
        cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
        model = IndependentLearningALIF(cfg, method="own")
        # Calibrate
        calib_seqs = np.concatenate([all_data[t]["train"][0][:8] for t in TASKS], axis=0)
        model.calibrate(calib_seqs)

        initial_w_hid = np.concatenate((model.core.w_in, model.core.w_rec), axis=1).copy()
        initial_w_out = model.core.w_out.copy()

        phase_results = []
        task_gradient_directions = {}

        for phase_idx, current_task in enumerate(TASKS):
            train_x, train_y = all_data[current_task]["train"]

            # Train on current task for 1024 episodes
            task_grads = []
            for ep in range(1024):
                idx = ep % len(train_x)
                loss, diag = model.learn_episode(train_x[idx], int(train_y[idx]), record_diagnostics=True)
                if diag and ep >= 1000:
                    task_grads.append(diag["raw_direction"].ravel())

            # Store representative gradient direction for this task
            if task_grads:
                task_gradient_directions[current_task] = np.mean(task_grads, axis=0)

            # Evaluate performance on ALL 4 tasks after this phase
            eval_snapshot = evaluate_on_all_tasks(model, all_data)

            # Measure weight drift
            current_w_hid = np.concatenate((model.core.w_in, model.core.w_rec), axis=1)
            current_w_out = model.core.w_out
            hid_drift = float(np.linalg.norm(current_w_hid - initial_w_hid))
            out_drift = float(np.linalg.norm(current_w_out - initial_w_out))

            phase_entry = {
                "trained_task": current_task,
                "phase": phase_idx + 1,
                "eval": eval_snapshot,
                "hidden_weight_drift": hid_drift,
                "readout_weight_drift": out_drift,
            }
            phase_results.append(phase_entry)

            print(f"Seed {seed} | Completed Phase {phase_idx+1}: {current_task:18s} (HidDrift: {hid_drift:.3f}, OutDrift: {out_drift:.3f})")
            for t_eval in TASKS:
                t_score = eval_snapshot[t_eval]["id_direct"] * 100
                print(f"   Score on {t_eval:18s}: {t_score:5.2f}%")

        # Measure pairwise task gradient alignment (destructive interference)
        task_interferences = {}
        for i, t1 in enumerate(TASKS):
            for j, t2 in enumerate(TASKS):
                if i < j and t1 in task_gradient_directions and t2 in task_gradient_directions:
                    cos_sim = compute_gradient_cosine(task_gradient_directions[t1], task_gradient_directions[t2])["cosine"]
                    task_interferences[f"{t1}_vs_{t2}"] = cos_sim

        results[f"seed_{seed}"] = {
            "phases": phase_results,
            "gradient_interference_cosines": task_interferences,
        }
        print(f"Seed {seed} Task Gradient Interferences: {task_interferences}")
    return results


def main():
    t_start = time.perf_counter()
    p = psutil.Process(os.getpid())
    print("Starting SCAR-X Round 11 Comprehensive Shared-Model & Readout Study...")

    # Arm A: Isolated baselines (SGD vs RLS)
    arm_a_sgd = run_arm_a_isolated("sgd")
    arm_a_rls = run_arm_a_isolated("rls")

    # Arm B: Joint learning, original budget (1024 episodes total)
    arm_b_sgd = run_joint_regime(1024, "sgd", "ARM_B (Original 1024 Budget)")
    arm_b_rls = run_joint_regime(1024, "rls", "ARM_B (Original 1024 Budget)")

    # Arm C: Joint learning, equal per-task exposure (4096 episodes total = 1024 per task)
    arm_c_sgd = run_joint_regime(4096, "sgd", "ARM_C (Equal-Exposure 4096 Budget)")
    arm_c_rls = run_joint_regime(4096, "rls", "ARM_C (Equal-Exposure 4096 Budget)")

    # P4: Sequential stream continual diagnostic
    p4_stream = run_p4_sequential_stream()

    total_wall = time.perf_counter() - t_start
    peak_rss = float(p.memory_info().rss) / (1024 * 1024)

    full_results = {
        "metadata": {
            "round": "R11",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_wall_seconds": total_wall,
            "peak_rss_mb": peak_rss,
            "core_neurons": 48,
        },
        "arm_a_isolated_sgd": arm_a_sgd,
        "arm_a_isolated_rls": arm_a_rls,
        "arm_b_joint_1024_sgd": arm_b_sgd,
        "arm_b_joint_1024_rls": arm_b_rls,
        "arm_c_joint_4096_sgd": arm_c_sgd,
        "arm_c_joint_4096_rls": arm_c_rls,
        "p4_sequential_stream": p4_stream,
    }

    out_file = ROOT / "artifacts/r11_shared_learning_readout_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE in {total_wall:.2f}s! Peak RSS: {peak_rss:.1f} MB")
    print(f"Artifact saved: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
