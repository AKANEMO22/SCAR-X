"""Comprehensive Independent Replication & Learning Bottleneck Diagnostic Suite for SCAR-X Round 10.

Executes:
- Phase A1: Exact DEV Reproduction (seeds 140011, 140017, 140023).
- Phase A2: Independent Falsification (seeds 140031, 140037, 140043).
- Phase B: Deep Bottleneck Diagnosis for H1, H2, H3, H4.
- Phase C: Static ACR Cross-Check (alpha = 0, 0.5, 1.0).
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

# Constrain threading for reproducible CPU benchmarking
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil

from scarx.neural.core import Config
from scarx.neural.r10_independent_replication import IndependentLearningALIF
from scarx.evaluation.r10_bottleneck_analysis import (
    evaluate_model_standard,
    full_surrogate_recurrent_sensitivity,
    compute_gradient_cosine,
)
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_distractor_binding,
    generate_xor_representation,
    generate_order_representation,
)

TASKS = ("strict_binding", "xor", "distractor_binding")
DEV_SEEDS_A1 = (140011, 140017, 140023)
EXPLORATORY_SEEDS_A2 = (140031, 140037, 140043)
METHODS_A = ("own", "stopped", "eprop_matched", "eprop_aligned", "braintrace")
ACR_PROFILES_C = ("acr_static_0", "acr_static_50", "acr_static_100")


def generate_task_dataset(task: str, seed: int) -> dict:
    """Generate datasets for train, refit, id, and ood using pinned offsets."""
    generators = {
        "strict_binding": generate_strict_binding,
        "xor": generate_xor_representation,
        "distractor_binding": generate_distractor_binding,
        "order": generate_order_representation,
    }
    gen = generators[task]
    splits = {}
    for split, offset, count, ood in [
        ("train", 200, 1024, False),
        ("refit", 300, 256, False),
        ("id", 400, 512, False),
        ("ood", 500, 512, True),
    ]:
        x, y = gen(seed + offset, count, ood=ood)
        splits[split] = (x, y)
    return splits


def run_training_and_eval(
    method: str,
    task: str,
    seed: int,
    data: dict,
    condition: str = "learned_mtsg",
    episodes: int = 1024,
) -> dict:
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
    model = IndependentLearningALIF(cfg, method=method, condition=condition)
    train_x, train_y = data["train"]
    model.calibrate(train_x[:32])

    t0 = time.perf_counter()
    cpu0 = time.process_time()
    losses = []
    for ep in range(episodes):
        idx = ep % len(train_x)
        loss, _ = model.learn_episode(train_x[idx], int(train_y[idx]))
        losses.append(loss)
    cpu_sec = time.process_time() - cpu0
    wall_sec = time.perf_counter() - t0

    metrics = evaluate_model_standard(model, data)
    metrics["cpu_seconds"] = cpu_sec
    metrics["wall_seconds"] = wall_sec
    metrics["final_train_loss"] = float(np.mean(losses[-32:]))
    return metrics


def run_phase_a1() -> dict:
    print("\n" + "=" * 70)
    print("PHASE A1: EXACT DEV REPRODUCTION (Seeds 140011, 140017, 140023)")
    print("=" * 70)
    results = {}
    for method in METHODS_A:
        results[method] = {}
        for task in TASKS:
            results[method][task] = []
            for seed in DEV_SEEDS_A1:
                data = generate_task_dataset(task, seed)
                # Standard training
                m_res = run_training_and_eval(method, task, seed, data, condition="learned_mtsg")
                # Shuffled signal control for matched contrast
                shuf_res = run_training_and_eval(method, task, seed, data, condition="shuffled_signal")
                m_res["shuffled_direct"] = shuf_res["id_direct"]
                m_res["shuffled_contrast"] = float(m_res["id_direct"] - shuf_res["id_direct"])
                m_res["seed"] = seed
                results[method][task].append(m_res)
                print(
                    f"[{method:14s}] Task: {task:18s} Seed: {seed} | "
                    f"Direct: {m_res['id_direct']*100:5.2f}% | Probe: {m_res['id_probe']*100:5.2f}% | "
                    f"OOD: {m_res['ood_direct']*100:5.2f}% | ShufDiff: {m_res['shuffled_contrast']*100:+5.2f}%"
                )
    return results


def run_phase_a2() -> dict:
    print("\n" + "=" * 70)
    print("PHASE A2: INDEPENDENT FALSIFICATION (Exploratory Seeds 140031, 140037, 140043)")
    print("=" * 70)
    results = {}
    for method in METHODS_A:
        results[method] = {}
        for task in TASKS:
            results[method][task] = []
            for seed in EXPLORATORY_SEEDS_A2:
                data = generate_task_dataset(task, seed)
                m_res = run_training_and_eval(method, task, seed, data, condition="learned_mtsg")
                shuf_res = run_training_and_eval(method, task, seed, data, condition="shuffled_signal")
                m_res["shuffled_direct"] = shuf_res["id_direct"]
                m_res["shuffled_contrast"] = float(m_res["id_direct"] - shuf_res["id_direct"])
                m_res["seed"] = seed
                results[method][task].append(m_res)
                print(
                    f"[{method:14s}] Task: {task:18s} Seed: {seed} | "
                    f"Direct: {m_res['id_direct']*100:5.2f}% | Probe: {m_res['id_probe']*100:5.2f}% | "
                    f"OOD: {m_res['ood_direct']*100:5.2f}% | ShufDiff: {m_res['shuffled_contrast']*100:+5.2f}%"
                )
    return results


def run_phase_b_diagnostics() -> dict:
    print("\n" + "=" * 70)
    print("PHASE B: DEEP BOTTLENECK DIAGNOSIS (H1, H2, H3, H4)")
    print("=" * 70)
    seed = 140011
    task = "strict_binding"
    data = generate_task_dataset(task, seed)
    train_x, train_y = data["train"]

    # Fixed frozen test fixture for measuring gradient alignment
    fix_rng = np.random.default_rng(9999)
    fix_seq = fix_rng.uniform(-0.1, 1.1, (21, 8)).astype(np.float32)
    fix_signal = fix_rng.normal(0, 1.0, 48)

    diag_results = {}
    checkpoints = [0, 256, 512, 1024]

    for method in ("own", "stopped", "eprop_matched", "eprop_aligned", "braintrace"):
        print(f"\n--- Diagnosing Method: {method} ---")
        cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
        model = IndependentLearningALIF(cfg, method=method)
        model.calibrate(train_x[:32])

        diag_history = []
        optimizer_cosines = []
        optimizer_norm_ratios = []
        probe_direct_gaps = []

        step_idx = 0
        for target_ckpt in checkpoints:
            while step_idx < target_ckpt:
                idx = step_idx % len(train_x)
                loss, diag = model.learn_episode(train_x[idx], int(train_y[idx]), record_diagnostics=True)
                # H2: Optimizer distortion measurement
                if diag:
                    raw_g = diag["raw_direction"].ravel()
                    step_g = diag["step"].ravel()
                    n_raw = np.linalg.norm(raw_g)
                    n_step = np.linalg.norm(step_g)
                    if n_raw > 0 and n_step > 0:
                        cos_opt = float(np.dot(raw_g, step_g) / (n_raw * n_step))
                        optimizer_cosines.append(cos_opt)
                        optimizer_norm_ratios.append(float(n_step / n_raw))
                step_idx += 1

            # H4: Measure gradient alignment with Full Surrogate Oracle on fixed fixture
            exact_full = full_surrogate_recurrent_sensitivity(model.core, fix_seq)
            full_ref = fix_signal @ exact_full

            # Local step on fixture
            clone = copy.deepcopy(model)
            clone.reset_episode()
            for row in fix_seq:
                clone.step(row, track=True)
            local_dir = (fix_signal[:, None] * clone.e_f).ravel()

            g_metrics = compute_gradient_cosine(local_dir, full_ref)

            # H3: Readout decoding gap at this checkpoint
            eval_metrics = evaluate_model_standard(model, data)
            gap = float(eval_metrics["id_probe"] - eval_metrics["id_direct"])
            probe_direct_gaps.append(gap)

            ckpt_entry = {
                "step": step_idx,
                "oracle_cosine": g_metrics["cosine"],
                "oracle_sign_agreement": g_metrics["sign_agreement"],
                "oracle_norm_ratio": g_metrics["norm_ratio"],
                "id_direct": eval_metrics["id_direct"],
                "id_probe": eval_metrics["id_probe"],
                "probe_direct_gap": gap,
            }
            diag_history.append(ckpt_entry)
            print(
                f"Step {step_idx:4d} | Oracle Cosine: {g_metrics['cosine']:6.3f} | "
                f"ID Direct: {eval_metrics['id_direct']*100:5.2f}% | ID Probe: {eval_metrics['id_probe']*100:5.2f}% | "
                f"Readout Gap: {gap*100:+5.2f} pp"
            )

        diag_results[method] = {
            "trajectory": diag_history,
            "mean_optimizer_cosine": float(np.mean(optimizer_cosines)) if optimizer_cosines else None,
            "mean_optimizer_norm_ratio": float(np.mean(optimizer_norm_ratios)) if optimizer_norm_ratios else None,
        }
        print(
            f"--> Method {method} Mean Optimizer Cosine (raw g vs step): {diag_results[method]['mean_optimizer_cosine']:.4f} | "
            f"Norm Ratio: {diag_results[method]['mean_optimizer_norm_ratio']:.4f}"
        )
    return diag_results


def run_phase_c_acr() -> dict:
    print("\n" + "=" * 70)
    print("PHASE C: STATIC ACR CROSS-CHECK (alpha = 0.0, 0.5, 1.0)")
    print("=" * 70)
    results = {}
    for profile in ACR_PROFILES_C:
        results[profile] = {}
        for task in TASKS:
            results[profile][task] = []
            for seed in DEV_SEEDS_A1:
                data = generate_task_dataset(task, seed)
                m_res = run_training_and_eval(profile, task, seed, data, condition="learned_mtsg")
                shuf_res = run_training_and_eval(profile, task, seed, data, condition="shuffled_signal")
                m_res["shuffled_direct"] = shuf_res["id_direct"]
                m_res["shuffled_contrast"] = float(m_res["id_direct"] - shuf_res["id_direct"])
                m_res["seed"] = seed
                results[profile][task].append(m_res)
                print(
                    f"[{profile:14s}] Task: {task:18s} Seed: {seed} | "
                    f"Direct: {m_res['id_direct']*100:5.2f}% | Probe: {m_res['id_probe']*100:5.2f}% | "
                    f"OOD: {m_res['ood_direct']*100:5.2f}% | ShufDiff: {m_res['shuffled_contrast']*100:+5.2f}%"
                )
    return results


def main():
    print("Starting SCAR-X Round 10 Full Independent Replication & Bottleneck Analysis...")
    t_start = time.perf_counter()
    p = psutil.Process(os.getpid())

    # Phase A1
    a1_results = run_phase_a1()

    # Phase A2
    a2_results = run_phase_a2()

    # Phase B
    b_results = run_phase_b_diagnostics()

    # Phase C
    c_results = run_phase_c_acr()

    total_wall = time.perf_counter() - t_start
    peak_rss = float(p.memory_info().rss) / (1024 * 1024)

    full_artifact = {
        "metadata": {
            "round": "R10",
            "phase": "Independent Replication & Bottleneck Diagnosis",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_wall_seconds": total_wall,
            "peak_rss_mb": peak_rss,
        },
        "phase_a1_reproduction": a1_results,
        "phase_a2_falsification": a2_results,
        "phase_b_diagnostics": b_results,
        "phase_c_acr": c_results,
    }

    out_path = ROOT / "artifacts/r10_replication_and_diagnosis_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_artifact, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE in {total_wall:.2f}s! Peak RSS: {peak_rss:.1f} MB")
    print(f"Artifact saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
