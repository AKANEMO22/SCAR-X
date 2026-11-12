"""SCAR-X Round 12: Continual Memory & Replay-Lite Study for ALIF H48.

Executes:
- Phase 2: Matched-Compute Stream across 5 Arms:
    Arm A: Sequential Baseline (No Replay, 4096 updates)
    Arm B: Equal-Update No-Replay Control (5120 updates)
    Arm C: Reservoir-16 Replay (5120 updates)
    Arm D: Reservoir-64 Replay (5120 updates)
    Arm E: Interleaved Upper Bound (4096 updates)
- Phase 3: True Retention & Forgetting Tracking (Acquisition, Final, Mean/Worst Forgetting).
- Phase 4: Disentanglement of Hidden vs Readout Forgetting (Probes, Drift, Counterfactual Restoration, Gradient Interference).
- Phase 5: Task Order Invariance (Order 1 vs Order 2 on 3 DEV seeds: 140011, 140017, 140023).
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

from scarx.neural.core import Config, SpikingCore
from scarx.neural.r10_independent_replication import IndependentLearningALIF
from scarx.neural.r12_reservoir_replay import ReservoirReplayBuffer
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

DEV_SEEDS = (140011, 140017, 140023)
TASKS_ORDER_1 = ("strict_binding", "xor", "distractor_binding", "order")
TASKS_ORDER_2 = ("order", "distractor_binding", "xor", "strict_binding")

TASK_GENERATORS = {
    "strict_binding": generate_strict_binding,
    "xor": generate_xor_representation,
    "distractor_binding": generate_distractor_binding,
    "order": generate_order_representation,
}


def build_task_datasets(seed: int) -> dict:
    """Build train, refit, ID test, and OOD test sets for all 4 tasks."""
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
    """Evaluate a single model on all 4 tasks."""
    results = {}
    for task_name in ("strict_binding", "xor", "distractor_binding", "order"):
        task_data = datasets[task_name]
        m_eval = evaluate_model_standard(model, task_data)
        results[task_name] = m_eval
    return results


def capture_model_state(model: IndependentLearningALIF) -> dict:
    """Capture snapshot of model weights, normalization, and traces."""
    return {
        "w_in": model.core.w_in.copy(),
        "w_rec": model.core.w_rec.copy(),
        "w_out": model.core.w_out.copy(),
        "bias": model.core.bias.copy(),
        "f_mean": model.f_mean.copy() if model.f_mean is not None else None,
        "f_var": model.f_var.copy() if model.f_var is not None else None,
        "syn_trace": model.syn_trace.copy(),
        "syn_energy": model.syn_energy.copy(),
        "calibrated": model.calibrated,
    }


def restore_readout_subspace(model: IndependentLearningALIF, saved_state: dict):
    """Restore only readout weights and normalization; preserve hidden weights."""
    model.core.w_out[:] = saved_state["w_out"]
    model.core.bias[:] = saved_state["bias"]
    if saved_state["f_mean"] is not None:
        model.f_mean[:] = saved_state["f_mean"]
        model.f_var[:] = saved_state["f_var"]


def restore_hidden_subspace(model: IndependentLearningALIF, saved_state: dict):
    """Restore only hidden weights (w_in, w_rec); preserve readout weights."""
    model.core.w_in[:] = saved_state["w_in"]
    model.core.w_rec[:] = saved_state["w_rec"]


def restore_full_model(model: IndependentLearningALIF, saved_state: dict):
    """Restore entire model state with exact bit fidelity."""
    model.core.w_in[:] = saved_state["w_in"]
    model.core.w_rec[:] = saved_state["w_rec"]
    model.core.w_out[:] = saved_state["w_out"]
    model.core.bias[:] = saved_state["bias"]
    model.syn_trace[:] = saved_state["syn_trace"]
    model.syn_energy[:] = saved_state["syn_energy"]
    if saved_state["f_mean"] is not None:
        model.f_mean[:] = saved_state["f_mean"]
        model.f_var[:] = saved_state["f_var"]
    model.calibrated = saved_state.get("calibrated", True)


def compute_gradient_directions_on_checkpoint(model: IndependentLearningALIF, datasets: dict, k_samples: int = 64) -> dict:
    """Compute task gradient vectors on the exact same frozen checkpoint."""
    saved_state = capture_model_state(model)
    task_gradients = {}

    for task_name in ("strict_binding", "xor", "distractor_binding", "order"):
        train_x, train_y = datasets[task_name]["train"]
        hid_grads = []
        out_grads = []
        full_grads = []

        for idx in range(min(k_samples, len(train_x))):
            # Restore frozen checkpoint before each episode
            restore_full_model(model, saved_state)
            seq = train_x[idx]
            target = int(train_y[idx])

            # Forward pass to get grad_out
            model.reset_episode()
            for row in seq:
                model.step(row, track=True)
            f = model.core.features
            f_hat, raw, scale = model.standardized_features(f)
            logits = model.core.w_out @ f_hat + model.core.bias
            logits -= np.max(logits)
            p = np.exp(logits)
            p /= np.sum(p)
            err = -p.copy()
            err[target] += 1.0
            grad_out = err[:, None] * f_hat[None, :]

            # Learn episode to capture hidden raw direction
            loss, diag = model.learn_episode(seq, target, record_diagnostics=True)
            if diag and "raw_direction" in diag:
                dir_hid = diag["raw_direction"].ravel()
                dir_out = grad_out.ravel()
                dir_full = np.concatenate([dir_hid, dir_out])

                hid_grads.append(dir_hid)
                out_grads.append(dir_out)
                full_grads.append(dir_full)

        task_gradients[task_name] = {
            "hidden": np.mean(hid_grads, axis=0) if hid_grads else np.zeros(48 * 16),
            "readout": np.mean(out_grads, axis=0) if out_grads else np.zeros(2 * 48),
            "full": np.mean(full_grads, axis=0) if full_grads else np.zeros(48 * 16 + 2 * 48),
        }

    # Restore model to clean state
    restore_full_model(model, saved_state)
    return task_gradients


def run_sequential_stream(
    seed: int,
    task_order: tuple,
    arm_type: str,
    buffer_capacity: int = 0,
    datasets: dict = None,
) -> dict:
    """Run sequential learning across the 4-task stream with specified arm policy."""
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
    model = IndependentLearningALIF(cfg, method="own")

    # Joint calibration on initial samples from all 4 tasks
    calib_seqs = np.concatenate([datasets[t]["train"][0][:8] for t in task_order], axis=0)
    model.calibrate(calib_seqs)

    w_hid_0 = np.concatenate((model.core.w_in, model.core.w_rec), axis=1).copy()
    w_out_0 = model.core.w_out.copy()

    # Setup replay buffer if applicable
    replay_buf = None
    if arm_type in ("reservoir_16", "reservoir_64"):
        replay_buf = ReservoirReplayBuffer(capacity=buffer_capacity, seed=seed + buffer_capacity)

    phase_snapshots = []
    phase_checkpoints = {}
    total_incoming_updates = 0
    total_replay_updates = 0
    recent_sample = None

    t0 = time.process_time()

    for phase_idx, current_task in enumerate(task_order):
        train_x, train_y = datasets[current_task]["train"]
        episodes_in_task = 1024

        for ep in range(episodes_in_task):
            idx = ep % len(train_x)
            seq = train_x[idx]
            target = int(train_y[idx])

            # Replay scheduling: 1 replay every 4 incoming examples
            replay_due = (total_incoming_updates + 1) % 4 == 0

            # 1. Sample replay candidate BEFORE inserting new incoming example
            rep_candidate = None
            if replay_due:
                if arm_type in ("reservoir_16", "reservoir_64") and replay_buf is not None:
                    rep_candidate = replay_buf.sample_replay()
                elif arm_type == "equal_update_recent" and recent_sample is not None:
                    rep_candidate = recent_sample

            # 2. Learn incoming example
            model.learn_episode(seq, target)
            total_incoming_updates += 1
            recent_sample = (seq.copy(), target)

            # 3. Insert incoming example into reservoir
            if replay_buf is not None:
                replay_buf.insert(seq, target)

            # 4. Learn replay candidate if due and available
            if replay_due and rep_candidate is not None:
                rep_x, rep_y = rep_candidate
                model.learn_episode(rep_x, rep_y)
                total_replay_updates += 1

        # End of phase: capture snapshot & checkpoint
        current_w_hid = np.concatenate((model.core.w_in, model.core.w_rec), axis=1)
        current_w_out = model.core.w_out
        hid_drift = float(np.linalg.norm(current_w_hid - w_hid_0))
        out_drift = float(np.linalg.norm(current_w_out - w_out_0))

        eval_all = evaluate_on_all_tasks(model, datasets)
        phase_checkpoints[current_task] = capture_model_state(model)

        phase_snapshots.append({
            "phase": phase_idx + 1,
            "trained_task": current_task,
            "eval": eval_all,
            "hidden_weight_drift": hid_drift,
            "readout_weight_drift": out_drift,
            "incoming_updates": total_incoming_updates,
            "replay_updates": total_replay_updates,
        })

    cpu_sec = time.process_time() - t0

    # Summary metrics
    final_eval = phase_snapshots[-1]["eval"]
    task_retention = {}
    for p_idx, t_name in enumerate(task_order):
        acq_direct = phase_snapshots[p_idx]["eval"][t_name]["id_direct"]
        acq_probe = phase_snapshots[p_idx]["eval"][t_name]["id_probe"]
        fin_direct = final_eval[t_name]["id_direct"]
        fin_probe = final_eval[t_name]["id_probe"]
        forgetting = float(acq_direct - fin_direct)

        task_retention[t_name] = {
            "acquisition_direct": float(acq_direct),
            "acquisition_probe": float(acq_probe),
            "final_direct": float(fin_direct),
            "final_probe": float(fin_probe),
            "forgetting": forgetting,
            "readout_gap_final": float(fin_probe - fin_direct),
        }

    # Mean and worst forgetting for tasks trained prior to the last phase
    prior_tasks = task_order[:-1]
    forgettings = [task_retention[t]["forgetting"] for t in prior_tasks]
    mean_forgetting = float(np.mean(forgettings)) if forgettings else 0.0
    worst_forgetting = float(np.max(forgettings)) if forgettings else 0.0
    mean_final_direct = float(np.mean([task_retention[t]["final_direct"] for t in task_order]))

    return {
        "arm_type": arm_type,
        "seed": seed,
        "task_order": list(task_order),
        "total_incoming_updates": total_incoming_updates,
        "total_replay_updates": total_replay_updates,
        "total_updates": total_incoming_updates + total_replay_updates,
        "cpu_seconds": cpu_sec,
        "buffer_capacity": buffer_capacity,
        "buffer_memory_bytes": replay_buf.memory_bytes if replay_buf else 0,
        "phase_snapshots": phase_snapshots,
        "task_retention": task_retention,
        "mean_forgetting": mean_forgetting,
        "worst_forgetting": worst_forgetting,
        "mean_final_direct": mean_final_direct,
        "phase_checkpoints": phase_checkpoints,
        "final_model": model,
    }


def run_counterfactual_restorations(model: IndependentLearningALIF, phase_checkpoints: dict, datasets: dict, task_order: tuple) -> dict:
    """Perform causal counterfactual swaps on final Phase 4 model."""
    results = {}
    final_state = capture_model_state(model)

    for target_task in task_order[:-1]:
        ckpt = phase_checkpoints[target_task]
        task_data = datasets[target_task]

        # 1. Untouched final model
        restore_full_model(model, final_state)
        score_final = evaluate_model_standard(model, task_data)["id_direct"]

        # 2. Restore Readout Only from target phase checkpoint
        restore_full_model(model, final_state)
        restore_readout_subspace(model, ckpt)
        score_restored_readout = evaluate_model_standard(model, task_data)["id_direct"]

        # 3. Restore Hidden Only from target phase checkpoint
        restore_full_model(model, final_state)
        restore_hidden_subspace(model, ckpt)
        score_restored_hidden = evaluate_model_standard(model, task_data)["id_direct"]

        # 4. Restore Full State from target phase checkpoint
        restore_full_model(model, ckpt)
        score_restored_full = evaluate_model_standard(model, task_data)["id_direct"]

        # Restore back to clean final state
        restore_full_model(model, final_state)

        gain_readout = score_restored_readout - score_final
        gain_hidden = score_restored_hidden - score_final

        results[target_task] = {
            "score_final": float(score_final),
            "score_restored_readout": float(score_restored_readout),
            "score_restored_hidden": float(score_restored_hidden),
            "score_restored_full": float(score_restored_full),
            "gain_readout_restoration": float(gain_readout),
            "gain_hidden_restoration": float(gain_hidden),
        }
    return results


def run_interleaved_upper_bound(seed: int, datasets: dict) -> dict:
    """Run fully interleaved training across 4 tasks as Upper Bound control."""
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
    model = IndependentLearningALIF(cfg, method="own")

    calib_seqs = np.concatenate([datasets[t]["train"][0][:8] for t in TASKS_ORDER_1], axis=0)
    model.calibrate(calib_seqs)

    mixed_examples = []
    for i in range(1024):
        for t_name in TASKS_ORDER_1:
            train_x, train_y = datasets[t_name]["train"]
            idx = i % len(train_x)
            mixed_examples.append((train_x[idx], int(train_y[idx]), t_name))

    mix_rng = np.random.default_rng(seed + 8888)
    perm = mix_rng.permutation(len(mixed_examples))
    shuffled_train = [mixed_examples[p] for p in perm]

    t0 = time.process_time()
    for seq, target, _ in shuffled_train:
        model.learn_episode(seq, target)
    cpu_sec = time.process_time() - t0

    eval_all = evaluate_on_all_tasks(model, datasets)
    task_retention = {}
    for t_name in TASKS_ORDER_1:
        task_retention[t_name] = {
            "acquisition_direct": float(eval_all[t_name]["id_direct"]),
            "acquisition_probe": float(eval_all[t_name]["id_probe"]),
            "final_direct": float(eval_all[t_name]["id_direct"]),
            "final_probe": float(eval_all[t_name]["id_probe"]),
            "forgetting": 0.0,
            "readout_gap_final": float(eval_all[t_name]["id_probe"] - eval_all[t_name]["id_direct"]),
        }

    mean_final_direct = float(np.mean([task_retention[t]["final_direct"] for t in TASKS_ORDER_1]))
    return {
        "arm_type": "interleaved_upper_bound",
        "seed": seed,
        "total_incoming_updates": 4096,
        "total_replay_updates": 0,
        "total_updates": 4096,
        "cpu_seconds": cpu_sec,
        "buffer_capacity": 0,
        "buffer_memory_bytes": 0,
        "eval_all": eval_all,
        "task_retention": task_retention,
        "mean_forgetting": 0.0,
        "worst_forgetting": 0.0,
        "mean_final_direct": mean_final_direct,
    }


def main():
    t_start = time.perf_counter()
    p = psutil.Process(os.getpid())
    print("=" * 75)
    print("SCAR-X ROUND 12: CONTINUAL MEMORY & REPLAY-LITE STUDY (ALIF H48)")
    print("=" * 75)

    all_seed_datasets = {s: build_task_datasets(s) for s in DEV_SEEDS}
    results = {
        "metadata": {
            "round": "R12",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dev_seeds": list(DEV_SEEDS),
            "core_neurons": 48,
        },
        "order_1_arms": {
            "arm_a_sequential_baseline": [],
            "arm_b_equal_update_control": [],
            "arm_c_reservoir_16": [],
            "arm_d_reservoir_64": [],
            "arm_e_interleaved_upper_bound": [],
        },
        "order_2_arms": {
            "arm_a_sequential_baseline": [],
            "arm_c_reservoir_16": [],
            "arm_d_reservoir_64": [],
        },
        "counterfactual_restorations_order_1": {},
        "gradient_interferences": {},
    }

    # =========================================================================
    # 1. RUN ORDER 1 ARMS
    # =========================================================================
    print("\n>>> EXECUTING STREAM ORDER 1: Binding -> XOR -> Distractor -> Order")
    for seed in DEV_SEEDS:
        print(f"\n--- DEV SEED {seed} ---")
        ds = all_seed_datasets[seed]

        # Arm A: Sequential Baseline
        res_a = run_sequential_stream(seed, TASKS_ORDER_1, "sequential_baseline", 0, ds)
        # Gradient interference on frozen Phase 1 checkpoint
        p1_model_copy = IndependentLearningALIF(Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed), method="own")
        restore_full_model(p1_model_copy, res_a["phase_checkpoints"]["strict_binding"])
        grad_dirs = compute_gradient_directions_on_checkpoint(p1_model_copy, ds, k_samples=64)

        pairwise_cos = {}
        for i, t1 in enumerate(TASKS_ORDER_1):
            for j, t2 in enumerate(TASKS_ORDER_1):
                if i < j:
                    c_hid = compute_gradient_cosine(grad_dirs[t1]["hidden"], grad_dirs[t2]["hidden"])["cosine"]
                    c_out = compute_gradient_cosine(grad_dirs[t1]["readout"], grad_dirs[t2]["readout"])["cosine"]
                    c_full = compute_gradient_cosine(grad_dirs[t1]["full"], grad_dirs[t2]["full"])["cosine"]
                    pairwise_cos[f"{t1}_vs_{t2}"] = {"cosine_hidden": c_hid, "cosine_readout": c_out, "cosine_full": c_full}
        results["gradient_interferences"][f"seed_{seed}"] = pairwise_cos

        # Counterfactual restorations for Arm A
        cf_res_a = run_counterfactual_restorations(res_a["final_model"], res_a["phase_checkpoints"], ds, TASKS_ORDER_1)
        results["counterfactual_restorations_order_1"][f"seed_{seed}"] = cf_res_a

        # Clean non-serializable objects before storing
        del res_a["final_model"]
        del res_a["phase_checkpoints"]
        results["order_1_arms"]["arm_a_sequential_baseline"].append(res_a)
        print(f"  [Arm A: Seq Baseline] MeanFinal: {res_a['mean_final_direct']*100:5.2f}% | MeanForgetting: {res_a['mean_forgetting']*100:5.2f}% | WorstForgetting: {res_a['worst_forgetting']*100:5.2f}%")

        # Arm B: Equal-Update No-Replay Control
        res_b = run_sequential_stream(seed, TASKS_ORDER_1, "equal_update_recent", 0, ds)
        del res_b["final_model"]
        del res_b["phase_checkpoints"]
        results["order_1_arms"]["arm_b_equal_update_control"].append(res_b)
        print(f"  [Arm B: Equal-Update] MeanFinal: {res_b['mean_final_direct']*100:5.2f}% | MeanForgetting: {res_b['mean_forgetting']*100:5.2f}% | WorstForgetting: {res_b['worst_forgetting']*100:5.2f}%")

        # Arm C: Reservoir-16 Replay
        res_c = run_sequential_stream(seed, TASKS_ORDER_1, "reservoir_16", 16, ds)
        del res_c["final_model"]
        del res_c["phase_checkpoints"]
        results["order_1_arms"]["arm_c_reservoir_16"].append(res_c)
        print(f"  [Arm C: Reservoir-16] MeanFinal: {res_c['mean_final_direct']*100:5.2f}% | MeanForgetting: {res_c['mean_forgetting']*100:5.2f}% | WorstForgetting: {res_c['worst_forgetting']*100:5.2f}%")

        # Arm D: Reservoir-64 Replay
        res_d = run_sequential_stream(seed, TASKS_ORDER_1, "reservoir_64", 64, ds)
        del res_d["final_model"]
        del res_d["phase_checkpoints"]
        results["order_1_arms"]["arm_d_reservoir_64"].append(res_d)
        print(f"  [Arm D: Reservoir-64] MeanFinal: {res_d['mean_final_direct']*100:5.2f}% | MeanForgetting: {res_d['mean_forgetting']*100:5.2f}% | WorstForgetting: {res_d['worst_forgetting']*100:5.2f}%")

        # Arm E: Interleaved Upper Bound
        res_e = run_interleaved_upper_bound(seed, ds)
        results["order_1_arms"]["arm_e_interleaved_upper_bound"].append(res_e)
        print(f"  [Arm E: Interleaved ] MeanFinal: {res_e['mean_final_direct']*100:5.2f}% | MeanForgetting: {res_e['mean_forgetting']*100:5.2f}% | WorstForgetting: {res_e['worst_forgetting']*100:5.2f}%")

    # =========================================================================
    # 2. RUN ORDER 2 ARMS (Task Order Invariance)
    # =========================================================================
    print("\n>>> EXECUTING STREAM ORDER 2 (Permuted): Order -> Distractor -> XOR -> Binding")
    for seed in DEV_SEEDS:
        print(f"\n--- DEV SEED {seed} (Order 2) ---")
        ds = all_seed_datasets[seed]

        res_a2 = run_sequential_stream(seed, TASKS_ORDER_2, "sequential_baseline", 0, ds)
        del res_a2["final_model"]
        del res_a2["phase_checkpoints"]
        results["order_2_arms"]["arm_a_sequential_baseline"].append(res_a2)
        print(f"  [Arm A: Seq Baseline] MeanFinal: {res_a2['mean_final_direct']*100:5.2f}% | MeanForgetting: {res_a2['mean_forgetting']*100:5.2f}% | WorstForgetting: {res_a2['worst_forgetting']*100:5.2f}%")

        res_c2 = run_sequential_stream(seed, TASKS_ORDER_2, "reservoir_16", 16, ds)
        del res_c2["final_model"]
        del res_c2["phase_checkpoints"]
        results["order_2_arms"]["arm_c_reservoir_16"].append(res_c2)
        print(f"  [Arm C: Reservoir-16] MeanFinal: {res_c2['mean_final_direct']*100:5.2f}% | MeanForgetting: {res_c2['mean_forgetting']*100:5.2f}% | WorstForgetting: {res_c2['worst_forgetting']*100:5.2f}%")

        res_d2 = run_sequential_stream(seed, TASKS_ORDER_2, "reservoir_64", 64, ds)
        del res_d2["final_model"]
        del res_d2["phase_checkpoints"]
        results["order_2_arms"]["arm_d_reservoir_64"].append(res_d2)
        print(f"  [Arm D: Reservoir-64] MeanFinal: {res_d2['mean_final_direct']*100:5.2f}% | MeanForgetting: {res_d2['mean_forgetting']*100:5.2f}% | WorstForgetting: {res_d2['worst_forgetting']*100:5.2f}%")

    total_wall = time.perf_counter() - t_start
    peak_rss = float(p.memory_info().rss) / (1024 * 1024)

    results["metadata"]["total_wall_seconds"] = total_wall
    results["metadata"]["peak_rss_mb"] = peak_rss

    # Serialize to JSON artifact
    def json_sanitize(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: json_sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [json_sanitize(v) for v in obj]
        return obj

    out_path = ROOT / "artifacts" / "r12_continual_memory_results.json"
    with open(out_path, "w") as f:
        json.dump(json_sanitize(results), f, indent=2)

    print("\n" + "=" * 75)
    print(f"STUDY COMPLETE! Total Wall: {total_wall:.2f}s | Peak RSS: {peak_rss:.2f} MB")
    print(f"Artifact successfully saved to {out_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
