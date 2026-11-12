"""SCAR-X Round 9b Factorial MTSG DEV Runner.

Executes the full 4x3x2 = 24 factorial MTSG design with strict matched controls
across locked DEV seeds [140011, 140017, 140023].
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Ensure src is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarx.neural.core import Config
from scarx.neural.r9b_factorial_eprop import (
    FactorialMTSGEProp,
    DECAY_MODES,
    GATING_MODES,
    RESET_MODES,
)
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_distractor_binding,
    generate_xor_representation,
    generate_order_representation,
)
from scarx.evaluation.tasks21 import ridge_probe

DEV_SEEDS = [140011, 140017, 140023]
TASKS = ["strict_binding", "xor", "distractor_binding", "order"]
N_TRAIN = 512
N_REFIT = 128
N_TEST = 256


def get_task_data(task: str, seed: int):
    rng_seed = seed
    if task == "strict_binding":
        train_x, train_y = generate_strict_binding(rng_seed, N_TRAIN, ood=False)
        refit_x, refit_y = generate_strict_binding(rng_seed + 1000, N_REFIT, ood=False)
        test_id_x, test_id_y = generate_strict_binding(rng_seed + 2000, N_TEST, ood=False)
        test_ood_x, test_ood_y = generate_strict_binding(rng_seed + 3000, N_TEST, ood=True)
    elif task == "distractor_binding":
        train_x, train_y = generate_distractor_binding(rng_seed, N_TRAIN, ood=False)
        refit_x, refit_y = generate_distractor_binding(rng_seed + 1000, N_REFIT, ood=False)
        test_id_x, test_id_y = generate_distractor_binding(rng_seed + 2000, N_TEST, ood=False)
        test_ood_x, test_ood_y = generate_distractor_binding(rng_seed + 3000, N_TEST, ood=True)
    elif task == "xor":
        train_x, train_y = generate_xor_representation(rng_seed, N_TRAIN, ood=False)
        refit_x, refit_y = generate_xor_representation(rng_seed + 1000, N_REFIT, ood=False)
        test_id_x, test_id_y = generate_xor_representation(rng_seed + 2000, N_TEST, ood=False)
        test_ood_x, test_ood_y = generate_xor_representation(rng_seed + 3000, N_TEST, ood=True)
    elif task == "order":
        train_x, train_y = generate_order_representation(rng_seed, N_TRAIN, ood=False)
        refit_x, refit_y = generate_order_representation(rng_seed + 1000, N_REFIT, ood=False)
        test_id_x, test_id_y = generate_order_representation(rng_seed + 2000, N_TEST, ood=False)
        test_ood_x, test_ood_y = generate_order_representation(rng_seed + 3000, N_TEST, ood=True)
    else:
        raise ValueError(f"Unknown task {task}")
    return train_x, train_y, refit_x, refit_y, test_id_x, test_id_y, test_ood_x, test_ood_y


def evaluate_model(model: FactorialMTSGEProp, refit_x, refit_y, test_id_x, test_id_y, test_ood_x, test_ood_y):
    direct_id = float(np.mean(model.predict_many(test_id_x) == test_id_y))
    direct_ood = float(np.mean(model.predict_many(test_ood_x) == test_ood_y))

    refit_feat = model.features_many(refit_x)
    test_id_feat = model.features_many(test_id_x)
    test_ood_feat = model.features_many(test_ood_x)

    probe_id = ridge_probe(refit_feat, refit_y, test_id_feat, test_id_y)
    probe_ood = ridge_probe(refit_feat, refit_y, test_ood_feat, test_ood_y)

    return direct_id, direct_ood, probe_id, probe_ood


def run_experiment():
    print("=" * 80)
    print("Starting SCAR-X Round 9b Factorial MTSG DEV Study")
    print(f"Factors: 4 Decays x 3 Gatings x 2 Resets = 24 Configs")
    print(f"Seeds: {DEV_SEEDS}")
    print(f"Tasks: {TASKS}")
    print("=" * 80)

    results = []
    t_start = time.time()

    total_runs = len(DECAY_MODES) * len(GATING_MODES) * len(RESET_MODES) * len(DEV_SEEDS) * len(TASKS)
    run_idx = 0

    for decay in DECAY_MODES:
        for gating in GATING_MODES:
            for reset in RESET_MODES:
                config_id = f"{decay}_{gating}_{reset}"
                print(f"\n[Config: {config_id}]")

                for seed in DEV_SEEDS:
                    for task in TASKS:
                        run_idx += 1
                        train_x, train_y, refit_x, refit_y, test_id_x, test_id_y, test_ood_x, test_ood_y = get_task_data(task, seed)

                        cfg = Config(
                            inputs=8,
                            outputs=2,
                            hidden=48,
                            fan_in=8,
                            plastic_rows=12,
                            neuron="alif",
                            seed=seed,
                        )

                        # 1. Candidate Learned
                        model = FactorialMTSGEProp(
                            config=cfg,
                            decay_mode=decay,
                            gating_mode=gating,
                            reset_mode=reset,
                            control_mode="learned",
                        )
                        model.calibrate(train_x[:32])

                        # Frozen probe baseline before training
                        frozen_clone = model.clone()
                        _, _, frozen_probe_id, _ = evaluate_model(frozen_clone, refit_x, refit_y, test_id_x, test_id_y, test_ood_x, test_ood_y)
                        frozen_direct_id = float(np.mean(frozen_clone.predict_many(test_id_x) == test_id_y))

                        # Train candidate
                        elig_norms, sig_norms, step_norms = [], [], []
                        for x_ep, y_ep in zip(train_x, train_y):
                            model.learn_episode(x_ep, y_ep)
                            elig_norms.append(model.last_step_eligibility_norm)
                            sig_norms.append(model.last_step_signal_norm)
                            step_norms.append(model.last_step_update_norm)

                        direct_id, direct_ood, probe_id, probe_ood = evaluate_model(
                            model, refit_x, refit_y, test_id_x, test_id_y, test_ood_x, test_ood_y
                        )

                        # Causal Revert Test
                        revert_model = model.clone()
                        revert_model.revert_hidden()
                        revert_direct_id = float(np.mean(revert_model.predict_many(test_id_x) == test_id_y))
                        causal_drop = direct_id - revert_direct_id

                        # 2. Matched Shuffled Control (Identical decay, gating, reset, traces; ONLY signal permuted)
                        shuf_model = FactorialMTSGEProp(
                            config=cfg,
                            decay_mode=decay,
                            gating_mode=gating,
                            reset_mode=reset,
                            control_mode="shuffled_signal",
                        )
                        shuf_model.calibrate(train_x[:32])
                        for x_ep, y_ep in zip(train_x, train_y):
                            shuf_model.learn_episode(x_ep, y_ep)
                        shuf_direct_id, _, shuf_probe_id, _ = evaluate_model(
                            shuf_model, refit_x, refit_y, test_id_x, test_id_y, test_ood_x, test_ood_y
                        )

                        # 3. Readout Only Control
                        ro_model = FactorialMTSGEProp(
                            config=cfg,
                            decay_mode=decay,
                            gating_mode=gating,
                            reset_mode=reset,
                            control_mode="readout_only",
                        )
                        ro_model.calibrate(train_x[:32])
                        for x_ep, y_ep in zip(train_x, train_y):
                            ro_model.learn_episode(x_ep, y_ep)
                        ro_direct_id = float(np.mean(ro_model.predict_many(test_id_x) == test_id_y))

                        # 4. Zero Signal Control
                        zero_model = FactorialMTSGEProp(
                            config=cfg,
                            decay_mode=decay,
                            gating_mode=gating,
                            reset_mode=reset,
                            control_mode="zero_signal",
                        )
                        zero_model.calibrate(train_x[:32])
                        for x_ep, y_ep in zip(train_x, train_y):
                            zero_model.learn_episode(x_ep, y_ep)
                        zero_direct_id = float(np.mean(zero_model.predict_many(test_id_x) == test_id_y))

                        record = {
                            "config_id": config_id,
                            "decay": decay,
                            "gating": gating,
                            "reset": reset,
                            "seed": seed,
                            "task": task,
                            "direct_id": direct_id,
                            "direct_ood": direct_ood,
                            "probe_id": probe_id,
                            "probe_ood": probe_ood,
                            "frozen_direct_id": frozen_direct_id,
                            "frozen_probe_id": frozen_probe_id,
                            "probe_gain": probe_id - frozen_probe_id,
                            "causal_revert_direct_id": revert_direct_id,
                            "causal_drop": causal_drop,
                            "shuffled_direct_id": shuf_direct_id,
                            "shuffled_probe_id": shuf_probe_id,
                            "readout_only_direct_id": ro_direct_id,
                            "zero_signal_direct_id": zero_direct_id,
                            "mean_eligibility_norm": float(np.mean(elig_norms)),
                            "mean_signal_norm": float(np.mean(sig_norms)),
                            "mean_update_norm": float(np.mean(step_norms)),
                        }
                        results.append(record)

                # Quick progress print for config
                cfg_records = [r for r in results if r["config_id"] == config_id]
                for t in TASKS:
                    t_recs = [r for r in cfg_records if r["task"] == t]
                    mean_dir = np.mean([r["direct_id"] for r in t_recs]) * 100
                    mean_shuf = np.mean([r["shuffled_direct_id"] for r in t_recs]) * 100
                    mean_probe = np.mean([r["probe_id"] for r in t_recs]) * 100
                    print(f"  {t:18s} | Direct: {mean_dir:5.1f}% | Shuf: {mean_shuf:5.1f}% | Probe: {mean_probe:5.1f}%")

    dt_total = time.time() - t_start
    print(f"\nAll {total_runs} runs completed in {dt_total:.1f} seconds.")

    # Save artifact
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "r9b_factorial_dev_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "study": "r9b_factorial_dev",
                    "factors": {"decay": DECAY_MODES, "gating": GATING_MODES, "reset": RESET_MODES},
                    "seeds": DEV_SEEDS,
                    "tasks": TASKS,
                    "total_records": len(results),
                    "wall_time_seconds": dt_total,
                },
                "records": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_path}")
    return results


if __name__ == "__main__":
    run_experiment()
