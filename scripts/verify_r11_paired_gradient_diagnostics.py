"""Paired Gradient Diagnostics on Actual TRAIN Episodes at Trained Checkpoints (P0.C).

Measures:
1. Full Surrogate Oracle direction.
2. Local eligibility direction (pre-optimizer).
3. Actual physical update after optimizer/cast/clip.
Evaluated on actual task TRAIN episodes across trained checkpoints.
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scarx.neural.core import Config
from scarx.neural.r10_independent_replication import IndependentLearningALIF
from scarx.evaluation.r10_bottleneck_analysis import (
    full_surrogate_recurrent_sensitivity,
    compute_gradient_cosine,
)
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_xor_representation,
    generate_distractor_binding,
)

TASKS = {
    "strict_binding": generate_strict_binding,
    "xor": generate_xor_representation,
    "distractor_binding": generate_distractor_binding,
}
SEEDS = (140011, 140017, 140023)
CHECKPOINT_INDICES = (0, 63, 511, 1023)


def run_paired_diagnostics():
    results = {}
    print("=" * 70)
    print("P0.C: PAIRED GRADIENT DIAGNOSTICS ON ACTUAL TRAIN EPISODES")
    print("=" * 70)

    for task_name, gen_func in TASKS.items():
        results[task_name] = {}
        for method in ("own", "stopped", "eprop_matched", "braintrace"):
            results[task_name][method] = []
            cosines_local_to_full = []
            cosines_step_to_full = []
            cosines_step_to_local = []
            norm_ratios_step_to_local = []

            for seed in SEEDS:
                train_x, train_y = gen_func(seed + 200, 1024)
                cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
                model = IndependentLearningALIF(cfg, method=method)
                model.calibrate(train_x[:32])

                # Train to 1024 episodes
                for ep in range(1024):
                    idx = ep % len(train_x)
                    model.learn_episode(train_x[idx], int(train_y[idx]))

                # Measure on fixed TRAIN indices
                for idx in CHECKPOINT_INDICES:
                    x = train_x[idx]
                    target = int(train_y[idx])

                    # Rollout local model to capture state and error signal
                    model.reset_episode()
                    for row in x:
                        model.step(row, track=True)

                    f = model.core.features
                    f_hat, raw, scale = model.standardized_features(f)
                    logits = model.core.w_out @ f_hat + model.core.bias
                    logits -= np.max(logits)
                    p = np.exp(logits)
                    p /= np.sum(p)
                    err = -p.copy()
                    err[target] += 1.0

                    signal = (model.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale
                    local_dir = (signal[:, None] * model.e_f).astype(np.float64).ravel()

                    # Exact Full Surrogate Recurrent Oracle on this actual episode
                    sensitivity = full_surrogate_recurrent_sensitivity(model.core, x)
                    full_dir = signal.astype(np.float64) @ sensitivity

                    # Simulate one actual optimizer update step from this local direction
                    direction = signal[:, None] * model.e_f
                    trace = 0.90 * model.syn_trace + 0.10 * direction
                    energy = 0.99 * model.syn_energy + 0.01 * (direction * direction)
                    rms = np.sqrt(energy) + 1e-4
                    norm_step = trace / rms
                    step = np.clip(0.008 * norm_step, -0.010, 0.010)
                    step_dir = step.astype(np.float64).ravel()

                    m_local_full = compute_gradient_cosine(local_dir, full_dir)
                    m_step_full = compute_gradient_cosine(step_dir, full_dir)
                    m_step_local = compute_gradient_cosine(step_dir, local_dir)

                    cosines_local_to_full.append(m_local_full["cosine"])
                    cosines_step_to_full.append(m_step_full["cosine"])
                    cosines_step_to_local.append(m_step_local["cosine"])
                    norm_ratios_step_to_local.append(m_step_local["norm_ratio"])

            mean_cos_lf = float(np.mean(cosines_local_to_full))
            mean_cos_sf = float(np.mean(cosines_step_to_full))
            mean_cos_sl = float(np.mean(cosines_step_to_local))
            mean_nr_sl = float(np.mean(norm_ratios_step_to_local))

            summary = {
                "mean_cosine_local_to_full": mean_cos_lf,
                "mean_cosine_step_to_full": mean_cos_sf,
                "mean_cosine_step_to_local": mean_cos_sl,
                "mean_norm_ratio_step_to_local": mean_nr_sl,
            }
            results[task_name][method] = summary
            print(
                f"[{task_name:18s} | {method:14s}] "
                f"cos(Local, Full Oracle): {mean_cos_lf:6.3f} | "
                f"cos(Step, Full Oracle): {mean_cos_sf:6.3f} | "
                f"cos(Step, Local): {mean_cos_sl:6.3f}"
            )

    out_file = ROOT / "artifacts/r11_paired_gradient_diagnostics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved paired diagnostics: {out_file}")
    return results


if __name__ == "__main__":
    run_paired_diagnostics()
