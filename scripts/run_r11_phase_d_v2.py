"""Re-run of Phase D (v2) with exact logistic surrogate spike_slope.

Evaluates Damped Reset E-Prop:
- alpha_reset = 0.0 (exact eprop_matched endpoint)
- alpha_reset = 0.5 (damped reset e-prop candidate)
- alpha_reset = 1.0 (exact own_reset endpoint)
across seeds 140011, 140017, 140023 for 1024 episodes.
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scarx.neural.core import Config
from scarx.neural.r11_repaired_eprop import DampedResetEPropALIF
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_xor_representation,
    generate_distractor_binding,
)
from scarx.evaluation.r10_bottleneck_analysis import evaluate_model_standard

TASKS = {
    "strict_binding": generate_strict_binding,
    "xor": generate_xor_representation,
    "distractor_binding": generate_distractor_binding,
}
SEEDS = (140011, 140017, 140023)
ALPHAS = [0.0, 0.5, 1.0]


def run_phase_d_v2():
    results = {}
    print("=" * 70)
    print("PHASE D v2: REPAIRED DAMPED RESET E-PROP EVALUATION")
    print("=" * 70)

    for task_name, gen_func in TASKS.items():
        results[task_name] = {}
        print(f"\n--- TASK: {task_name} ---")
        for alpha in ALPHAS:
            label = f"alpha_{alpha:.1f}"
            results[task_name][label] = []
            directs = []
            probes = []
            oods = []

            for seed in SEEDS:
                train_x, train_y = gen_func(seed + 200, 1024)
                refit_x, refit_y = gen_func(seed + 300, 256)
                id_x, id_y = gen_func(seed + 400, 512)
                ood_x, ood_y = gen_func(seed + 500, 512, ood=True)
                data = {"train": (train_x, train_y), "refit": (refit_x, refit_y), "id": (id_x, id_y), "ood": (ood_x, ood_y)}

                cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
                model = DampedResetEPropALIF(cfg, alpha_reset=alpha)
                model.calibrate(train_x[:32])

                for ep in range(1024):
                    idx = ep % len(train_x)
                    model.learn_episode(train_x[idx], int(train_y[idx]))

                m_res = evaluate_model_standard(model, data)
                m_res["seed"] = seed
                results[task_name][label].append(m_res)
                directs.append(m_res["id_direct"])
                probes.append(m_res["id_probe"])
                oods.append(m_res["ood_direct"])

            mean_dir = np.mean(directs) * 100
            mean_prb = np.mean(probes) * 100
            mean_ood = np.mean(oods) * 100
            print(f"[{label:10s}] Mean Direct: {mean_dir:5.2f}% | Mean Probe: {mean_prb:5.2f}% | Mean OOD: {mean_ood:5.2f}% | Per Seed: {[round(d*100, 2) for d in directs]}")

    out_file = ROOT / "artifacts/r11_phase_d_v2_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nArtifact saved to: {out_file}")
    return results


if __name__ == "__main__":
    run_phase_d_v2()
