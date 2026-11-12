"""SCAR-X Round 9b Distractor Root-Cause & Information Loss Attribution Study.

Tracks information preservation across 6 computational stages:
1. Stage 1: Input encoding (x)
2. Stage 2: Subthreshold voltage trajectories (u)
3. Stage 3: Recurrent state & Spikes (z, a)
4. Stage 4: Eligibility traces (e_f)
5. Stage 5: Hidden representation features (f)
6. Stage 6: Direct readout output (y_hat)

Tests 6 perturbation conditions:
- clean (no distractor)
- weak_distractor (amp = 0.3)
- standard_distractor (amp = 1.2 on ch 6)
- strong_distractor (amp = 2.5 on ch 6)
- swapped_channel (distractor pulse on ch 1, 2, 3)
- timing_variation (early tick 3, mid tick 5, late tick 8)
- learned vs frozen hidden vs zero-weighted channel
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
from scarx.evaluation.tasks21 import ridge_probe

DEV_SEEDS = [140011, 140017, 140023]


def generate_custom_distractor_dataset(
    seed: int,
    count: int,
    distractor_amp: float = 1.2,
    distractor_channel: int = 6,
    distractor_ticks: tuple[int, ...] = (3, 7, 8),
    clean: bool = False,
):
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    x = rng.normal(0, 0.10, (count, 12, 8)).astype(np.float32)

    for i in range(count):
        query = int(rng.integers(2))
        a = int(y[i]) ^ query
        values = (a, 1 - a)
        order = rng.permutation(2)
        positions = (int(rng.integers(1, 3)), int(rng.integers(4, 6)))

        for position, key in zip(positions, order):
            x[i, position, key] += 1.5
            x[i, position, 2 + values[key]] += 1.5

        if not clean:
            for dt in distractor_ticks:
                if dt not in positions and dt < 10:
                    x[i, dt, distractor_channel] += float(rng.choice([-distractor_amp, distractor_amp]))

        x[i, 10, 4 + query] += 1.5
        x[i, 11, 7] += 1.0

    order = rng.permutation(count)
    return x[order], y[order].astype(np.int32)


def trace_stage_representations(model: FactorialMTSGEProp, sequences: np.ndarray):
    c = model.config
    n_seq = len(sequences)
    h = c.hidden

    stage1_inputs = []
    stage2_voltages = []
    stage3_spikes = []
    stage4_traces = []
    stage5_features = []

    for seq in sequences:
        model.reset_episode()
        v_seq = []
        z_seq = []
        e_seq = []

        for row in seq:
            model.step(row, track=True)
            v_seq.append(model.core.voltage.copy())
            z_seq.append(model.core.spikes.copy())
            e_seq.append(np.linalg.norm(model.e_f, axis=1))

        stage1_inputs.append(seq.flatten())
        stage2_voltages.append(np.array(v_seq).flatten())
        stage3_spikes.append(np.array(z_seq).flatten())
        stage4_traces.append(np.array(e_seq).flatten())
        stage5_features.append(model.core.features.copy())

    return (
        np.array(stage1_inputs, dtype=np.float32),
        np.array(stage2_voltages, dtype=np.float32),
        np.array(stage3_spikes, dtype=np.float32),
        np.array(stage4_traces, dtype=np.float32),
        np.array(stage5_features, dtype=np.float32),
    )


def run_distractor_study():
    print("=" * 80)
    print("SCAR-X Round 9b Distractor Root-Cause & Information Flow Study")
    print("=" * 80)

    conditions = [
        {"name": "clean", "amp": 0.0, "ch": 6, "ticks": (), "clean": True},
        {"name": "weak_amp_0.3", "amp": 0.3, "ch": 6, "ticks": (3, 7, 8), "clean": False},
        {"name": "standard_amp_1.2", "amp": 1.2, "ch": 6, "ticks": (3, 7, 8), "clean": False},
        {"name": "strong_amp_2.5", "amp": 2.5, "ch": 6, "ticks": (3, 7, 8), "clean": False},
        {"name": "swapped_ch_1", "amp": 1.2, "ch": 1, "ticks": (7, 8), "clean": False},
        {"name": "early_ticks_3_4", "amp": 1.2, "ch": 6, "ticks": (3, 4), "clean": False},
        {"name": "late_ticks_7_8", "amp": 1.2, "ch": 6, "ticks": (7, 8), "clean": False},
    ]

    study_results = {}

    for cond in conditions:
        c_name = cond["name"]
        print(f"\nCondition: [{c_name}]")

        seed_directs = []
        seed_stage_probes = {1: [], 2: [], 3: [], 4: [], 5: []}

        for seed in DEV_SEEDS:
            # Generate train and test under this distractor condition
            tr_x, tr_y = generate_custom_distractor_dataset(
                seed, 512, cond["amp"], cond["ch"], cond["ticks"], cond["clean"]
            )
            te_x, te_y = generate_custom_distractor_dataset(
                seed + 7000, 256, cond["amp"], cond["ch"], cond["ticks"], cond["clean"]
            )
            fit_x, fit_y = generate_custom_distractor_dataset(
                seed + 8000, 128, cond["amp"], cond["ch"], cond["ticks"], cond["clean"]
            )

            cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, neuron="alif", seed=seed)
            model = FactorialMTSGEProp(
                cfg,
                decay_mode="uniform_fast",
                gating_mode="soft_sensory",
                reset_mode="stopped_reset",
                control_mode="learned",
            )
            model.calibrate(tr_x[:32])

            for x_ep, y_ep in zip(tr_x, tr_y):
                model.learn_episode(x_ep, y_ep)

            # Direct accuracy
            direct_acc = float(np.mean(model.predict_many(te_x) == te_y))
            seed_directs.append(direct_acc)

            # Trace representations for probing
            fit_s1, fit_s2, fit_s3, fit_s4, fit_s5 = trace_stage_representations(model, fit_x)
            te_s1, te_s2, te_s3, te_s4, te_s5 = trace_stage_representations(model, te_x)

            p1 = ridge_probe(fit_s1, fit_y, te_s1, te_y)
            p2 = ridge_probe(fit_s2, fit_y, te_s2, te_y)
            p3 = ridge_probe(fit_s3, fit_y, te_s3, te_y)
            p4 = ridge_probe(fit_s4, fit_y, te_s4, te_y)
            p5 = ridge_probe(fit_s5, fit_y, te_s5, te_y)

            seed_stage_probes[1].append(p1)
            seed_stage_probes[2].append(p2)
            seed_stage_probes[3].append(p3)
            seed_stage_probes[4].append(p4)
            seed_stage_probes[5].append(p5)

        mean_dir = float(np.mean(seed_directs))
        s_means = {stg: float(np.mean(seed_stage_probes[stg])) for stg in range(1, 6)}

        study_results[c_name] = {
            "direct_accuracy": mean_dir,
            "stage1_input_probe": s_means[1],
            "stage2_voltage_probe": s_means[2],
            "stage3_spikes_probe": s_means[3],
            "stage4_trace_probe": s_means[4],
            "stage5_feature_probe": s_means[5],
        }

        print(
            f"  Direct Acc: {mean_dir*100:5.2f}% | "
            f"S1(Input): {s_means[1]*100:5.2f}% | "
            f"S2(Voltage): {s_means[2]*100:5.2f}% | "
            f"S3(Spikes): {s_means[3]*100:5.2f}% | "
            f"S4(Trace): {s_means[4]*100:5.2f}% | "
            f"S5(Feature): {s_means[5]*100:5.2f}%"
        )

    # Print Information Flow Table
    print("\n" + "=" * 95)
    print("INFORMATION PRESERVATION ACROSS NEURAL STAGES (% Probing Accuracy)")
    print("=" * 95)
    print(
        f"{'Condition':20s} | {'S1: Input':10s} | {'S2: Voltage':11s} | {'S3: Spikes':11s} | "
        f"{'S4: Trace':10s} | {'S5: Feature':11s} | {'Stage 6: Direct'}"
    )
    print("-" * 95)
    for c_name, data in study_results.items():
        print(
            f"{c_name:20s} | {data['stage1_input_probe']*100:9.2f}% | {data['stage2_voltage_probe']*100:10.2f}% | "
            f"{data['stage3_spikes_probe']*100:10.2f}% | {data['stage4_trace_probe']*100:9.2f}% | "
            f"{data['stage5_feature_probe']*100:10.2f}% | {data['direct_accuracy']*100:14.2f}%"
        )

    # Save artifact
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "r9b_distractor_study_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"study": "r9b_distractor_study", "results": study_results}, f, indent=2)
    print(f"\nDistractor study artifact saved to {out_path}")
    return study_results


if __name__ == "__main__":
    run_distractor_study()
