"""SCAR-X Round 9c Empirical Information Localization & Causal Diagnostics Runner.

Executes:
1. Trains a SINGLE reference ALIF H48 network per DEV seed on registered clean training data.
2. Evaluates the EXACT same model on 6 paired conditions:
   - clean
   - weak_distractor (0.4)
   - medium_distractor (1.2)
   - strong_distractor (2.5)
   - swapped_channel (channel 4 vs 6)
   - shifted_time (ticks 2, 5, 9)
3. Multi-layer information tracing across S1..S6 with Linear and Non-Linear probes.
4. Causal counterfactual interventions (Voltage, Spike, Adaptation, Full Soma restoration).
5. Aggregates metrics across 3 independent DEV seeds [140011, 140017, 140023].
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
from scarx.evaluation.r9c_paired_dataset import generate_paired_dataset, PAIRED_CONDITIONS
from scarx.evaluation.r9c_causal_diagnostics import (
    trace_episode,
    run_causal_restoration_experiment,
    evaluate_multi_layer_decodability,
)

DEV_SEEDS = [140011, 140017, 140023]
N_TRAIN = 512
N_TEST = 256


def train_reference_model(seed: int) -> tuple[FactorialMTSGEProp, np.ndarray, np.ndarray]:
    """Trains a single reference ALIF H48 model on clean relational binding data."""
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=seed)
    model = FactorialMTSGEProp(
        config=cfg,
        decay_mode="uniform_fast",
        gating_mode="none",
        reset_mode="stopped_reset",
        control_mode="learned",
        lr_hid=0.008,
        lr_out=0.04,
    )

    # Clean training dataset
    train_batch = generate_paired_dataset(seed=seed, count=N_TRAIN, ood=False)
    train_x = train_batch.sequences["clean"]
    train_y = train_batch.labels

    # Calibrate on baseline
    model.calibrate(train_x[:64])

    # Supervised learning over training episodes
    for ep in range(N_TRAIN):
        model.learn_episode(train_x[ep], int(train_y[ep]))

    return model, train_x, train_y


def main():
    print("=" * 80)
    print("SCAR-X ROUND 9c: INFORMATION LOCALIZATION & CAUSAL DIAGNOSTICS")
    print(f"DEV SEEDS: {DEV_SEEDS} | N_TRAIN: {N_TRAIN} | N_TEST: {N_TEST}")
    print("=" * 80)

    start_time = time.time()
    per_seed_results = []

    for seed_idx, seed in enumerate(DEV_SEEDS):
        print(f"\n--- Running DEV Seed {seed} ({seed_idx + 1}/{len(DEV_SEEDS)}) ---")
        model, train_x, train_y = train_reference_model(seed)
        print(f"  Reference model trained: {model.episodes_learned} episodes, {model.hidden_updates} hidden updates")

        # Generate strictly paired test batch
        test_batch = generate_paired_dataset(seed=seed + 5000, count=N_TEST, ood=False)
        test_y = test_batch.labels

        # 1. Evaluate Direct Accuracy Across All 6 Paired Conditions
        cond_direct_acc = {}
        cond_traces = {}
        for cond in PAIRED_CONDITIONS:
            seqs = test_batch.sequences[cond]
            traces = [trace_episode(model, seqs[i], i, int(test_y[i])) for i in range(N_TEST)]
            cond_traces[cond] = traces
            preds = np.array([tr.prediction for tr in traces], dtype=np.int32)
            acc = float(np.mean(preds == test_y))
            cond_direct_acc[cond] = acc
            print(f"  Condition [{cond:18s}]: Direct Acc = {acc * 100:5.2f}%")

        # 2. Multi-Layer Decodability Tracing (Clean vs Standard Medium Distractor)
        print("  Evaluating multi-layer decodability (S1..S6)...")
        train_traces_clean = [trace_episode(model, train_x[i], i, int(train_y[i])) for i in range(min(128, N_TRAIN))]
        clean_decodability = evaluate_multi_layer_decodability(
            train_traces=train_traces_clean,
            test_traces=cond_traces["clean"],
            raw_train_x=train_x[: len(train_traces_clean)],
            raw_test_x=test_batch.sequences["clean"],
        )
        print(f"    Clean Decodability Attribution: {clean_decodability['attribution_category']}")
        for st in ("S1_Input", "S2_Voltage", "S3_Spikes", "S4_Adaptation", "S5_Traces", "S6_Features"):
            lin = clean_decodability[st]["linear_probe"] * 100
            nonlin = clean_decodability[st]["nonlinear_probe"] * 100
            print(f"      {st:15s} | Lin: {lin:5.1f}% | NonLin: {nonlin:5.1f}%")

        # Trace noisy stream multi-layer decodability
        noisy_x = test_batch.sequences["medium_distractor"]
        train_noisy_batch = generate_paired_dataset(seed=seed + 9000, count=128, ood=False)
        train_traces_noisy = [
            trace_episode(model, train_noisy_batch.sequences["medium_distractor"][i], i, int(train_noisy_batch.labels[i]))
            for i in range(128)
        ]
        noisy_decodability = evaluate_multi_layer_decodability(
            train_traces=train_traces_noisy,
            test_traces=cond_traces["medium_distractor"],
            raw_train_x=train_noisy_batch.sequences["medium_distractor"],
            raw_test_x=noisy_x,
        )
        print(f"    Noisy Decodability Attribution: {noisy_decodability['attribution_category']}")
        for st in ("S1_Input", "S2_Voltage", "S3_Spikes", "S4_Adaptation", "S5_Traces", "S6_Features"):
            lin = noisy_decodability[st]["linear_probe"] * 100
            nonlin = noisy_decodability[st]["nonlinear_probe"] * 100
            print(f"      {st:15s} | Lin: {lin:5.1f}% | NonLin: {nonlin:5.1f}%")

        # 3. Causal State Restoration Counterfactual Interventions
        print("  Running counterfactual state restorations...")
        causal_results = run_causal_restoration_experiment(
            model=model,
            clean_seqs=test_batch.sequences["clean"],
            noisy_seqs=test_batch.sequences["medium_distractor"],
            labels=test_y,
            distractor_ticks=(3, 7, 8),
        )
        print(f"    Clean Acc: {causal_results['clean_direct_acc']*100:5.2f}% | Noisy Acc: {causal_results['noisy_direct_acc']*100:5.2f}%")
        print(f"    Intervention A (Voltage Restoration): {causal_results['vsr_direct_acc']*100:5.2f}% (Recovery: {causal_results['recovery_vsr']*100:5.1f}%)")
        print(f"    Intervention B (Spike Restoration):   {causal_results['ssr_direct_acc']*100:5.2f}%")
        print(f"    Intervention C (Adaptation Restor.):  {causal_results['asr_direct_acc']*100:5.2f}%")
        print(f"    Intervention A+B+C (Full Restoration):{causal_results['fsr_direct_acc']*100:5.2f}% (Recovery: {causal_results['recovery_fsr']*100:5.1f}%)")
        print(f"    Mean Voltage Diff at Distractor: {causal_results['mean_voltage_perturbation']:.4f}")
        print(f"    Mean Extra Spikes at Distractor: {causal_results['mean_extra_spikes']:.2f}")

        per_seed_results.append({
            "seed": seed,
            "cond_direct_acc": cond_direct_acc,
            "clean_decodability": clean_decodability,
            "noisy_decodability": noisy_decodability,
            "causal_restoration": causal_results,
        })

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"DIAGNOSTICS COMPLETE (Elapsed: {elapsed:.2f}s)")
    print("=" * 80)

    # Aggregate Statistics across 3 independent seeds
    print("\nSUMMARY ACROSS 3 INDEPENDENT DEV SEEDS:")
    summary_conditions = {}
    for cond in PAIRED_CONDITIONS:
        accs = [r["cond_direct_acc"][cond] * 100 for r in per_seed_results]
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        summary_conditions[cond] = {"mean": mean_acc, "std": std_acc, "raw": accs}
        print(f"  {cond:20s}: Mean = {mean_acc:5.2f}% (std = {std_acc:4.2f}%) | Raw: {[round(x, 2) for x in accs]}")

    summary_causal = {}
    for key in ("clean_direct_acc", "noisy_direct_acc", "vsr_direct_acc", "ssr_direct_acc", "asr_direct_acc", "fsr_direct_acc", "recovery_vsr", "recovery_fsr"):
        vals = [r["causal_restoration"][key] for r in per_seed_results]
        summary_causal[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "raw": vals}

    print("\nCAUSAL INTERVENTIONS SUMMARY:")
    print(f"  Noisy Baseline:           {summary_causal['noisy_direct_acc']['mean']*100:5.2f}%")
    print(f"  Voltage Restoration:      {summary_causal['vsr_direct_acc']['mean']*100:5.2f}% (Recovery: {summary_causal['recovery_vsr']['mean']*100:5.1f}%)")
    print(f"  Spike Restoration:        {summary_causal['ssr_direct_acc']['mean']*100:5.2f}%")
    print(f"  Adaptation Restoration:   {summary_causal['asr_direct_acc']['mean']*100:5.2f}%")
    print(f"  Full State Restoration:   {summary_causal['fsr_direct_acc']['mean']*100:5.2f}% (Recovery: {summary_causal['recovery_fsr']['mean']*100:5.1f}%)")

    # Save artifact
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "r9c_information_diagnostics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "study": "r9c_information_localization_and_causal_diagnostics",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dev_seeds": DEV_SEEDS,
            "summary_conditions": summary_conditions,
            "summary_causal": summary_causal,
            "per_seed_results": per_seed_results,
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"\nArtifact saved to: {out_file}")


if __name__ == "__main__":
    main()
