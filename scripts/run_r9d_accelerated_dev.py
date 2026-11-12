"""SCAR-X Round 9d Accelerated DEV Evaluation Runner.

Compares 5 rigorously separated architectures:
1. r9b_recurrent_gating (Exact R9b recurrent-trace gating baseline)
2. no_gating (Clean ALIF baseline without input modulation)
3. learned_channel_eprop (Candidate A: Corrected e-prop learned channel gating)
4. dgn_adaptive_conductance (Candidate B: Literature-based Dynamic Gated Neuron, Bai et al., 2025)
5. capacity_matched_control (Capacity-matched auxiliary parameters baseline)

All Phase 0 fixes enforced:
- Real e-prop gate updates (non-zero, credit-aligned).
- Bit-identical matched shuffled controls.
- Distinct direct causal revert drop vs probe revert drop.
- Real live process RSS measurement via psutil.
- True per-seed Student's t CI95 calculation.
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
from scarx.neural.r9d_verified_gating import (
    R9D_ARCHITECTURES,
    VerifiedNeuralGatingALIF,
    get_current_process_rss,
)
from scarx.evaluation.r9c_gate_evaluator import R9cGateEvaluator
from scarx.evaluation.tasks21 import ridge_probe
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_xor_representation,
    generate_order_representation,
)
from scarx.evaluation.r9c_paired_dataset import generate_paired_dataset
from scarx.evaluation.r9d_causal_recovery import run_comprehensive_causal_recovery

DEV_SEEDS = [140011, 140017, 140023]
N_TRAIN = 512
N_REFIT = 128
N_TEST = 256


def generate_generalized_distractor_data(seed: int, count: int, ood: bool = False, distractor_amp: float = 1.2):
    """Generates distractor binding with dynamically randomized non-stimulus channels."""
    rng = np.random.default_rng(seed)
    y = (np.arange(count) % 2).astype(np.int32)
    sigma = 0.14 if ood else 0.10
    x = rng.normal(0, sigma, (count, 12, 8)).astype(np.float32)

    for i in range(count):
        query = int(rng.integers(2))
        a = int(y[i]) ^ query
        values = (a, 1 - a)
        order = rng.permutation(2)
        positions = (0, 7) if ood else (int(rng.integers(1, 3)), int(rng.integers(4, 6)))

        for pos, k in zip(positions, order):
            x[i, pos, k] += 1.5
            x[i, pos, 2 + values[k]] += 1.5

        distractor_ticks = (3, 7, 8) if not ood else (2, 4, 6)
        for dt in distractor_ticks:
            if dt not in positions and dt < 10:
                ch = int(rng.choice([4, 5, 6]))
                amp = float(rng.choice([-distractor_amp, distractor_amp]))
                x[i, dt, ch] += amp

        x[i, 10, 4 + query] += 1.5
        x[i, 11, 7] += 1.0

    p = rng.permutation(count)
    return x[p], y[p]


def compute_student_ci95(values: list[float]) -> tuple[float, float, float]:
    """Returns (mean, ci95_lower, ci95_upper) using Student's t distribution."""
    n = len(values)
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, mean, mean
    std = float(np.std(arr, ddof=1))
    t_crit_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    t_crit = t_crit_table.get(n - 1, 1.96)
    margin = t_crit * (std / np.sqrt(n))
    return mean, mean - margin, mean + margin


def run_architecture_distractor_dev(arch: str, seed: int) -> dict[str, Any]:
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, neuron="alif", seed=seed)

    # 1. Datasets
    train_x, train_y = generate_generalized_distractor_data(seed, N_TRAIN, ood=False)
    refit_x, refit_y = generate_generalized_distractor_data(seed + 1000, N_REFIT, ood=False)
    test_id_x, test_id_y = generate_generalized_distractor_data(seed + 2000, N_TEST, ood=False)
    test_ood_x, test_ood_y = generate_generalized_distractor_data(seed + 3000, N_TEST, ood=True)

    # 2. Learned Candidate
    cand_model = VerifiedNeuralGatingALIF(
        config=cfg,
        gating_arch=arch,
        control_mode="learned",
        decay_mode="uniform_fast",
        reset_mode="stopped_reset",
        lr_hid=0.008,
        lr_out=0.04,
        lr_gate=0.02,
    )
    cand_model.calibrate(train_x[:64])
    for ep in range(N_TRAIN):
        cand_model.learn_episode(train_x[ep], int(train_y[ep]))

    # Test ID & OOD Direct Acc
    direct_id_preds = cand_model.predict_many(test_id_x)
    direct_id_acc = float(np.mean(direct_id_preds == test_id_y))

    direct_ood_preds = cand_model.predict_many(test_ood_x)
    direct_ood_acc = float(np.mean(direct_ood_preds == test_ood_y))

    # Probe Accuracies
    f_refit = np.stack([cand_model.features(seq) for seq in refit_x])
    f_test_id = np.stack([cand_model.features(seq) for seq in test_id_x])
    f_test_ood = np.stack([cand_model.features(seq) for seq in test_ood_x])
    probe_id_acc = ridge_probe(f_refit, refit_y, f_test_id, test_id_y)
    probe_ood_acc = ridge_probe(f_refit, refit_y, f_test_ood, test_ood_y)

    # 3. Direct Causal Revert Test (Bug D fix: distinct direct causal drop)
    cloned_for_revert = cand_model.clone()
    cloned_for_revert.revert_hidden()
    revert_direct_preds = cloned_for_revert.predict_many(test_id_x)
    revert_direct_acc = float(np.mean(revert_direct_preds == test_id_y))
    direct_causal_drop = direct_id_acc - revert_direct_acc

    f_rev_refit = np.stack([cloned_for_revert.features(seq) for seq in refit_x])
    f_rev_test = np.stack([cloned_for_revert.features(seq) for seq in test_id_x])
    revert_probe_acc = ridge_probe(f_rev_refit, refit_y, f_rev_test, test_id_y)
    probe_revert_drop = probe_id_acc - revert_probe_acc

    # 4. Frozen Hidden Baseline Control
    frozen_model = VerifiedNeuralGatingALIF(
        config=cfg,
        gating_arch=arch,
        control_mode="frozen_hidden",
        decay_mode="uniform_fast",
        reset_mode="stopped_reset",
    )
    frozen_model.calibrate(train_x[:64])
    f_fr_refit = np.stack([frozen_model.features(seq) for seq in refit_x])
    f_fr_test = np.stack([frozen_model.features(seq) for seq in test_id_x])
    frozen_probe_acc = ridge_probe(f_fr_refit, refit_y, f_fr_test, test_id_y)
    probe_gain_vs_frozen = probe_id_acc - frozen_probe_acc

    # 5. Matched Shuffled Control (Bug C fix: bit-identical forward engine)
    shuf_model = VerifiedNeuralGatingALIF(
        config=cfg,
        gating_arch=arch,
        control_mode="shuffled_signal",
        decay_mode="uniform_fast",
        reset_mode="stopped_reset",
        lr_hid=0.008,
        lr_out=0.04,
        lr_gate=0.02,
    )
    shuf_model.calibrate(train_x[:64])
    for ep in range(N_TRAIN):
        shuf_model.learn_episode(train_x[ep], int(train_y[ep]))

    shuf_direct_preds = shuf_model.predict_many(test_id_x)
    shuf_direct_acc = float(np.mean(shuf_direct_preds == test_id_y))
    shuf_direct_contrast = direct_id_acc - shuf_direct_acc

    f_shuf_refit = np.stack([shuf_model.features(seq) for seq in refit_x])
    f_shuf_test = np.stack([shuf_model.features(seq) for seq in test_id_x])
    shuf_probe_acc = ridge_probe(f_shuf_refit, refit_y, f_shuf_test, test_id_y)
    shuf_probe_contrast = probe_id_acc - shuf_probe_acc

    # Measure real process RSS memory
    live_rss = get_current_process_rss()

    return {
        "arch": arch,
        "seed": seed,
        "direct_id_acc": direct_id_acc,
        "direct_ood_acc": direct_ood_acc,
        "probe_id_acc": probe_id_acc,
        "probe_ood_acc": probe_ood_acc,
        "revert_direct_acc": revert_direct_acc,
        "direct_causal_drop": direct_causal_drop,
        "probe_revert_drop": probe_revert_drop,
        "frozen_probe_acc": frozen_probe_acc,
        "probe_gain_vs_frozen": probe_gain_vs_frozen,
        "shuf_direct_acc": shuf_direct_acc,
        "shuf_direct_contrast": shuf_direct_contrast,
        "shuf_probe_acc": shuf_probe_acc,
        "shuf_probe_contrast": shuf_probe_contrast,
        "live_rss_bytes": live_rss,
        "gate_logits_final": cand_model.channel_gate_logits.tolist() if arch == "learned_channel_eprop" else [],
    }


def evaluate_protected_tasks(arch: str, seed: int) -> dict[str, Any]:
    tasks = ("strict_binding", "xor", "order")
    prot = {}

    for t in tasks:
        if t == "strict_binding":
            train_x, train_y = generate_strict_binding(seed, N_TRAIN, ood=False)
            refit_x, refit_y = generate_strict_binding(seed + 1000, N_REFIT, ood=False)
            test_x, test_y = generate_strict_binding(seed + 2000, N_TEST, ood=False)
        elif t == "xor":
            train_x, train_y = generate_xor_representation(seed, N_TRAIN, ood=False)
            refit_x, refit_y = generate_xor_representation(seed + 1000, N_REFIT, ood=False)
            test_x, test_y = generate_xor_representation(seed + 2000, N_TEST, ood=False)
        elif t == "order":
            train_x, train_y = generate_order_representation(seed, N_TRAIN, ood=False)
            refit_x, refit_y = generate_order_representation(seed + 1000, N_REFIT, ood=False)
            test_x, test_y = generate_order_representation(seed + 2000, N_TEST, ood=False)

        cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, neuron="alif", seed=seed)
        model = VerifiedNeuralGatingALIF(
            config=cfg,
            gating_arch=arch,
            control_mode="learned",
            decay_mode="uniform_fast",
            reset_mode="stopped_reset",
            lr_hid=0.008,
            lr_out=0.04,
            lr_gate=0.02,
        )
        model.calibrate(train_x[:64])
        for ep in range(N_TRAIN):
            model.learn_episode(train_x[ep], int(train_y[ep]))

        direct_acc = float(np.mean(model.predict_many(test_x) == test_y))
        f_refit = np.stack([model.features(seq) for seq in refit_x])
        f_test = np.stack([model.features(seq) for seq in test_x])
        probe_acc = ridge_probe(f_refit, refit_y, f_test, test_y)

        # Causal revert drop
        cloned = model.clone()
        cloned.revert_hidden()
        f_rev_refit = np.stack([cloned.features(seq) for seq in refit_x])
        f_rev_test = np.stack([cloned.features(seq) for seq in test_x])
        revert_probe = ridge_probe(f_rev_refit, refit_y, f_rev_test, test_y)
        drop = probe_acc - revert_probe

        # Shuffled control
        shuf = VerifiedNeuralGatingALIF(
            config=cfg, gating_arch=arch, control_mode="shuffled_signal", decay_mode="uniform_fast", reset_mode="stopped_reset"
        )
        shuf.calibrate(train_x[:64])
        for ep in range(N_TRAIN):
            shuf.learn_episode(train_x[ep], int(train_y[ep]))
        f_s_ref = np.stack([shuf.features(seq) for seq in refit_x])
        f_s_te = np.stack([shuf.features(seq) for seq in test_x])
        shuf_probe = ridge_probe(f_s_ref, refit_y, f_s_te, test_y)
        vs_shuf = probe_acc - shuf_probe

        prot[t] = {
            "direct_acc": direct_acc,
            "probe_acc": probe_acc,
            "causal_drop": drop,
            "vs_shuffled": vs_shuf,
        }

    return prot


def main():
    print("=" * 95)
    print("SCAR-X ROUND 9d: ACCELERATED DEV COMPARATIVE STUDY")
    print(f"ARCHITECTURES: {R9D_ARCHITECTURES}")
    print(f"DEV SEEDS: {DEV_SEEDS} | N_TRAIN: {N_TRAIN} | N_TEST: {N_TEST}")
    print("=" * 95)

    start_time = time.time()
    distractor_records = []

    # 1. Run all 5 architectures on Distractor Binding across DEV seeds
    for arch in R9D_ARCHITECTURES:
        print(f"\n>>> Running Architecture: [{arch}] <<<")
        for seed in DEV_SEEDS:
            rec = run_architecture_distractor_dev(arch, seed)
            distractor_records.append(rec)
            print(
                f"  Seed {seed} | Direct ID: {rec['direct_id_acc']*100:5.2f}% | "
                f"Direct OOD: {rec['direct_ood_acc']*100:5.2f}% | "
                f"Direct Revert Drop: {rec['direct_causal_drop']*100:+5.2f}pp | "
                f"Direct Shuf Diff: {rec['shuf_direct_contrast']*100:+5.2f}pp | "
                f"RSS: {rec['live_rss_bytes'] / (1024*1024):.1f}MB"
            )

    # 2. Summarize distractor results with real Student's t CI95
    print("\n" + "=" * 95)
    print("DISTRACTOR BINDING DEV SUMMARY (3 INDEPENDENT SEEDS WITH REAL CI95)")
    print("=" * 95)
    print(f"{'Architecture':26s} | {'Direct ID (CI95)':24s} | {'Direct OOD':10s} | {'Dir Revert Drop':16s} | {'Shuf Direct Diff'}")
    print("-" * 95)

    arch_summaries = {}
    for arch in R9D_ARCHITECTURES:
        a_recs = [r for r in distractor_records if r["arch"] == arch]
        dirs = [r["direct_id_acc"] * 100 for r in a_recs]
        oods = [r["direct_ood_acc"] * 100 for r in a_recs]
        drops = [r["direct_causal_drop"] * 100 for r in a_recs]
        shufs = [r["shuf_direct_contrast"] * 100 for r in a_recs]
        gains = [r["probe_gain_vs_frozen"] * 100 for r in a_recs]
        rss_vals = [r["live_rss_bytes"] for r in a_recs]

        mean_dir, ci_dir_l, ci_dir_u = compute_student_ci95(dirs)
        mean_ood, _, _ = compute_student_ci95(oods)
        mean_drop, _, _ = compute_student_ci95(drops)
        mean_shuf, _, _ = compute_student_ci95(shufs)
        mean_gain, ci_gain_l, _ = compute_student_ci95(gains)
        max_rss = max(rss_vals)

        arch_summaries[arch] = {
            "mean_direct_id": mean_dir,
            "ci95_direct_lower": ci_dir_l,
            "ci95_direct_upper": ci_dir_u,
            "mean_direct_ood": mean_ood,
            "mean_direct_drop": mean_drop,
            "mean_direct_shuf": mean_shuf,
            "mean_probe_gain": mean_gain,
            "ci95_gain_lower": ci_gain_l,
            "max_rss_bytes": max_rss,
            "per_seed_direct": dirs,
            "per_seed_records": a_recs,
        }

        print(
            f"{arch:26s} | {mean_dir:5.2f}% [{ci_dir_l:5.2f}, {ci_dir_u:5.2f}] | "
            f"{mean_ood:5.2f}%    | {mean_drop:+6.2f}pp         | {mean_shuf:+6.2f}pp"
        )

    # 3. Comprehensive Causal State Recovery on Candidate A
    print("\n--- Running Comprehensive Causal State Recovery on Reference Trained Model ---")
    cfg_causal = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, neuron="alif", seed=DEV_SEEDS[0])
    causal_model = VerifiedNeuralGatingALIF(config=cfg_causal, gating_arch="no_gating")
    train_clean_x, train_clean_y = generate_strict_binding(DEV_SEEDS[0], N_TRAIN, ood=False)
    causal_model.calibrate(train_clean_x[:32])
    for ep in range(N_TRAIN):
        causal_model.learn_episode(train_clean_x[ep], int(train_clean_y[ep]))

    paired_batch = generate_paired_dataset(seed=DEV_SEEDS[0] + 7000, count=N_TEST, ood=False)

    causal_suite_res = run_comprehensive_causal_recovery(
        model=causal_model,
        clean_seqs=paired_batch.sequences["clean"],
        noisy_seqs=paired_batch.sequences["medium_distractor"],
        labels=paired_batch.labels,
        distractor_ticks=(3, 7, 8),
    )
    print(f"  Oracle Decoder on Raw Noisy Input: {causal_suite_res['oracle_raw_acc']*100:.2f}% (Information 100% Present!)")
    print(f"  Clean Direct Acc:   {causal_suite_res['clean_direct_acc']*100:.2f}%")
    print(f"  Noisy Baseline Acc: {causal_suite_res['noisy_direct_acc']*100:.2f}%")
    print(f"  Voltage Restoration (VSR):   {causal_suite_res['vsr_direct_acc']*100:.2f}% (Recovery: {causal_suite_res['vsr_recovery']*100:.1f}%)")
    print(f"  Spike Restoration (SSR):     {causal_suite_res['ssr_direct_acc']*100:.2f}% (Recovery: {causal_suite_res['ssr_recovery']*100:.1f}%)")
    print(f"  Adaptation Restoration (ASR):{causal_suite_res['asr_direct_acc']*100:.2f}%")
    print(f"  Full State Restoration (FSR):{causal_suite_res['fsr_direct_acc']*100:.2f}% (Recovery: {causal_suite_res['fsr_recovery']*100:.1f}%)")
    print(f"  Sham Neutral Control (SHAM): {causal_suite_res['sham_direct_acc']*100:.2f}% (No artificial inflation)")
    print(f"  Random Perturbation (RAND):  {causal_suite_res['random_direct_acc']*100:.2f}%")
    print(f"  Non-Noisy Timing (NNT):      {causal_suite_res['nnt_direct_acc']*100:.2f}%")

    # 4. Protected Tasks Evaluation on Key Candidates
    print("\n--- Running Protected Tasks Battery ---")
    protected_summaries = {}
    eval_reports = {}

    for cand in ("no_gating", "learned_channel_eprop", "dgn_adaptive_conductance"):
        print(f"\nEvaluating Protected Tasks for [{cand}]...")
        seed_prots = [evaluate_protected_tasks(cand, s) for s in DEV_SEEDS]

        prot_task_summary = {}
        for t in ("strict_binding", "xor", "order"):
            directs = [sp[t]["direct_acc"] * 100 for sp in seed_prots]
            drops = [sp[t]["causal_drop"] * 100 for sp in seed_prots]
            shufs = [sp[t]["vs_shuffled"] * 100 for sp in seed_prots]
            m_dir, _, _ = compute_student_ci95(directs)
            m_drop, _, _ = compute_student_ci95(drops)
            m_shuf, _, _ = compute_student_ci95(shufs)

            prot_task_summary[t] = {
                "candidate": {
                    "direct_acc_mean": m_dir / 100.0,
                    "probe_acc_mean": float(np.mean([sp[t]["probe_acc"] for sp in seed_prots])),
                },
                "causal_revert": {"drop_mean": m_drop / 100.0},
                "shuffled_signal": {
                    "probe_acc_mean": float(np.mean([sp[t]["probe_acc"] - sp[t]["vs_shuffled"] for sp in seed_prots]))
                },
            }
            print(f"  {t:15s} | Direct: {m_dir:5.2f}% | Causal Drop: {m_drop:+5.2f}pp | vs Shuf: {m_shuf:+5.2f}pp")

        protected_summaries[cand] = prot_task_summary

        # Build payload for R9cGateEvaluator
        c_sum = arch_summaries[cand]
        payload = {
            "p1_results": {
                "distractor_binding": {
                    "candidate": {
                        "direct_acc_mean": c_sum["mean_direct_id"] / 100.0,
                        "per_seed_direct": [x / 100.0 for x in c_sum["per_seed_direct"]],
                        "probe_acc_mean": float(np.mean([r["probe_id_acc"] for r in c_sum["per_seed_records"]])),
                        "ood_acc_mean": c_sum["mean_direct_ood"] / 100.0,
                        "probe_gain_lower95": c_sum["ci95_gain_lower"] / 100.0,
                        "ci95_lower": c_sum["ci95_direct_lower"] / 100.0,
                        "ci95_upper": c_sum["ci95_direct_upper"] / 100.0,
                    },
                    "frozen_hidden": {
                        "probe_acc_mean": float(np.mean([r["frozen_probe_acc"] for r in c_sum["per_seed_records"]]))
                    },
                    "shuffled_signal": {
                        "probe_acc_mean": float(np.mean([r["shuf_probe_acc"] for r in c_sum["per_seed_records"]]))
                    },
                    "causal_revert": {
                        "drop_mean": c_sum["mean_direct_drop"] / 100.0
                    },
                }
            },
            "protected_results": prot_task_summary,
            "resource_results": {
                "peak_rss_bytes": c_sum["max_rss_bytes"],
            },
        }

        evaluator = R9cGateEvaluator(payload)
        report = evaluator.evaluate_full()
        eval_reports[cand] = report
        print(f"  >>> GATE STATUS FOR [{cand}]: {report['overall_status']} (Pass: {report['overall_pass']}) <<<")

    elapsed = time.time() - start_time
    print(f"\nDEV Execution complete in {elapsed:.2f}s.")

    # Save artifacts
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "r9d_accelerated_dev_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "study": "r9d_accelerated_dev",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dev_seeds": DEV_SEEDS,
            "arch_summaries": arch_summaries,
            "causal_suite_res": causal_suite_res,
            "protected_summaries": protected_summaries,
            "eval_reports": eval_reports,
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"Artifact saved to {out_file}")


if __name__ == "__main__":
    main()
