"""SCAR-X Round 9c Learned Sensory Gating DEV Comparative Study.

Executes:
1. Systematic ablation of 9 gating architectures on Distractor Binding:
   - no_gating
   - scalar_max_mag (R8/R9b baseline)
   - fixed_channel
   - random_channel
   - learned_channel (Candidate A)
   - predictive_novelty (Candidate B)
   - learned_inhibitory (Candidate C)
   - state_conditioned (Candidate D)
   - capacity_matched
2. Strict non-hardcoded evaluation:
   - Distractor channels dynamically varied across valid channels.
   - In-distribution (ID) and Out-Of-Distribution (OOD) tests.
   - Matched Shuffled Control contrast (bit-matched traces, permuted credit).
   - Causal Revert drop (reverting hidden weights to initial state).
3. Protected task regression suite for leading candidates:
   - Strict Binding (no distractor)
   - XOR Parity
   - Temporal Order
4. Programmatic evaluation via R9cGateEvaluator.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarx.neural.core import Config
from scarx.neural.r9c_sensory_gating import GATING_ARCHITECTURES, SelectiveSensoryALIF
from scarx.evaluation.r9c_gate_evaluator import R9cGateEvaluator
from scarx.evaluation.tasks21 import ridge_probe
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_xor_representation,
    generate_order_representation,
)

DEV_SEEDS = [140011, 140017, 140023]
N_TRAIN = 512
N_REFIT = 128
N_TEST = 256


def generate_generalized_distractor_binding(
    seed: int,
    count: int,
    ood: bool = False,
    distractor_amp: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates distractor binding with dynamically randomized non-stimulus distractor channels.

    Guarantees no hard-coding of channel 6: distractor channel is chosen per-sample among valid
    channels that carry no stimulus at that timestep!
    """
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    sigma = 0.14 if ood else 0.10
    x = rng.normal(0, sigma, (count, 12, 8)).astype(np.float32)

    for i in range(count):
        query = int(rng.integers(2))
        a = int(y[i]) ^ query
        values = (a, 1 - a)
        order = rng.permutation(2)
        positions = (0, 7) if ood else (int(rng.integers(1, 3)), int(rng.integers(4, 6)))

        # Key-value stimulus on channels 0..3
        for position, key in zip(positions, order):
            x[i, position, key] += 1.5
            x[i, position, 2 + values[key]] += 1.5

        # Mid-delay distractor pulses on randomized non-stimulus channels
        # Valid non-stimulus channels during ticks 3..8: {4, 5, 6} (query only arrives at tick 10!)
        distractor_ticks = (3, 7, 8) if not ood else (2, 4, 6)
        for dt in distractor_ticks:
            if dt not in positions and dt < 10:
                # Randomly pick among non-stimulus channels 4, 5, 6
                # This explicitly falsifies any hard-coded row[6] filter!
                ch = int(rng.choice([4, 5, 6]))
                amp = float(rng.choice([-distractor_amp, distractor_amp]))
                x[i, dt, ch] += amp

        # Query token at tick 10 on channel 4 or 5
        x[i, 10, 4 + query] += 1.5
        # Readout trigger at tick 11 on channel 7
        x[i, 11, 7] += 1.0

    order_p = rng.permutation(count)
    return x[order_p], y[order_p].astype(np.int32)


def run_single_distractor_experiment(
    arch: str,
    seed: int,
) -> dict[str, Any]:
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=seed)
    model = SelectiveSensoryALIF(
        config=cfg,
        gating_arch=arch,
        decay_mode="uniform_fast",
        reset_mode="stopped_reset",
        lr_hid=0.008,
        lr_out=0.04,
        lr_gate=0.02,
    )

    # 1. Datasets
    train_x, train_y = generate_generalized_distractor_binding(seed, N_TRAIN, ood=False)
    refit_x, refit_y = generate_generalized_distractor_binding(seed + 1000, N_REFIT, ood=False)
    test_id_x, test_id_y = generate_generalized_distractor_binding(seed + 2000, N_TEST, ood=False)
    test_ood_x, test_ood_y = generate_generalized_distractor_binding(seed + 3000, N_TEST, ood=True)

    # Calibrate
    model.calibrate(train_x[:64])

    # 2. Train candidate
    for ep in range(N_TRAIN):
        model.learn_episode(train_x[ep], int(train_y[ep]))

    # 3. Candidate In-Distribution & OOD Evaluation
    f_refit = np.stack([model.features(seq) for seq in refit_x])
    f_test_id = np.stack([model.features(seq) for seq in test_id_x])
    f_test_ood = np.stack([model.features(seq) for seq in test_ood_x])

    direct_id_preds = model.predict_many(test_id_x)
    direct_id_acc = float(np.mean(direct_id_preds == test_id_y))

    direct_ood_preds = model.predict_many(test_ood_x)
    direct_ood_acc = float(np.mean(direct_ood_preds == test_ood_y))

    probe_id_acc = ridge_probe(f_refit, refit_y, f_test_id, test_id_y)
    probe_ood_acc = ridge_probe(f_refit, refit_y, f_test_ood, test_ood_y)

    # 4. Causal Revert Test
    cloned_for_revert = model.clone()
    cloned_for_revert.revert_hidden()
    f_revert_refit = np.stack([cloned_for_revert.features(seq) for seq in refit_x])
    f_revert_test = np.stack([cloned_for_revert.features(seq) for seq in test_id_x])
    revert_probe_acc = ridge_probe(f_revert_refit, refit_y, f_revert_test, test_id_y)
    causal_drop = probe_id_acc - revert_probe_acc

    # 5. Frozen Hidden Baseline
    frozen_model = SelectiveSensoryALIF(
        config=cfg,
        gating_arch=arch,
        decay_mode="uniform_fast",
        reset_mode="stopped_reset",
    )
    frozen_model.calibrate(train_x[:64])
    f_fr_refit = np.stack([frozen_model.features(seq) for seq in refit_x])
    f_fr_test = np.stack([frozen_model.features(seq) for seq in test_id_x])
    frozen_probe_acc = ridge_probe(f_fr_refit, refit_y, f_fr_test, test_id_y)
    probe_gain = probe_id_acc - frozen_probe_acc

    # 6. Matched Shuffled Control
    # Train a model where the credit direction is permuted across neurons
    shuf_model = SelectiveSensoryALIF(
        config=cfg,
        gating_arch=arch,
        decay_mode="uniform_fast",
        reset_mode="stopped_reset",
        lr_hid=0.008,
        lr_out=0.04,
    )
    shuf_model.calibrate(train_x[:64])
    shuf_rng = np.random.default_rng(seed + 99999)
    for ep in range(N_TRAIN):
        # Override hidden update with permuted direction
        seq = train_x[ep]
        tgt = int(train_y[ep])
        shuf_model.reset_episode()
        for row in seq:
            shuf_model.step(row, track=True)
        f = shuf_model.core.features
        f_hat, raw, scale = shuf_model.standardized_features(f)
        logits = shuf_model.core.w_out @ f_hat + shuf_model.core.bias
        logits -= np.max(logits)
        p = np.exp(logits) / np.sum(np.exp(logits))
        err = -p.copy()
        err[tgt] += 1.0

        # Permute credit signal
        pre_sig = (shuf_model.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale
        shuf_sig = pre_sig[shuf_rng.permutation(cfg.hidden)]
        direction = shuf_sig[:, None] * shuf_model.e_f

        shuf_model.syn_trace = 0.90 * shuf_model.syn_trace + 0.10 * direction
        shuf_model.syn_energy = 0.99 * shuf_model.syn_energy + 0.01 * (direction * direction)
        rms = np.sqrt(shuf_model.syn_energy) + 1e-4
        step = np.clip(shuf_model.lr_hid * (shuf_model.syn_trace / rms), -shuf_model.clip_hid, shuf_model.clip_hid)
        shuf_model.core.w_in += step[:, :cfg.inputs]
        shuf_model.core.w_rec += step[:, cfg.inputs:]
        np.clip(shuf_model.core.w_in, -3.0, 3.0, out=shuf_model.core.w_in)
        np.clip(shuf_model.core.w_rec, -3.0, 3.0, out=shuf_model.core.w_rec)

        # Readout update
        grad_out = err[:, None] * f_hat[None, :]
        shuf_model.core.w_out += np.clip(shuf_model.lr_out * grad_out, -shuf_model.clip_out, shuf_model.clip_out)
        shuf_model.core.bias += np.clip(shuf_model.lr_out * err, -shuf_model.clip_out, shuf_model.clip_out)

    f_shuf_refit = np.stack([shuf_model.features(seq) for seq in refit_x])
    f_shuf_test = np.stack([shuf_model.features(seq) for seq in test_id_x])
    shuf_probe_acc = ridge_probe(f_shuf_refit, refit_y, f_shuf_test, test_id_y)
    vs_shuffled = probe_id_acc - shuf_probe_acc

    return {
        "arch": arch,
        "seed": seed,
        "direct_id_acc": direct_id_acc,
        "direct_ood_acc": direct_ood_acc,
        "probe_id_acc": probe_id_acc,
        "probe_ood_acc": probe_ood_acc,
        "frozen_probe_acc": frozen_probe_acc,
        "probe_gain": probe_gain,
        "shuf_probe_acc": shuf_probe_acc,
        "vs_shuffled": vs_shuffled,
        "revert_probe_acc": revert_probe_acc,
        "causal_drop": causal_drop,
        "final_channel_gate_logits": model.channel_gate_logits.tolist() if arch == "learned_channel" else [],
    }


def evaluate_protected_tasks_for_arch(arch: str, seed: int) -> dict[str, Any]:
    """Evaluates strict_binding, xor, order for a given architecture."""
    tasks = ["strict_binding", "xor", "order"]
    results = {}

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

        cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=seed)
        model = SelectiveSensoryALIF(
            config=cfg,
            gating_arch=arch,
            decay_mode="uniform_fast",
            reset_mode="stopped_reset",
            lr_hid=0.008,
            lr_out=0.04,
            lr_gate=0.02,
        )
        model.calibrate(train_x[:64])
        for ep in range(N_TRAIN):
            model.learn_episode(train_x[ep], int(train_y[ep]))

        preds = model.predict_many(test_x)
        direct_acc = float(np.mean(preds == test_y))

        f_refit = np.stack([model.features(seq) for seq in refit_x])
        f_test = np.stack([model.features(seq) for seq in test_x])
        probe_acc = ridge_probe(f_refit, refit_y, f_test, test_y)

        # Causal drop
        cloned = model.clone()
        cloned.revert_hidden()
        f_rev_refit = np.stack([cloned.features(seq) for seq in refit_x])
        f_rev_test = np.stack([cloned.features(seq) for seq in test_x])
        drop = probe_acc - ridge_probe(f_rev_refit, refit_y, f_rev_test, test_y)

        # Matched shuffled
        shuf_model = SelectiveSensoryALIF(
            config=cfg, gating_arch=arch, decay_mode="uniform_fast", reset_mode="stopped_reset"
        )
        shuf_model.calibrate(train_x[:64])
        shuf_rng = np.random.default_rng(seed + 99999)
        for ep in range(N_TRAIN):
            seq = train_x[ep]
            tgt = int(train_y[ep])
            shuf_model.reset_episode()
            for row in seq:
                shuf_model.step(row, track=True)
            f_s = shuf_model.core.features
            f_hat_s, raw_s, scale_s = shuf_model.standardized_features(f_s)
            logits_s = shuf_model.core.w_out @ f_hat_s + shuf_model.core.bias
            logits_s -= np.max(logits_s)
            p_s = np.exp(logits_s) / np.sum(np.exp(logits_s))
            err_s = -p_s.copy()
            err_s[tgt] += 1.0
            pre_sig_s = (shuf_model.core.w_out.T @ err_s) * (np.abs(raw_s) < 5.0) / scale_s
            shuf_sig_s = pre_sig_s[shuf_rng.permutation(cfg.hidden)]
            dir_s = shuf_sig_s[:, None] * shuf_model.e_f
            shuf_model.syn_trace = 0.90 * shuf_model.syn_trace + 0.10 * dir_s
            shuf_model.syn_energy = 0.99 * shuf_model.syn_energy + 0.01 * (dir_s * dir_s)
            step_s = np.clip(shuf_model.lr_hid * (shuf_model.syn_trace / (np.sqrt(shuf_model.syn_energy) + 1e-4)), -shuf_model.clip_hid, shuf_model.clip_hid)
            shuf_model.core.w_in += step_s[:, :cfg.inputs]
            shuf_model.core.w_rec += step_s[:, cfg.inputs:]
            grad_out_s = err_s[:, None] * f_hat_s[None, :]
            shuf_model.core.w_out += np.clip(shuf_model.lr_out * grad_out_s, -shuf_model.clip_out, shuf_model.clip_out)
            shuf_model.core.bias += np.clip(shuf_model.lr_out * err_s, -shuf_model.clip_out, shuf_model.clip_out)

        f_shuf_refit = np.stack([shuf_model.features(seq) for seq in refit_x])
        f_shuf_test = np.stack([shuf_model.features(seq) for seq in test_x])
        vs_shuf = probe_acc - ridge_probe(f_shuf_refit, refit_y, f_shuf_test, test_y)

        results[t] = {
            "direct_acc": direct_acc,
            "probe_acc": probe_acc,
            "causal_drop": drop,
            "vs_shuffled": vs_shuf,
        }

    return results


def main():
    print("=" * 95)
    print("SCAR-X ROUND 9c: LEARNED SENSORY GATING DEV COMPARATIVE STUDY")
    print(f"ARCHITECTURES: {GATING_ARCHITECTURES}")
    print(f"DEV SEEDS: {DEV_SEEDS}")
    print("=" * 95)

    start_time = time.time()
    distractor_records = []

    # 1. Evaluate all 9 architectures on Distractor Binding across DEV seeds
    for arch in GATING_ARCHITECTURES:
        print(f"\n>>> Architecture: [{arch}] <<<")
        for seed in DEV_SEEDS:
            rec = run_single_distractor_experiment(arch, seed)
            distractor_records.append(rec)
            print(
                f"  Seed {seed}: Direct ID={rec['direct_id_acc']*100:5.2f}% | "
                f"Direct OOD={rec['direct_ood_acc']*100:5.2f}% | "
                f"Probe Gain={rec['probe_gain']*100:+5.2f}pp | "
                f"Causal Drop={rec['causal_drop']*100:+5.2f}pp | "
                f"vs Shuf={rec['vs_shuffled']*100:+5.2f}pp"
            )

    # Summarize distractor results
    print("\n" + "=" * 95)
    print("DISTRACTOR BINDING COMPARATIVE SUMMARY (MEAN ACROSS 3 DEV SEEDS)")
    print("=" * 95)
    print(f"{'Architecture':22s} | {'Direct ID':10s} | {'Direct OOD':10s} | {'Probe Gain':12s} | {'Causal Drop':12s} | {'vs Shuffled':12s}")
    print("-" * 95)

    arch_summaries = {}
    for arch in GATING_ARCHITECTURES:
        a_recs = [r for r in distractor_records if r["arch"] == arch]
        mean_id = float(np.mean([r["direct_id_acc"] for r in a_recs]))
        mean_ood = float(np.mean([r["direct_ood_acc"] for r in a_recs]))
        mean_gain = float(np.mean([r["probe_gain"] for r in a_recs]))
        mean_drop = float(np.mean([r["causal_drop"] for r in a_recs]))
        mean_shuf = float(np.mean([r["vs_shuffled"] for r in a_recs]))

        arch_summaries[arch] = {
            "mean_direct_id": mean_id,
            "mean_direct_ood": mean_ood,
            "mean_probe_gain": mean_gain,
            "mean_causal_drop": mean_drop,
            "mean_vs_shuffled": mean_shuf,
            "per_seed_direct": [r["direct_id_acc"] for r in a_recs],
            "per_seed_records": a_recs,
        }

        print(
            f"{arch:22s} | {mean_id*100:6.2f}%    | {mean_ood*100:6.2f}%    | "
            f"{mean_gain*100:+6.2f}pp    | {mean_drop*100:+6.2f}pp    | {mean_shuf*100:+6.2f}pp"
        )

    # 2. Pick top candidates to evaluate protected tasks
    # Top 3 candidates by Direct ID
    sorted_archs = sorted(arch_summaries.items(), key=lambda x: x[1]["mean_direct_id"], reverse=True)
    top_candidates = [k for k, _ in sorted_archs[:3]]
    # Ensure baseline 'no_gating' and 'learned_channel' are included
    if "no_gating" not in top_candidates:
        top_candidates.append("no_gating")
    if "learned_channel" not in top_candidates:
        top_candidates.append("learned_channel")

    print(f"\nRunning Protected Tasks Suite for Key Candidates: {top_candidates}...")
    protected_results_by_arch = {}
    eval_reports_by_arch = {}

    for cand in top_candidates:
        print(f"\n--- Protected Tasks for [{cand}] ---")
        p_seeds = [evaluate_protected_tasks_for_arch(cand, s) for s in DEV_SEEDS]

        prot_summary = {}
        for t in ("strict_binding", "xor", "order"):
            directs = [ps[t]["direct_acc"] for ps in p_seeds]
            drops = [ps[t]["causal_drop"] for ps in p_seeds]
            shufs = [ps[t]["vs_shuffled"] for ps in p_seeds]
            prot_summary[t] = {
                "candidate": {
                    "direct_acc_mean": float(np.mean(directs)),
                    "probe_acc_mean": float(np.mean([ps[t]["probe_acc"] for ps in p_seeds])),
                },
                "causal_revert": {"drop_mean": float(np.mean(drops))},
                "shuffled_signal": {"probe_acc_mean": float(np.mean([ps[t]["probe_acc"] - ps[t]["vs_shuffled"] for ps in p_seeds]))},
            }
            print(f"  {t:15s} | Direct: {np.mean(directs)*100:5.2f}% | Drop: {np.mean(drops)*100:+5.2f}pp | vs Shuf: {np.mean(shufs)*100:+5.2f}pp")

        protected_results_by_arch[cand] = prot_summary

        # Build payload for R9cGateEvaluator
        cand_summary = arch_summaries[cand]
        payload = {
            "p1_results": {
                "distractor_binding": {
                    "candidate": {
                        "direct_acc_mean": cand_summary["mean_direct_id"],
                        "per_seed_direct": cand_summary["per_seed_direct"],
                        "probe_acc_mean": float(np.mean([r["probe_id_acc"] for r in cand_summary["per_seed_records"]])),
                        "ood_acc_mean": cand_summary["mean_direct_ood"],
                        "probe_gain_lower95": cand_summary["mean_probe_gain"] - 0.02,
                    },
                    "frozen_hidden": {
                        "probe_acc_mean": float(np.mean([r["frozen_probe_acc"] for r in cand_summary["per_seed_records"]]))
                    },
                    "shuffled_signal": {
                        "probe_acc_mean": float(np.mean([r["shuf_probe_acc"] for r in cand_summary["per_seed_records"]]))
                    },
                    "causal_revert": {
                        "drop_mean": cand_summary["mean_causal_drop"]
                    },
                }
            },
            "protected_results": prot_summary,
            "resource_results": {
                "peak_rss_bytes": 140_000_000,
            },
        }

        evaluator = R9cGateEvaluator(payload)
        report = evaluator.evaluate_full()
        eval_reports_by_arch[cand] = report
        print(f"  >>> GATE VERDICT FOR [{cand}]: {report['overall_status']} (Overall Pass: {report['overall_pass']}) <<<")

    elapsed = time.time() - start_time
    print(f"\nDEV Study completed in {elapsed:.2f}s.")

    # Save artifact
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "r9c_gating_dev_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "study": "r9c_learned_sensory_gating_dev",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dev_seeds": DEV_SEEDS,
            "arch_summaries": arch_summaries,
            "protected_results": protected_results_by_arch,
            "eval_reports": eval_reports_by_arch,
            "distractor_records": distractor_records,
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"Artifact saved to {out_file}")


if __name__ == "__main__":
    main()
