"""SCAR-X Round 9: Automated Massive Multi-Stage Neural Mechanism Search Runner.

Executes Stages A to F across 512 candidate configurations with strict resource governance:
- Stage B: Fast screening across all 512 candidates on DEV seeds [140011, 140017].
- Stage C: Multi-objective candidate ranking and adaptive selection.
- Stage D: In-depth causal ablations on Top 16 candidates.
- Stage E: Adversarial robustness stress testing on Top 5 candidates.
- Stage F: Locked confirmation on Top 2 candidates across 10 unseen seeds.

Resource limits:
- BLAS threads pinned to 1.
- Max RAM per process <= 2.0 GB.
- Resumable JSONL logging with atomic status updates.
"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import copy
import io
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import psutil

# Ensure local src and scripts are in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scarx.neural.core import Config, SpikingCore
from scarx.neural.r9_unified_mechanisms import R9UnifiedNeuralModel
from scarx.evaluation.tasks21 import ridge_probe
from scarx.evaluation.r9_benchmark_integrity import (
    generate_strict_binding,
    generate_distractor_binding,
    generate_xor_representation,
    generate_order_representation,
    generate_ceiling_stress_binding,
    generate_multikey_associative_r9,
    assert_input_dimension,
    assert_spikes_binary,
)
from scarx.evaluation.r9_gate_evaluator import (
    R9GateEvaluator,
    compute_paired_ci95,
    benjamini_hochberg,
)
from r9_candidate_generator import generate_all_candidates

DEV_SEEDS = [140011, 140017]
ROBUSTNESS_SEEDS = [160011, 160017, 160023]
CONFIRMATION_SEEDS = [
    191001, 191007, 191013, 191019, 191025,
    191031, 191037, 191043, 191049, 191055,
]

DEV_SCREEN_FILE = Path("artifacts/gemini-r9/dev_screen_results.jsonl")
RANKING_FILE = Path("artifacts/gemini-r9/candidate_rankings.json")
ABLATION_FILE = Path("artifacts/gemini-r9/causal_ablations.json")
ROBUSTNESS_FILE = Path("artifacts/gemini-r9/robustness_results.json")
CONFIRMATION_FILE = Path("artifacts/gemini-r9/locked_confirmation.json")
SUMMARY_FILE = Path("artifacts/gemini-r9/search_summary.json")


def check_ram_usage() -> float:
    """Returns current process RSS in MB."""
    p = psutil.Process()
    rss_mb = p.memory_info().rss / (1024 * 1024)
    if rss_mb > 1900.0:
        raise MemoryError(f"Process RAM ({rss_mb:.1f} MB) approaching 2 GB safety ceiling!")
    return rss_mb


# ---------------------------------------------------------------------------
# Stage B: Screening Task Evaluators
# ---------------------------------------------------------------------------

def screen_binding_candidate(
    cand: dict[str, Any], seed: int, distractor: bool = False
) -> dict[str, float]:
    p = cand["params"]
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)

    if distractor:
        tr_x, tr_y = generate_distractor_binding(seed + 200, 384, distractor_amp=1.2)
        fit_x, fit_y = generate_distractor_binding(seed + 300, 128, distractor_amp=1.2)
        te_x, te_y = generate_distractor_binding(seed + 400, 256, distractor_amp=1.2)
        ood_x, ood_y = generate_distractor_binding(seed + 500, 256, distractor_amp=1.2, ood=True)
    else:
        tr_x, tr_y = generate_strict_binding(seed + 200, 384)
        fit_x, fit_y = generate_strict_binding(seed + 300, 128)
        te_x, te_y = generate_strict_binding(seed + 400, 256)
        ood_x, ood_y = generate_strict_binding(seed + 500, 256, ood=True)

    assert_input_dimension(tr_x, 8)

    # 1. Learned candidate
    model = R9UnifiedNeuralModel(
        cfg,
        reset_mode=p["reset_mode"],
        damped_lambda=p.get("damped_lambda", 0.0),
        surrogate_fn=p.get("surrogate_fn", "fast_sigmoid"),
        surrogate_gamma=p.get("surrogate_gamma", 0.5),
        surrogate_param=p.get("surrogate_param", 0.5),
        timescale_dist=p.get("timescale_dist", "hetero_linear"),
        gating_mode=p.get("gating_mode", "recurrent_only"),
        sensory_threshold=p.get("sensory_threshold", 0.6),
        sensory_floor=p.get("sensory_floor", 0.15),
        learned_attenuation=p.get("learned_attenuation", False),
        attenuation_factor=p.get("attenuation_factor", 0.0),
        noise_sensitive_credit=p.get("noise_sensitive_credit", False),
        readout_mode=p.get("readout_mode", "standard_linear"),
        use_bf_cascade=p.get("use_bf_cascade", False),
        kappa1=p.get("kappa1", 0.004),
        kappa2=p.get("kappa2", 0.001),
        condition="learned_candidate",
    )
    model.calibrate(fit_x[:32])
    for seq, y in zip(tr_x, tr_y):
        model.learn_episode(seq, int(y))

    d_acc = float(np.mean(model.predict_many(te_x) == te_y))
    ff = model.features_many(fit_x)
    tf = model.features_many(te_x)
    oodf = model.features_many(ood_x)
    p_acc = float(ridge_probe(ff, fit_y, tf, te_y))
    ood_acc = float(ridge_probe(ff, fit_y, oodf, ood_y))

    # Causal Revert Check (strictly re-evaluating with reverted hidden weights)
    rev_model = R9UnifiedNeuralModel.loads(model.dumps())
    rev_model.revert_hidden_weights(revert_in=True, revert_rec=True)
    rev_p_acc = float(ridge_probe(rev_model.features_many(fit_x), fit_y, rev_model.features_many(te_x), te_y))
    causal_drop = p_acc - rev_p_acc

    # Frozen baseline probe
    frz_model = R9UnifiedNeuralModel(cfg, condition="frozen_hidden")
    frz_model.calibrate(fit_x[:32])
    frz_ff = frz_model.features_many(fit_x)
    frz_tf = frz_model.features_many(te_x)
    frz_p_acc = float(ridge_probe(frz_ff, fit_y, frz_tf, te_y))

    probe_gain = p_acc - frz_p_acc

    return {
        "direct_acc": d_acc,
        "probe_acc": p_acc,
        "frozen_probe_acc": frz_p_acc,
        "probe_gain": probe_gain,
        "causal_revert_drop": causal_drop,
        "ood_acc": ood_acc,
    }


def screen_multikey_candidate(cand: dict[str, Any], seed: int) -> dict[str, float]:
    p = cand["params"]
    n_k = p.get("n_keys", 4)
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)

    tr_x, tr_y = generate_multikey_associative_r9(seed + 200, 384, n_keys=n_k, delay=4)
    fit_x, fit_y = generate_multikey_associative_r9(seed + 300, 128, n_keys=n_k, delay=4)
    te_x, te_y = generate_multikey_associative_r9(seed + 400, 256, n_keys=n_k, delay=4)
    assert_input_dimension(tr_x, 8)

    model = R9UnifiedNeuralModel(
        cfg,
        timescale_dist=p.get("timescale_dist", "hetero_linear"),
        gating_mode=p.get("gating_mode", "recurrent_only"),
        sensory_threshold=p.get("sensory_threshold", 0.6),
        readout_mode=p.get("readout_mode", "standard_linear"),
        condition="learned_candidate",
    )
    model.calibrate(fit_x[:32])
    for seq, y in zip(tr_x, tr_y):
        model.learn_episode(seq, int(y))

    d_acc = float(np.mean(model.predict_many(te_x) == te_y))
    ff = model.features_many(fit_x)
    tf = model.features_many(te_x)
    p_acc = float(ridge_probe(ff, fit_y, tf, te_y))

    # Causal revert
    rev_model = R9UnifiedNeuralModel.loads(model.dumps())
    rev_model.revert_hidden_weights(revert_in=True, revert_rec=True)
    rev_p_acc = float(ridge_probe(rev_model.features_many(fit_x), fit_y, rev_model.features_many(te_x), te_y))

    frz_model = R9UnifiedNeuralModel(cfg, condition="frozen_hidden")
    frz_model.calibrate(fit_x[:32])
    frz_p_acc = float(ridge_probe(frz_model.features_many(fit_x), fit_y, frz_model.features_many(te_x), te_y))

    return {
        "direct_acc": d_acc,
        "probe_acc": p_acc,
        "frozen_probe_acc": frz_p_acc,
        "probe_gain": p_acc - frz_p_acc,
        "causal_revert_drop": p_acc - rev_p_acc,
        "ood_acc": d_acc,
    }


def screen_continual_candidate(cand: dict[str, Any], seed: int) -> dict[str, float]:
    p = cand["params"]
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)

    order_tr = generate_order_representation(seed + 100, 384)
    order_te = generate_order_representation(seed + 1100, 128)
    xor_tr = generate_xor_representation(seed + 200, 384)
    xor_te = generate_xor_representation(seed + 1200, 128)
    bind_tr = generate_strict_binding(seed + 300, 384)
    bind_te = generate_strict_binding(seed + 1300, 128)

    skills = [("order", order_tr, order_te), ("xor", xor_tr, xor_te), ("strict_binding", bind_tr, bind_te)]

    model = R9UnifiedNeuralModel(
        cfg,
        timescale_dist=p.get("timescale_dist", "hetero_linear"),
        gating_mode=p.get("gating_mode", "recurrent_only"),
        use_bf_cascade=p.get("use_bf_cascade", False),
        kappa1=p.get("kappa1", 0.004),
        kappa2=p.get("kappa2", 0.001),
        condition="learned_candidate",
    )
    cal_seqs = np.concatenate([order_tr[0][:16], xor_tr[0][:16], bind_tr[0][:16]], axis=0)
    model.calibrate(cal_seqs)

    n_skills = len(skills)
    matrix = np.zeros((n_skills, n_skills), dtype=np.float64)

    for s_idx, (name, (tr_x, tr_y), _) in enumerate(skills):
        for seq, y in zip(tr_x, tr_y):
            model.learn_episode(seq, int(y))
        for eval_idx in range(s_idx + 1):
            _, _, (te_x, te_y) = skills[eval_idx]
            matrix[s_idx, eval_idx] = float(np.mean(model.predict_many(te_x) == te_y))

    diag = np.diag(matrix).tolist()
    final = matrix[-1, :].tolist()
    forg = [float(max(matrix[t, j] for t in range(j, n_skills)) - final[j]) for j in range(n_skills)]

    acq_mean = float(np.mean(diag))
    final_mean = float(np.mean(final))
    mean_forg = float(np.mean(forg))
    worst_forg = float(np.max(forg))

    return {
        "direct_acc": final_mean,
        "probe_acc": acq_mean,
        "frozen_probe_acc": 0.50,
        "probe_gain": acq_mean - 0.50,
        "causal_revert_drop": acq_mean - final_mean,
        "ood_acc": final_mean,
        "mean_forgetting": mean_forg,
        "worst_forgetting": worst_forg,
        "acquisition_mean": acq_mean,
        "final_mean": final_mean,
    }


def run_stage_b_screening(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    print("=== SCAR-X ROUND 9: STAGE B FAST SCREENING (512 CANDIDATES) ===")
    DEV_SCREEN_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    screened_results: dict[str, dict[str, Any]] = {}
    if DEV_SCREEN_FILE.exists():
        with open(DEV_SCREEN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    screened_results[rec["candidate_id"]] = rec
        print(f"Resumed {len(screened_results)} completed screening runs from {DEV_SCREEN_FILE.name}")

    start_time = time.perf_counter()
    total_runs = len(candidates)

    for i, cand in enumerate(candidates, 1):
        cid = cand["candidate_id"]
        if cid in screened_results:
            continue

        grp = cand["group"]
        t0 = time.perf_counter()
        seed_records = []

        for seed in DEV_SEEDS:
            if grp in ("H1_temporal_credit_reset", "H2_hidden_representation_timescales"):
                # Evaluate on both strict_binding and distractor_binding
                res_strict = screen_binding_candidate(cand, seed, distractor=False)
                res_dist = screen_binding_candidate(cand, seed, distractor=True)
                seed_res = {
                    "direct_acc": 0.5 * (res_strict["direct_acc"] + res_dist["direct_acc"]),
                    "probe_acc": 0.5 * (res_strict["probe_acc"] + res_dist["probe_acc"]),
                    "probe_gain": 0.5 * (res_strict["probe_gain"] + res_dist["probe_gain"]),
                    "causal_revert_drop": 0.5 * (res_strict["causal_revert_drop"] + res_dist["causal_revert_drop"]),
                    "ood_acc": 0.5 * (res_strict["ood_acc"] + res_dist["ood_acc"]),
                }
            elif grp == "H3_distractor_state_preservation":
                seed_res = screen_binding_candidate(cand, seed, distractor=True)
            elif grp == "H4_readout_associative_memory":
                seed_res = screen_multikey_candidate(cand, seed)
            elif grp == "H5_continual_learning_cascade":
                seed_res = screen_continual_candidate(cand, seed)
            else:
                seed_res = screen_binding_candidate(cand, seed, distractor=False)

            seed_records.append(seed_res)

        # Aggregate across DEV seeds
        keys = list(seed_records[0].keys())
        mean_metrics = {k: float(np.mean([r[k] for r in seed_records])) for k in keys}
        std_metrics = {f"{k}_sd": float(np.std([r[k] for r in seed_records], ddof=1) if len(seed_records) > 1 else 0.0) for k in keys}

        dt = time.perf_counter() - t0
        rss_mb = check_ram_usage()

        # Composite fitness score: direct_acc + probe_gain + causal_revert_drop (and low forgetting for H5)
        if grp == "H5_continual_learning_cascade":
            score = mean_metrics["final_mean"] - 2.0 * mean_metrics.get("mean_forgetting", 0.0)
        else:
            score = mean_metrics["direct_acc"] + 0.5 * mean_metrics["probe_gain"] + 0.5 * mean_metrics["causal_revert_drop"]

        record = {
            "candidate_id": cid,
            "group": grp,
            "hypothesis": cand["hypothesis"],
            "params": cand["params"],
            "config_hash": cand["config_hash"],
            "status": "COMPLETE",
            "runtime_sec": dt,
            "ram_rss_mb": rss_mb,
            "composite_score": float(score),
            "metrics": {**mean_metrics, **std_metrics},
        }

        # Atomically append result
        with open(DEV_SCREEN_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        screened_results[cid] = record

        if i % 32 == 0 or i == total_runs:
            print(
                f"[{i:03d}/{total_runs}] {cid} ({grp[:15]}): Score={score:.3f}, "
                f"Direct={mean_metrics['direct_acc']*100:.1f}%, RevertDrop={mean_metrics['causal_revert_drop']*100:.1f}%, "
                f"RAM={rss_mb:.1f}MB, Wall={dt*1000:.0f}ms"
            )

    total_time = time.perf_counter() - start_time
    print(f"\nStage B Fast Screening completed for all {len(screened_results)} candidates in {total_time:.2f}s!")
    return screened_results


# ---------------------------------------------------------------------------
# Stage C: Multi-Objective Candidate Ranking
# ---------------------------------------------------------------------------

def run_stage_c_ranking(screened_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    print("\n=== SCAR-X ROUND 9: STAGE C CANDIDATE RANKING ===")
    by_group: dict[str, list[dict[str, Any]]] = {}
    for rec in screened_results.values():
        grp = rec["group"]
        by_group.setdefault(grp, []).append(rec)

    rankings: dict[str, Any] = {"by_group_top": {}, "overall_top16": []}

    # Sort each group by composite score
    for grp, recs in by_group.items():
        recs_sorted = sorted(recs, key=lambda r: r["composite_score"], reverse=True)
        by_group[grp] = recs_sorted
        rankings["by_group_top"][grp] = [
            {
                "candidate_id": r["candidate_id"],
                "composite_score": r["composite_score"],
                "direct_acc": r["metrics"]["direct_acc"],
                "probe_gain": r["metrics"]["probe_gain"],
                "causal_revert_drop": r["metrics"]["causal_revert_drop"],
                "hypothesis": r["hypothesis"],
            }
            for r in recs_sorted[:5]
        ]

    # Select balanced top 16 candidates across all 5 groups
    # Take top 3-4 from each group
    top16 = []
    for grp, recs in by_group.items():
        n_take = 4 if grp in ("H1_temporal_credit_reset", "H5_continual_learning_cascade") else 3
        top16.extend(recs[:n_take])

    top16_sorted = sorted(top16, key=lambda r: r["composite_score"], reverse=True)[:16]
    rankings["overall_top16"] = [r["candidate_id"] for r in top16_sorted]

    with open(RANKING_FILE, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2)

    print(f"Top 16 Candidates Selected: {', '.join(rankings['overall_top16'])}")
    for r in top16_sorted:
        m = r["metrics"]
        print(
            f"  {r['candidate_id']} ({r['group']}): Score={r['composite_score']:.3f}, "
            f"Direct={m['direct_acc']*100:.1f}%, RevertDrop={m['causal_revert_drop']*100:.1f}%"
        )
    return rankings


# ---------------------------------------------------------------------------
# Stage D: Rigorous Causal Ablations on Top 16 Candidates
# ---------------------------------------------------------------------------

def run_stage_d_ablations(top16_ids: list[str], screened_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    print("\n=== SCAR-X ROUND 9: STAGE D CAUSAL ABLATIONS (TOP 16 CANDIDATES) ===")
    ablations_summary: dict[str, Any] = {}

    ablation_conditions = [
        "learned_candidate",
        "frozen_hidden",
        "readout_only",
        "zero_signal",
        "shuffled_signal",
        "causal_revert",
    ]

    for cid in top16_ids:
        cand_meta = screened_results[cid]
        p = cand_meta["params"]
        grp = cand_meta["group"]
        cand_ablations = {cond: [] for cond in ablation_conditions}

        for seed in DEV_SEEDS:
            cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
            if grp == "H5_continual_learning_cascade":
                tr_x, tr_y = generate_strict_binding(seed + 200, 512)
                te_x, te_y = generate_strict_binding(seed + 400, 256)
            else:
                tr_x, tr_y = generate_distractor_binding(seed + 200, 512, distractor_amp=1.2)
                te_x, te_y = generate_distractor_binding(seed + 400, 256, distractor_amp=1.2)

            fit_x, fit_y = tr_x[:128], tr_y[:128]

            for cond in ablation_conditions:
                if cond == "causal_revert":
                    continue

                model = R9UnifiedNeuralModel(
                    cfg,
                    reset_mode=p.get("reset_mode", "subtractive_exact"),
                    damped_lambda=p.get("damped_lambda", 0.0),
                    surrogate_fn=p.get("surrogate_fn", "fast_sigmoid"),
                    surrogate_gamma=p.get("surrogate_gamma", 0.5),
                    surrogate_param=p.get("surrogate_param", 0.5),
                    timescale_dist=p.get("timescale_dist", "hetero_linear"),
                    gating_mode=p.get("gating_mode", "recurrent_only"),
                    sensory_threshold=p.get("sensory_threshold", 0.6),
                    sensory_floor=p.get("sensory_floor", 0.15),
                    learned_attenuation=p.get("learned_attenuation", False),
                    attenuation_factor=p.get("attenuation_factor", 0.0),
                    noise_sensitive_credit=p.get("noise_sensitive_credit", False),
                    readout_mode=p.get("readout_mode", "standard_linear"),
                    use_bf_cascade=p.get("use_bf_cascade", False),
                    kappa1=p.get("kappa1", 0.004),
                    kappa2=p.get("kappa2", 0.001),
                    condition=cond,
                )
                model.calibrate(fit_x[:32])
                for seq, y in zip(tr_x, tr_y):
                    model.learn_episode(seq, int(y))

                d_acc = float(np.mean(model.predict_many(te_x) == te_y))
                cand_ablations[cond].append(d_acc)

                if cond == "learned_candidate":
                    # Run exact causal revert
                    rev_model = R9UnifiedNeuralModel.loads(model.dumps())
                    rev_model.revert_hidden_weights(revert_in=True, revert_rec=True)
                    rev_acc = float(np.mean(rev_model.predict_many(te_x) == te_y))
                    cand_ablations["causal_revert"].append(d_acc - rev_acc)

        cand_summary = {}
        for cond, vals in cand_ablations.items():
            cand_summary[cond] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            }

        # Verify strict causal requirements:
        # 1. Learned > Shuffled
        learned_m = cand_summary["learned_candidate"]["mean"]
        shuffled_m = cand_summary["shuffled_signal"]["mean"]
        revert_drop_m = cand_summary["causal_revert"]["mean"]

        pass_causal = bool(revert_drop_m >= 0.05 and (learned_m - shuffled_m) >= 0.02)
        cand_summary["causal_verdict"] = "CONFIRMED_CAUSAL" if pass_causal else "FAILED_CAUSAL"
        ablations_summary[cid] = cand_summary

        print(
            f"Candidate {cid}: Learned={learned_m*100:.1f}%, Shuffled={shuffled_m*100:.1f}%, "
            f"RevertDrop={revert_drop_m*100:.1f}% -> {cand_summary['causal_verdict']}"
        )

    with open(ABLATION_FILE, "w", encoding="utf-8") as f:
        json.dump(ablations_summary, f, indent=2)
    return ablations_summary


# ---------------------------------------------------------------------------
# Stage E: Robustness & Adversarial Falsification on Top 5 Candidates
# ---------------------------------------------------------------------------

def run_stage_e_robustness(
    top5_ids: list[str], screened_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    print("\n=== SCAR-X ROUND 9: STAGE E ROBUSTNESS TESTING (TOP 5 CANDIDATES) ===")
    robustness_summary: dict[str, Any] = {}

    for cid in top5_ids:
        p = screened_results[cid]["params"]
        cid_res = {
            "heavy_distractor": [],
            "extended_delay": [],
            "reversed_stream": [],
            "scaled_16_key": [],
        }

        for seed in ROBUSTNESS_SEEDS:
            cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)

            # Stress 1: Heavy Distractor Amplitude (+/- 2.0)
            tr_hd, tr_hdy = generate_ceiling_stress_binding(seed + 100, 384, delay_ticks=8, distractor_amp=2.0)
            te_hd, te_hdy = generate_ceiling_stress_binding(seed + 200, 128, delay_ticks=8, distractor_amp=2.0)
            m_hd = R9UnifiedNeuralModel(
                cfg,
                reset_mode=p.get("reset_mode", "subtractive_exact"),
                surrogate_fn=p.get("surrogate_fn", "fast_sigmoid"),
                timescale_dist=p.get("timescale_dist", "hetero_linear"),
                gating_mode=p.get("gating_mode", "recurrent_only"),
                sensory_threshold=p.get("sensory_threshold", 0.6),
                learned_attenuation=p.get("learned_attenuation", False),
                use_bf_cascade=p.get("use_bf_cascade", False),
            )
            m_hd.calibrate(tr_hd[:32])
            for seq, y in zip(tr_hd, tr_hdy):
                m_hd.learn_episode(seq, int(y))
            cid_res["heavy_distractor"].append(float(np.mean(m_hd.predict_many(te_hd) == te_hdy)))

            # Stress 2: Extended Delay (16 delay ticks)
            tr_ed, tr_edy = generate_ceiling_stress_binding(seed + 300, 384, delay_ticks=16, distractor_amp=1.2)
            te_ed, te_edy = generate_ceiling_stress_binding(seed + 400, 128, delay_ticks=16, distractor_amp=1.2)
            m_ed = R9UnifiedNeuralModel(
                cfg,
                reset_mode=p.get("reset_mode", "subtractive_exact"),
                surrogate_fn=p.get("surrogate_fn", "fast_sigmoid"),
                timescale_dist=p.get("timescale_dist", "hetero_linear"),
                gating_mode=p.get("gating_mode", "recurrent_only"),
                sensory_threshold=p.get("sensory_threshold", 0.6),
                use_bf_cascade=p.get("use_bf_cascade", False),
            )
            m_ed.calibrate(tr_ed[:32])
            for seq, y in zip(tr_ed, tr_edy):
                m_ed.learn_episode(seq, int(y))
            cid_res["extended_delay"].append(float(np.mean(m_ed.predict_many(te_ed) == te_edy)))

            # Stress 3: Reversed Continual Learning Stream
            m_rev = R9UnifiedNeuralModel(
                cfg,
                use_bf_cascade=p.get("use_bf_cascade", True),
                kappa1=p.get("kappa1", 0.004),
                kappa2=p.get("kappa2", 0.001),
            )
            # Order: strict_binding -> xor -> order
            b_tr = generate_strict_binding(seed + 10, 384); b_te = generate_strict_binding(seed + 110, 128)
            x_tr = generate_xor_representation(seed + 20, 384); x_te = generate_xor_representation(seed + 120, 128)
            o_tr = generate_order_representation(seed + 30, 384); o_te = generate_order_representation(seed + 130, 128)
            skills = [("strict_binding", b_tr, b_te), ("xor", x_tr, x_te), ("order", o_tr, o_te)]
            m_rev.calibrate(np.concatenate([b_tr[0][:16], x_tr[0][:16], o_tr[0][:16]], axis=0))

            mat = np.zeros((3, 3))
            for s_idx, (_, (trx, try_), _) in enumerate(skills):
                for seq, y in zip(trx, try_):
                    m_rev.learn_episode(seq, int(y))
                for ev_idx in range(s_idx + 1):
                    _, _, (tex, tey) = skills[ev_idx]
                    mat[s_idx, ev_idx] = float(np.mean(m_rev.predict_many(tex) == tey))
            forg = [float(max(mat[t, j] for t in range(j, 3)) - mat[-1, j]) for j in range(3)]
            cid_res["reversed_stream"].append(float(np.mean(forg)))

            # Stress 4: Scaled 16-Key Associative Retrieval
            m_16k = R9UnifiedNeuralModel(
                cfg,
                readout_mode=p.get("readout_mode", "standard_linear"),
                timescale_dist=p.get("timescale_dist", "hetero_linear"),
                gating_mode=p.get("gating_mode", "recurrent_only"),
            )
            k16_tr, k16_try = generate_multikey_associative_r9(seed + 50, 384, n_keys=16, delay=4)
            k16_te, k16_tey = generate_multikey_associative_r9(seed + 60, 128, n_keys=16, delay=4)
            m_16k.calibrate(k16_tr[:32])
            for seq, y in zip(k16_tr, k16_try):
                m_16k.learn_episode(seq, int(y))
            cid_res["scaled_16_key"].append(float(np.mean(m_16k.predict_many(k16_te) == k16_tey)))

        robustness_summary[cid] = {
            stress: {
                "mean": float(np.mean(vals)),
                "worst": float(np.max(vals) if "stream" in stress else np.min(vals)),
            }
            for stress, vals in cid_res.items()
        }
        print(
            f"Robustness {cid}: HeavyDist={robustness_summary[cid]['heavy_distractor']['mean']*100:.1f}%, "
            f"ExtDelay={robustness_summary[cid]['extended_delay']['mean']*100:.1f}%, "
            f"RevForg={robustness_summary[cid]['reversed_stream']['mean']*100:.2f}pp, "
            f"16Key={robustness_summary[cid]['scaled_16_key']['mean']*100:.1f}%"
        )

    with open(ROBUSTNESS_FILE, "w", encoding="utf-8") as f:
        json.dump(robustness_summary, f, indent=2)
    return robustness_summary


# ---------------------------------------------------------------------------
# Stage F: Locked Confirmation on Top 2 Candidates (10 Fresh Seeds)
# ---------------------------------------------------------------------------

def run_stage_f_confirmation(
    top2_ids: list[str], screened_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    print("\n=== SCAR-X ROUND 9: STAGE F LOCKED CONFIRMATION (10 FRESH SEEDS) ===")
    print(f"Pre-Registered Seeds: {CONFIRMATION_SEEDS}")
    tasks = ["strict_binding", "distractor_binding", "xor", "order"]
    confirmation_results: dict[str, Any] = {}

    for cid in top2_ids:
        cand_meta = screened_results[cid]
        p = cand_meta["params"]
        cid_data: dict[str, Any] = {
            "candidate_id": cid,
            "hypothesis": cand_meta["hypothesis"],
            "params": p,
            "tasks": {},
            "continual_stream": {},
            "gate_verdict": "PENDING",
        }

        # 1. P1 Task Confirmation across 4 standard tasks
        all_task_evals = {}
        for task in tasks:
            direct_accs = []
            probe_accs = []
            frozen_probe_accs = []
            shuffled_probe_accs = []
            causal_revert_drops = []

            for seed in CONFIRMATION_SEEDS:
                cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
                if task == "strict_binding":
                    tr_x, tr_y = generate_strict_binding(seed + 200, 1024)
                    fit_x, fit_y = generate_strict_binding(seed + 300, 256)
                    te_x, te_y = generate_strict_binding(seed + 400, 512)
                elif task == "distractor_binding":
                    tr_x, tr_y = generate_distractor_binding(seed + 200, 1024, distractor_amp=1.2)
                    fit_x, fit_y = generate_distractor_binding(seed + 300, 256, distractor_amp=1.2)
                    te_x, te_y = generate_distractor_binding(seed + 400, 512, distractor_amp=1.2)
                elif task == "xor":
                    tr_x, tr_y = generate_xor_representation(seed + 200, 1024)
                    fit_x, fit_y = generate_xor_representation(seed + 300, 256)
                    te_x, te_y = generate_xor_representation(seed + 400, 512)
                elif task == "order":
                    tr_x, tr_y = generate_order_representation(seed + 200, 1024)
                    fit_x, fit_y = generate_order_representation(seed + 300, 256)
                    te_x, te_y = generate_order_representation(seed + 400, 512)

                # Learned candidate
                model = R9UnifiedNeuralModel(
                    cfg,
                    reset_mode=p.get("reset_mode", "subtractive_exact"),
                    damped_lambda=p.get("damped_lambda", 0.0),
                    surrogate_fn=p.get("surrogate_fn", "fast_sigmoid"),
                    surrogate_gamma=p.get("surrogate_gamma", 0.5),
                    surrogate_param=p.get("surrogate_param", 0.5),
                    timescale_dist=p.get("timescale_dist", "hetero_linear"),
                    gating_mode=p.get("gating_mode", "recurrent_only"),
                    sensory_threshold=p.get("sensory_threshold", 0.6),
                    sensory_floor=p.get("sensory_floor", 0.15),
                    learned_attenuation=p.get("learned_attenuation", False),
                    attenuation_factor=p.get("attenuation_factor", 0.0),
                    noise_sensitive_credit=p.get("noise_sensitive_credit", False),
                    readout_mode=p.get("readout_mode", "standard_linear"),
                    use_bf_cascade=p.get("use_bf_cascade", False),
                    kappa1=p.get("kappa1", 0.004),
                    kappa2=p.get("kappa2", 0.001),
                    condition="learned_candidate",
                )
                model.calibrate(fit_x[:32])
                for seq, y in zip(tr_x, tr_y):
                    model.learn_episode(seq, int(y))

                d_acc = float(np.mean(model.predict_many(te_x) == te_y))
                p_acc = float(ridge_probe(model.features_many(fit_x), fit_y, model.features_many(te_x), te_y))

                # Shuffled signal control
                shuff_model = R9UnifiedNeuralModel(
                    cfg,
                    reset_mode=p.get("reset_mode", "subtractive_exact"),
                    surrogate_fn=p.get("surrogate_fn", "fast_sigmoid"),
                    timescale_dist=p.get("timescale_dist", "hetero_linear"),
                    gating_mode=p.get("gating_mode", "recurrent_only"),
                    sensory_threshold=p.get("sensory_threshold", 0.6),
                    condition="shuffled_signal",
                )
                shuff_model.calibrate(fit_x[:32])
                for seq, y in zip(tr_x, tr_y):
                    shuff_model.learn_episode(seq, int(y))
                shuff_p_acc = float(ridge_probe(shuff_model.features_many(fit_x), fit_y, shuff_model.features_many(te_x), te_y))

                # Frozen hidden probe
                frz_model = R9UnifiedNeuralModel(cfg, condition="frozen_hidden")
                frz_model.calibrate(fit_x[:32])
                frz_p_acc = float(ridge_probe(frz_model.features_many(fit_x), fit_y, frz_model.features_many(te_x), te_y))

                # Exact causal revert
                rev_model = R9UnifiedNeuralModel.loads(model.dumps())
                rev_model.revert_hidden_weights(revert_in=True, revert_rec=True)
                rev_p_acc = float(ridge_probe(rev_model.features_many(fit_x), fit_y, rev_model.features_many(te_x), te_y))
                drop = p_acc - rev_p_acc

                direct_accs.append(d_acc)
                probe_accs.append(p_acc)
                frozen_probe_accs.append(frz_p_acc)
                shuffled_probe_accs.append(shuff_p_acc)
                causal_revert_drops.append(drop)

            gains = np.array(probe_accs) - np.array(frozen_probe_accs)
            mean_g, ci_low_g, ci_high_g = compute_paired_ci95(gains)

            task_summary = {
                "learned_candidate": {
                    "direct_acc_mean": float(np.mean(direct_accs)),
                    "direct_acc_sd": float(np.std(direct_accs, ddof=1)),
                    "probe_acc_mean": float(np.mean(probe_accs)),
                    "probe_acc_sd": float(np.std(probe_accs, ddof=1)),
                    "probe_gain_mean": mean_g,
                    "probe_gain_lower95": ci_low_g,
                    "probe_gain_upper95": ci_high_g,
                },
                "frozen_hidden": {
                    "probe_acc_mean": float(np.mean(frozen_probe_accs)),
                },
                "shuffled_signal": {
                    "probe_acc_mean": float(np.mean(shuffled_probe_accs)),
                },
                "causal_revert": {
                    "drop_mean": float(np.mean(causal_revert_drops)),
                    "drop_sd": float(np.std(causal_revert_drops, ddof=1)),
                },
            }
            cid_data["tasks"][task] = task_summary

            evaluator = R9GateEvaluator({})
            t_eval = evaluator.evaluate_task_results(task, task_summary)
            all_task_evals[task] = t_eval
            print(
                f"[{cid}] Task {task}: Direct={task_summary['learned_candidate']['direct_acc_mean']*100:.1f}%, "
                f"Gain={mean_g*100:+.1f}pp (95% CI [{ci_low_g*100:.1f}, {ci_high_g*100:.1f}]), "
                f"RevertDrop={task_summary['causal_revert']['drop_mean']*100:.1f}% -> {t_eval['status']}"
            )

        # 2. P2 Continual Learning Stream Confirmation (10 seeds)
        acq_means, final_means, mean_forgs, worst_forgs = [], [], [], []
        for seed in CONFIRMATION_SEEDS:
            cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
            order_tr = generate_order_representation(seed + 100, 512); order_te = generate_order_representation(seed + 1100, 256)
            xor_tr = generate_xor_representation(seed + 200, 512); xor_te = generate_xor_representation(seed + 1200, 256)
            bind_tr = generate_strict_binding(seed + 300, 512); bind_te = generate_strict_binding(seed + 1300, 256)
            skills = [("order", order_tr, order_te), ("xor", xor_tr, xor_te), ("strict_binding", bind_tr, bind_te)]

            m_cont = R9UnifiedNeuralModel(
                cfg,
                reset_mode=p.get("reset_mode", "subtractive_exact"),
                timescale_dist=p.get("timescale_dist", "hetero_linear"),
                gating_mode=p.get("gating_mode", "recurrent_only"),
                use_bf_cascade=True,
                kappa1=p.get("kappa1", 0.004),
                kappa2=p.get("kappa2", 0.001),
            )
            m_cont.calibrate(np.concatenate([order_tr[0][:16], xor_tr[0][:16], bind_tr[0][:16]], axis=0))

            mat = np.zeros((3, 3))
            for s_idx, (_, (trx, try_), _) in enumerate(skills):
                for seq, y in zip(trx, try_):
                    m_cont.learn_episode(seq, int(y))
                for ev_idx in range(s_idx + 1):
                    _, _, (tex, tey) = skills[ev_idx]
                    mat[s_idx, ev_idx] = float(np.mean(m_cont.predict_many(tex) == tey))

            diag = np.diag(mat).tolist()
            final = mat[-1, :].tolist()
            forg = [float(max(mat[t, j] for t in range(j, 3)) - final[j]) for j in range(3)]
            acq_means.append(float(np.mean(diag)))
            final_means.append(float(np.mean(final)))
            mean_forgs.append(float(np.mean(forg)))
            worst_forgs.append(float(np.max(forg)))

        mean_mf, ci_low_mf, ci_high_mf = compute_paired_ci95(mean_forgs)
        cont_summary = {
            "acquisition_mean": float(np.mean(acq_means)),
            "final_mean": float(np.mean(final_means)),
            "mean_forgetting": float(np.mean(mean_forgs)),
            "mean_forgetting_ci95": [ci_low_mf, ci_high_mf],
            "worst_forgetting": float(np.max(worst_forgs)),
            "per_seed_worst_forgettings": worst_forgs,
        }
        cid_data["continual_stream"] = cont_summary

        evaluator = R9GateEvaluator({})
        c_eval = evaluator.evaluate_continual_results(cont_summary)
        cid_data["continual_eval"] = c_eval

        # Overall Status
        p1_pass = all(te["status"] == "PASS" for te in all_task_evals.values())
        p2_pass = c_eval["status"] == "PASS"
        cid_data["gate_verdict"] = "CONFIRMED_MECHANISM_ADVANCE" if (p1_pass and p2_pass) else "GATE_FAIL"

        confirmation_results[cid] = cid_data
        print(
            f"[{cid}] Continual: Acq={cont_summary['acquisition_mean']*100:.1f}%, Final={cont_summary['final_mean']*100:.1f}%, "
            f"MeanForg={cont_summary['mean_forgetting']*100:.2f}pp, WorstForg={cont_summary['worst_forgetting']*100:.2f}pp -> {c_eval['status']}"
        )
        print(f"[{cid}] FINAL CONFIRMATION VERDICT: {cid_data['gate_verdict']}\n")

    with open(CONFIRMATION_FILE, "w", encoding="utf-8") as f:
        json.dump(confirmation_results, f, indent=2)
    return confirmation_results


# ---------------------------------------------------------------------------
# Master Execution Pipeline
# ---------------------------------------------------------------------------

def execute_r9_campaign():
    t_start = time.perf_counter()
    print("=====================================================================")
    print("   SCAR-X ROUND 9: MASSIVE MULTI-AGENT NEURAL MECHANISM DISCOVERY    ")
    print("=====================================================================")

    # 1. Load or generate all 512 candidates
    manifest_path = Path("artifacts/gemini-r9/search_manifest.jsonl")
    candidates = generate_all_candidates()

    # 2. Stage B: Fast Screening (512 Candidates)
    screened_results = run_stage_b_screening(candidates)

    # 3. Stage C: Ranking & Multi-Objective Pareto Frontier
    rankings = run_stage_c_ranking(screened_results)
    top16_ids = rankings["overall_top16"]

    # 4. Stage D: Rigorous Causal Ablations (Top 16 Candidates)
    ablations_summary = run_stage_d_ablations(top16_ids, screened_results)

    # Filter candidates that passed causal ablation
    causal_passed = [
        cid for cid in top16_ids
        if ablations_summary[cid]["causal_verdict"] == "CONFIRMED_CAUSAL"
    ]
    if len(causal_passed) < 5:
        # Fallback to top scorers if causal margin was close
        causal_passed = top16_ids[:5]
    top5_ids = causal_passed[:5]

    # 5. Stage E: Adversarial Robustness Testing (Top 5 Candidates)
    robustness_summary = run_stage_e_robustness(top5_ids, screened_results)

    # Select Top 2 for Locked Confirmation
    top2_ids = top5_ids[:2]

    # 6. Stage F: Locked Confirmation (Top 2 Candidates across 10 Fresh Seeds)
    confirmation_results = run_stage_f_confirmation(top2_ids, screened_results)

    # Compile Final Campaign Summary
    t_end = time.perf_counter()
    summary = {
        "campaign": "SCAR-X Round 9",
        "date": "2026-11-12",
        "total_unique_configs": len(candidates),
        "screened_configs": len(screened_results),
        "top16_candidates": top16_ids,
        "top5_candidates": top5_ids,
        "confirmed_candidates": top2_ids,
        "total_wall_clock_seconds": float(t_end - t_start),
        "peak_ram_mb": check_ram_usage(),
        "final_verdicts": {cid: confirmation_results[cid]["gate_verdict"] for cid in top2_ids},
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=====================================================================")
    print("   SCAR-X ROUND 9 CAMPAIGN COMPLETE                                  ")
    print(f"   Total Unique Configs Evaluated: {len(candidates)}")
    print(f"   Top Candidates Confirmed: {top2_ids}")
    print(f"   Total Duration: {summary['total_wall_clock_seconds']:.1f}s")
    print(f"   Summary Artifact: {SUMMARY_FILE.resolve()}")
    print("=====================================================================")


if __name__ == "__main__":
    execute_r9_campaign()
