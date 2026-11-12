"""SCAR-X Round 9: Deterministic 512-Candidate Configuration Generator.

Pre-registers and hashes 512 unique configurations across 5 scientific hypothesis groups:
- Group H1: Temporal Credit Assignment & Subtractive Spike Reset (160 configs)
- Group H2: Factorial Timescale Distributions & Sensory Gating (96 configs)
- Group H3: Distractor Robustness & State Preservation (96 configs)
- Group H4: Readout & Associative Memory Retrieval (80 configs)
- Group H5: Task-Agnostic Continual Learning & BF-Cascade (80 configs)

Total initial search budget: exactly 512 candidates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_config_hash(group: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"group": group, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def generate_all_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    # =======================================================================
    # Group H1: Temporal Credit Assignment & Reset (160 configs)
    # =======================================================================
    h1_reset_modes = [
        ("subtractive_exact", 0.0),
        ("hard_reset", 0.0),
        ("damped_subtractive", 0.2),
        ("damped_subtractive", 0.5),
        ("damped_subtractive", 0.8),
    ]
    h1_surrogates = [
        ("fast_sigmoid", 0.3, 0.5),
        ("fast_sigmoid", 0.5, 0.5),
        ("fast_sigmoid", 0.8, 0.5),
        ("exponential", 0.5, 0.3),
        ("exponential", 0.5, 0.6),
        ("boxcar", 0.5, 0.4),
        ("gaussian", 0.5, 0.4),
        ("gaussian", 0.8, 0.6),
    ]
    h1_timescales = ["hetero_linear", "uniform_mid", "hetero_log", "uniform_slow"]

    idx = 1
    for r_mode, d_lam in h1_reset_modes:
        for s_fn, s_gam, s_par in h1_surrogates:
            for t_dist in h1_timescales:
                p = {
                    "reset_mode": r_mode,
                    "damped_lambda": d_lam,
                    "surrogate_fn": s_fn,
                    "surrogate_gamma": s_gam,
                    "surrogate_param": s_par,
                    "timescale_dist": t_dist,
                    "gating_mode": "recurrent_only",
                    "sensory_threshold": 0.6,
                    "sensory_floor": 0.15,
                    "learned_attenuation": False,
                    "attenuation_factor": 0.0,
                    "noise_sensitive_credit": False,
                    "readout_mode": "standard_linear",
                    "use_bf_cascade": False,
                    "kappa1": 0.004,
                    "kappa2": 0.001,
                }
                cid = f"H1-{idx:03d}"
                c_hash = compute_config_hash("H1", p)
                candidates.append({
                    "candidate_id": cid,
                    "group": "H1_temporal_credit_reset",
                    "hypothesis": f"Eligibility under {r_mode} (lambda={d_lam}) with {s_fn} surrogate and {t_dist} timescale",
                    "params": p,
                    "config_hash": c_hash,
                    "matched_baseline": "H1-033",  # hard_reset baseline
                    "expected_causal_signature": "subtractive reset preserves subthreshold credit traces after spikes",
                })
                idx += 1
                if idx > 160:
                    break
            if idx > 160:
                break
        if idx > 160:
            break

    # =======================================================================
    # Group H2: Factorial Multi-Timescale & Gating (96 configs)
    # 6 timescale distributions x 4 gating modes x 4 sensory thresholds = 96
    # =======================================================================
    h2_timescales = [
        "uniform_fast",
        "uniform_mid",
        "uniform_slow",
        "hetero_linear",
        "hetero_log",
        "hetero_bimodal",
    ]
    h2_gatings = ["none", "hard_sensory", "soft_sigmoid", "recurrent_only"]
    h2_thresholds = [0.4, 0.6, 0.8, 1.0]

    idx = 1
    for t_dist in h2_timescales:
        for g_mode in h2_gatings:
            for s_thresh in h2_thresholds:
                p = {
                    "reset_mode": "subtractive_exact",
                    "damped_lambda": 0.0,
                    "surrogate_fn": "fast_sigmoid",
                    "surrogate_gamma": 0.5,
                    "surrogate_param": 0.5,
                    "timescale_dist": t_dist,
                    "gating_mode": g_mode,
                    "sensory_threshold": s_thresh,
                    "sensory_floor": 0.15,
                    "learned_attenuation": False,
                    "attenuation_factor": 0.0,
                    "noise_sensitive_credit": False,
                    "readout_mode": "standard_linear",
                    "use_bf_cascade": False,
                    "kappa1": 0.004,
                    "kappa2": 0.001,
                }
                cid = f"H2-{idx:03d}"
                c_hash = compute_config_hash("H2", p)
                candidates.append({
                    "candidate_id": cid,
                    "group": "H2_hidden_representation_timescales",
                    "hypothesis": f"Factorial interaction of {t_dist} timescale decay with {g_mode} sensory gating (thresh={s_thresh})",
                    "params": p,
                    "config_hash": c_hash,
                    "matched_baseline": "H2-001",  # uniform_fast ungated
                    "expected_causal_signature": "heterogeneous timescales prevent premature decay; gating protects recurrent memory during delay",
                })
                idx += 1

    # =======================================================================
    # Group H3: Distractor Robustness & State Preservation (96 configs)
    # =======================================================================
    h3_attenuations = [
        (False, 0.0),
        (True, 0.2),
        (True, 0.4),
        (True, 0.6),
        (True, 0.8),
        (True, 0.9),
    ]
    h3_noise_credits = [False, True]
    h3_resets = ["subtractive_exact", "hard_reset"]
    h3_floors = [0.05, 0.15, 0.25, 0.40]

    idx = 1
    for atten_en, atten_fac in h3_attenuations:
        for n_cred in h3_noise_credits:
            for r_mode in h3_resets:
                for s_floor in h3_floors:
                    p = {
                        "reset_mode": r_mode,
                        "damped_lambda": 0.0,
                        "surrogate_fn": "fast_sigmoid",
                        "surrogate_gamma": 0.5,
                        "surrogate_param": 0.5,
                        "timescale_dist": "hetero_linear",
                        "gating_mode": "recurrent_only",
                        "sensory_threshold": 0.6,
                        "sensory_floor": s_floor,
                        "learned_attenuation": atten_en,
                        "attenuation_factor": atten_fac,
                        "noise_sensitive_credit": n_cred,
                        "readout_mode": "standard_linear",
                        "use_bf_cascade": False,
                        "kappa1": 0.004,
                        "kappa2": 0.001,
                    }
                    cid = f"H3-{idx:03d}"
                    c_hash = compute_config_hash("H3", p)
                    candidates.append({
                        "candidate_id": cid,
                        "group": "H3_distractor_state_preservation",
                        "hypothesis": f"Sensory attenuation={atten_fac} (noise_credit={n_cred}, floor={s_floor}) under {r_mode}",
                        "params": p,
                        "config_hash": c_hash,
                        "matched_baseline": "H3-001",  # No attenuation baseline
                        "expected_causal_signature": "sensory attenuation prevents distractor spikes from corrupting recurrent working memory",
                    })
                    idx += 1

    # =======================================================================
    # Group H4: Readout & Associative Memory (80 configs)
    # =======================================================================
    h4_readouts = ["standard_linear", "context_gated"]
    h4_n_keys = [2, 4, 8, 16]
    h4_timescales = ["hetero_linear", "uniform_slow"]
    h4_gatings = ["recurrent_only", "none"]
    h4_thresholds = [0.4, 0.6, 0.8]

    idx = 1
    for r_mode in h4_readouts:
        for n_k in h4_n_keys:
            for t_dist in h4_timescales:
                for g_mode in h4_gatings:
                    for s_th in h4_thresholds:
                        p = {
                            "reset_mode": "subtractive_exact",
                            "damped_lambda": 0.0,
                            "surrogate_fn": "fast_sigmoid",
                            "surrogate_gamma": 0.5,
                            "surrogate_param": 0.5,
                            "timescale_dist": t_dist,
                            "gating_mode": g_mode,
                            "sensory_threshold": s_th,
                            "sensory_floor": 0.15,
                            "learned_attenuation": False,
                            "attenuation_factor": 0.0,
                            "noise_sensitive_credit": False,
                            "readout_mode": r_mode,
                            "n_keys": n_k,
                            "use_bf_cascade": False,
                            "kappa1": 0.004,
                            "kappa2": 0.001,
                        }
                        cid = f"H4-{idx:03d}"
                        c_hash = compute_config_hash("H4", p)
                        candidates.append({
                            "candidate_id": cid,
                            "group": "H4_readout_associative_memory",
                            "hypothesis": f"Associative retrieval for {n_k} keys with {r_mode} readout ({t_dist}, {g_mode})",
                            "params": p,
                            "config_hash": c_hash,
                            "matched_baseline": "H4-001",  # 2-key linear baseline
                            "expected_causal_signature": "context-conditioned readout resolves key-value interference without growing hidden state",
                        })
                        idx += 1
                        if idx > 80:
                            break
                    if idx > 80:
                        break
                if idx > 80:
                    break
            if idx > 80:
                break
        if idx > 80:
            break

    # =======================================================================
    # Group H5: Continual Learning & Cascade (80 configs)
    # =======================================================================
    h5_cascades = [(False, 0.0, 0.0)]
    for k1 in [0.001, 0.002, 0.004, 0.008, 0.012]:
        for k2 in [0.0002, 0.0005, 0.001, 0.002]:
            h5_cascades.append((True, k1, k2))

    h5_timescales = ["hetero_linear", "uniform_slow"]
    h5_gatings = ["recurrent_only", "none"]

    idx = 1
    for use_c, k1, k2 in h5_cascades:
        for t_dist in h5_timescales:
            for g_mode in h5_gatings:
                p = {
                    "reset_mode": "subtractive_exact",
                    "damped_lambda": 0.0,
                    "surrogate_fn": "fast_sigmoid",
                    "surrogate_gamma": 0.5,
                    "surrogate_param": 0.5,
                    "timescale_dist": t_dist,
                    "gating_mode": g_mode,
                    "sensory_threshold": 0.6,
                    "sensory_floor": 0.15,
                    "learned_attenuation": False,
                    "attenuation_factor": 0.0,
                    "noise_sensitive_credit": False,
                    "readout_mode": "standard_linear",
                    "use_bf_cascade": use_c,
                    "kappa1": k1,
                    "kappa2": k2,
                }
                cid = f"H5-{idx:03d}"
                c_hash = compute_config_hash("H5", p)
                candidates.append({
                    "candidate_id": cid,
                    "group": "H5_continual_learning_cascade",
                    "hypothesis": f"Continual learning with cascade={use_c} (k1={k1}, k2={k2}) across {t_dist}",
                    "params": p,
                    "config_hash": c_hash,
                    "matched_baseline": "H5-001",  # Unconsolidated baseline
                    "expected_causal_signature": "two-compartment diffusion consolidates learned weights passively into u2 reservoir",
                })
                idx += 1
                if idx > 80:
                    break
            if idx > 80:
                break
        if idx > 80:
            break

    return candidates


def save_manifest(out_path: Path = Path("artifacts/gemini-r9/search_manifest.jsonl")) -> list[dict[str, Any]]:
    candidates = generate_all_candidates()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"Generated and pre-registered {len(candidates)} candidates in {out_path.resolve()}")
    return candidates


if __name__ == "__main__":
    cands = save_manifest()
    by_grp: dict[str, int] = {}
    for c in cands:
        g = c["group"]
        by_grp[g] = by_grp.get(g, 0) + 1
    print("\nCandidate counts by group:")
    for g, count in by_grp.items():
        print(f"  {g}: {count}")
    print(f"Total: {sum(by_grp.values())}")
