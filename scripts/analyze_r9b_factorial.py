"""SCAR-X Round 9b Factorial DEV Analysis Script.

Computes main effects, interaction effects, matched control differences,
and causal revert drops from artifacts/r9b_factorial_dev_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "artifacts" / "r9b_factorial_dev_results.json"


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    tasks = ["strict_binding", "xor", "distractor_binding", "order"]

    print("=" * 80)
    print("SCAR-X ROUND 9b FACTORIAL ANALYSIS REPORT")
    print("=" * 80)

    # 1. Main Effect: Factor A (Eligibility Decay)
    print("\n--- 1. FACTOR A: ELIGIBILITY DECAY (Mean Direct Accuracy %) ---")
    decays = ["uniform_fast", "uniform_mid", "uniform_slow", "heterogeneous"]
    print(f"{'Decay':20s} | {'Binding':10s} | {'XOR':10s} | {'Distractor':10s} | {'Order':10s} | {'Overall':10s}")
    print("-" * 75)
    for d in decays:
        d_recs = [r for r in records if r["decay"] == d]
        row_str = f"{d:20s}"
        vals = []
        for t in tasks:
            t_vals = [r["direct_id"] * 100 for r in d_recs if r["task"] == t]
            m = np.mean(t_vals)
            vals.append(m)
            row_str += f" | {m:9.2f}%"
        row_str += f" | {np.mean(vals):9.2f}%"
        print(row_str)

    # 2. Main Effect: Factor B (Sensory Gating)
    print("\n--- 2. FACTOR B: SENSORY GATING (Mean Direct Accuracy %) ---")
    gatings = ["none", "recurrent_only", "soft_sensory"]
    print(f"{'Gating':20s} | {'Binding':10s} | {'XOR':10s} | {'Distractor':10s} | {'Order':10s} | {'Overall':10s}")
    print("-" * 75)
    for g in gatings:
        g_recs = [r for r in records if r["gating"] == g]
        row_str = f"{g:20s}"
        vals = []
        for t in tasks:
            t_vals = [r["direct_id"] * 100 for r in g_recs if r["task"] == t]
            m = np.mean(t_vals)
            vals.append(m)
            row_str += f" | {m:9.2f}%"
        row_str += f" | {np.mean(vals):9.2f}%"
        print(row_str)

    # 3. Main Effect: Factor C (Reset Jacobian)
    print("\n--- 3. FACTOR C: RESET JACOBIAN (Mean Direct Accuracy %) ---")
    resets = ["own_reset", "stopped_reset"]
    print(f"{'Reset':20s} | {'Binding':10s} | {'XOR':10s} | {'Distractor':10s} | {'Order':10s} | {'Overall':10s}")
    print("-" * 75)
    for r_mode in resets:
        r_recs = [r for r in records if r["reset"] == r_mode]
        row_str = f"{r_mode:20s}"
        vals = []
        for t in tasks:
            t_vals = [r["direct_id"] * 100 for r in r_recs if r["task"] == t]
            m = np.mean(t_vals)
            vals.append(m)
            row_str += f" | {m:9.2f}%"
        row_str += f" | {np.mean(vals):9.2f}%"
        print(row_str)

    # 4. Strict Matched Control Audit
    print("\n--- 4. STRICT MATCHED CONTROLS CONTRAST (Candidate Direct vs Shuffled Control) ---")
    print(f"{'Task':18s} | {'Learned Direct':15s} | {'Shuffled Direct':15s} | {'Difference (pp)':15s} | {'Sign-test p'}")
    print("-" * 75)
    for t in tasks:
        t_recs = [r for r in records if r["task"] == t]
        l_vals = np.array([r["direct_id"] * 100 for r in t_recs])
        s_vals = np.array([r["shuffled_direct_id"] * 100 for r in t_recs])
        diff = l_vals - s_vals
        mean_diff = np.mean(diff)
        pos = np.sum(diff > 0)
        n = len(diff)
        print(f"{t:18s} | {np.mean(l_vals):14.2f}% | {np.mean(s_vals):14.2f}% | {mean_diff:+14.2f}pp | {pos}/{n} wins")

    # 5. Causal Revert Audit
    print("\n--- 5. CAUSAL REVERT AUDIT (Learned Direct vs Reverted Direct) ---")
    print(f"{'Task':18s} | {'Learned Direct':15s} | {'Reverted Direct':15s} | {'Causal Drop (pp)':16s}")
    print("-" * 75)
    for t in tasks:
        t_recs = [r for r in records if r["task"] == t]
        l_vals = np.array([r["direct_id"] * 100 for r in t_recs])
        rev_vals = np.array([r["causal_revert_direct_id"] * 100 for r in t_recs])
        drop = np.array([r["causal_drop"] * 100 for r in t_recs])
        print(f"{t:18s} | {np.mean(l_vals):14.2f}% | {np.mean(rev_vals):14.2f}% | {np.mean(drop):+15.2f}pp")

    # 6. Direct vs Probe Disentanglement
    print("\n--- 6. DIRECT VS PROBE DISENTANGLEMENT ---")
    print(f"{'Task':18s} | {'Direct ID':10s} | {'Frozen Probe':13s} | {'Learned Probe':14s} | {'Probe Gain':12s} | {'Direct - Probe'}")
    print("-" * 80)
    for t in tasks:
        t_recs = [r for r in records if r["task"] == t]
        dir_m = np.mean([r["direct_id"] * 100 for r in t_recs])
        fr_probe_m = np.mean([r["frozen_probe_id"] * 100 for r in t_recs])
        l_probe_m = np.mean([r["probe_id"] * 100 for r in t_recs])
        gain_m = np.mean([r["probe_gain"] * 100 for r in t_recs])
        gap = dir_m - l_probe_m
        print(f"{t:18s} | {dir_m:9.2f}% | {fr_probe_m:12.2f}% | {l_probe_m:13.2f}% | {gain_m:+11.2f}pp | {gap:+12.2f}pp")

    # 7. Top Configurations on Strict Binding and XOR
    print("\n--- 7. TOP CONFIGURATIONS RANKED BY MEAN DIRECT ACCURACY (Strict Binding + XOR) ---")
    config_scores = {}
    for r in records:
        cid = r["config_id"]
        if cid not in config_scores:
            config_scores[cid] = {"strict_binding": [], "xor": [], "distractor_binding": [], "order": []}
        config_scores[cid][r["task"]].append(r["direct_id"] * 100)

    ranked = sorted(
        config_scores.items(),
        key=lambda item: (np.mean(item[1]["strict_binding"]) + np.mean(item[1]["xor"])) / 2,
        reverse=True,
    )
    print(f"{'Rank':4s} | {'Configuration':42s} | {'Binding':9s} | {'XOR':9s} | {'Distractor':10s} | {'Order':9s} | {'Score'}")
    print("-" * 100)
    for idx, (cid, scores) in enumerate(ranked[:10], 1):
        b = np.mean(scores["strict_binding"])
        x = np.mean(scores["xor"])
        d = np.mean(scores["distractor_binding"])
        o = np.mean(scores["order"])
        comb = (b + x) / 2
        print(f"{idx:4d} | {cid:42s} | {b:8.2f}% | {x:8.2f}% | {d:9.2f}% | {o:8.2f}% | {comb:6.2f}%")


if __name__ == "__main__":
    main()
