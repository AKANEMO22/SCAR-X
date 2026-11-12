"""Summarize SCAR-X Round 12 results."""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "artifacts" / "r12_continual_memory_results.json") as f:
    data = json.load(f)

print("=" * 80)
print("SCAR-X ROUND 12 SUMMARY: CONTINUAL MEMORY & REPLAY-LITE STUDY")
print("=" * 80)

# 1. Order 1 Arms Summary
print("\n--- 1. ORDER 1 ARMS: RETENTION & FORGETTING (3 DEV SEEDS) ---")
print(f"{'Arm':<28} | {'Mean Final':<10} | {'Mean Forgetting':<15} | {'Worst Forgetting':<16} | {'Compute Updates'}")
print("-" * 88)

for arm_name, label in [
    ("arm_a_sequential_baseline", "Arm A: Seq Baseline (M=0)"),
    ("arm_b_equal_update_control", "Arm B: Equal-Update (M=0)"),
    ("arm_c_reservoir_16", "Arm C: Reservoir-16 (M=16)"),
    ("arm_d_reservoir_64", "Arm D: Reservoir-64 (M=64)"),
    ("arm_e_interleaved_upper_bound", "Arm E: Interleaved Bound"),
]:
    entries = data["order_1_arms"][arm_name]
    finals = [e["mean_final_direct"] * 100 for e in entries]
    m_forg = [e["mean_forgetting"] * 100 for e in entries]
    w_forg = [e["worst_forgetting"] * 100 for e in entries]
    updates = entries[0]["total_updates"]
    print(f"{label:<28} | {np.mean(finals):6.2f}%    | {np.mean(m_forg):6.2f}%          | {np.mean(w_forg):6.2f}%           | {updates}")

# 2. Per-Task Breakdown for Order 1
print("\n--- 2. PER-TASK ACCURACY & FORGETTING BREAKDOWN (ORDER 1) ---")
tasks = ["strict_binding", "xor", "distractor_binding", "order"]

for task in tasks:
    print(f"\n[TASK: {task.upper()}]")
    print(f"{'Arm':<28} | {'Acquisition':<12} | {'Final Direct':<12} | {'Forgetting':<12} | {'Final Probe':<12} | {'Readout Gap'}")
    print("-" * 92)
    for arm_name, label in [
        ("arm_a_sequential_baseline", "Arm A: Seq Baseline"),
        ("arm_b_equal_update_control", "Arm B: Equal-Update"),
        ("arm_c_reservoir_16", "Arm C: Reservoir-16"),
        ("arm_d_reservoir_64", "Arm D: Reservoir-64"),
        ("arm_e_interleaved_upper_bound", "Arm E: Interleaved"),
    ]:
        entries = data["order_1_arms"][arm_name]
        acqs = [e["task_retention"][task]["acquisition_direct"] * 100 for e in entries]
        fins = [e["task_retention"][task]["final_direct"] * 100 for e in entries]
        forgs = [e["task_retention"][task]["forgetting"] * 100 for e in entries]
        probes = [e["task_retention"][task]["final_probe"] * 100 for e in entries]
        gaps = [e["task_retention"][task]["readout_gap_final"] * 100 for e in entries]
        print(f"{label:<28} | {np.mean(acqs):6.2f}%     | {np.mean(fins):6.2f}%       | {np.mean(forgs):6.2f}%      | {np.mean(probes):6.2f}%      | {np.mean(gaps):6.2f}%")

# 3. Counterfactual Restorations (Phase 4)
print("\n--- 3. COUNTERFACTUAL RESTORATION ANALYSIS (ARM A CHECKPOINTS) ---")
print("Target Task: strict_binding (Trained in Phase 1, evaluated at Phase 4)")
print(f"{'Seed':<10} | {'Final':<8} | {'Restored Readout':<18} | {'Restored Hidden':<18} | {'Restored Full':<15} | {'Readout Gain':<14} | {'Hidden Gain'}")
print("-" * 105)

cf_data = data["counterfactual_restorations_order_1"]
readout_gains = []
hidden_gains = []
for seed_k, s_data in cf_data.items():
    b_data = s_data["strict_binding"]
    fin = b_data["score_final"] * 100
    r_out = b_data["score_restored_readout"] * 100
    r_hid = b_data["score_restored_hidden"] * 100
    r_full = b_data["score_restored_full"] * 100
    g_out = b_data["gain_readout_restoration"] * 100
    g_hid = b_data["gain_hidden_restoration"] * 100
    readout_gains.append(g_out)
    hidden_gains.append(g_hid)
    print(f"{seed_k:<10} | {fin:5.2f}%  | {r_out:5.2f}%             | {r_hid:5.2f}%             | {r_full:5.2f}%         | {g_out:+6.2f} pp      | {g_hid:+6.2f} pp")

print(f"MEAN GAIN  |        |                       |                       |                 | {np.mean(readout_gains):+6.2f} pp      | {np.mean(hidden_gains):+6.2f} pp")

# 4. Gradient Interference Cosines
print("\n--- 4. TASK GRADIENT INTERFERENCE COSINES ON FROZEN PHASE 1 CHECKPOINT ---")
cos_data = data["gradient_interferences"]
pair_cosines = {}
for seed_k, pairs in cos_data.items():
    for p_name, cos_vals in pairs.items():
        if p_name not in pair_cosines:
            pair_cosines[p_name] = {"hidden": [], "readout": [], "full": []}
        pair_cosines[p_name]["hidden"].append(cos_vals["cosine_hidden"])
        pair_cosines[p_name]["readout"].append(cos_vals["cosine_readout"])
        pair_cosines[p_name]["full"].append(cos_vals["cosine_full"])

print(f"{'Task Pair':<40} | {'Cosine Hidden':<15} | {'Cosine Readout':<16} | {'Cosine Full'}")
print("-" * 88)
for p_name, c_dict in pair_cosines.items():
    print(f"{p_name:<40} | {np.mean(c_dict['hidden']):+8.4f}        | {np.mean(c_dict['readout']):+8.4f}         | {np.mean(c_dict['full']):+8.4f}")

# 5. Task Order Invariance (Order 1 vs Order 2)
print("\n--- 5. TASK ORDER INVARIANCE: ORDER 1 vs ORDER 2 (PERMUTED) ---")
print(f"{'Arm':<28} | {'Order 1 Final':<15} | {'Order 2 Final':<15} | {'Order 1 Forgetting':<20} | {'Order 2 Forgetting'}")
print("-" * 96)
for arm_key, label in [
    ("arm_a_sequential_baseline", "Arm A: Seq Baseline"),
    ("arm_c_reservoir_16", "Arm C: Reservoir-16"),
    ("arm_d_reservoir_64", "Arm D: Reservoir-64"),
]:
    o1_finals = [e["mean_final_direct"] * 100 for e in data["order_1_arms"][arm_key]]
    o2_finals = [e["mean_final_direct"] * 100 for e in data["order_2_arms"][arm_key]]
    o1_forg = [e["mean_forgetting"] * 100 for e in data["order_1_arms"][arm_key]]
    o2_forg = [e["mean_forgetting"] * 100 for e in data["order_2_arms"][arm_key]]
    print(f"{label:<28} | {np.mean(o1_finals):6.2f}%         | {np.mean(o2_finals):6.2f}%         | {np.mean(o1_forg):6.2f}%               | {np.mean(o2_forg):6.2f}%")
