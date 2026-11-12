import json
from pathlib import Path

with open('artifacts/r11_shared_learning_readout_results.json') as f:
    d = json.load(f)

print("=" * 80)
print("SCAR-X ROUND 11 SUMMARY REPORT")
print("=" * 80)

tasks = ["strict_binding", "xor", "distractor_binding", "order"]

def print_arm_a(arm_data, name):
    print(f"\n--- {name} ---")
    for t in tasks:
        rows = arm_data[t]
        directs = [r['id_direct'] * 100 for r in rows]
        probes = [r['id_probe'] * 100 for r in rows]
        oods = [r['ood_direct'] * 100 for r in rows]
        print(f"  {t:18s}: Direct = {sum(directs)/len(directs):5.2f}% | Probe = {sum(probes)/len(probes):5.2f}% | OOD = {sum(oods)/len(oods):5.2f}% | Seeds = {[round(x, 1) for x in directs]}")

print_arm_a(d['arm_a_isolated_sgd'], "ARM A: ISOLATED (Online SGD)")
print_arm_a(d['arm_a_isolated_rls'], "ARM A: ISOLATED (Online RLS)")

def print_joint(arm_data, name):
    print(f"\n--- {name} ---")
    summary = arm_data['summary_per_task']
    for t in tasks:
        s = summary[t]
        print(f"  {t:18s}: Direct = {s['mean_direct']*100:5.2f}% | Probe = {s['mean_probe']*100:5.2f}% | OOD = {s['mean_ood']*100:5.2f}% | Probe-Direct Gap = {s['mean_probe']*100 - s['mean_direct']*100:+5.2f} pp")

print_joint(d['arm_b_joint_1024_sgd'], "ARM B: JOINT 1024 TOTAL (256/task) - Online SGD")
print_joint(d['arm_b_joint_1024_rls'], "ARM B: JOINT 1024 TOTAL (256/task) - Online RLS")
print_joint(d['arm_c_joint_4096_sgd'], "ARM C: JOINT 4096 TOTAL (1024/task) - Online SGD")
print_joint(d['arm_c_joint_4096_rls'], "ARM C: JOINT 4096 TOTAL (1024/task) - Online RLS")

print("\n" + "=" * 80)
print("P4 SEQUENTIAL STREAM CONTINUAL LEARNING: FORGETTING MATRIX")
print("=" * 80)
for seed_key, seed_data in d['p4_sequential_stream'].items():
    print(f"\n--- {seed_key.upper()} ---")
    phases = seed_data['phases']
    for p in phases:
        t_trained = p['trained_task']
        p_num = p['phase']
        evals = p['eval']
        scores = [f"{t[:4]}:{evals[t]['id_direct']*100:4.1f}%" for t in tasks]
        print(f"  Phase {p_num} (Trained {t_trained:18s}) -> " + " | ".join(scores))
    print(f"  Gradient Interference: {seed_data['gradient_interference_cosines']}")
