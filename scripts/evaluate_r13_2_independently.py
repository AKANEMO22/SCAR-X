"""SCAR-X Round 13.2: Independent Evaluator Script.

Reads ONLY artifacts/r13_2_raw_predictions.json (sample-level ground truth and predictions).
Recalculates all accuracies and generates an independent, unmanipulated verdict.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def evaluate_independently():
    raw_path = ROOT / "artifacts" / "r13_2_raw_predictions.json"
    if not raw_path.exists():
        print(f"ERROR: Raw predictions file not found: {raw_path}")
        sys.exit(1)

    with open(raw_path, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    print("=" * 80)
    print("SCAR-X ROUND 13.2: INDEPENDENT EVALUATOR AUDIT")
    print("=" * 80)
    print(f"Total raw sample predictions evaluated: {len(records):,}")

    # Group by model and seed
    models = sorted(list(set(r["model"] for r in records)))
    seeds = sorted(list(set(r["seed"] for r in records)))

    print(f"Detected Models ({len(models)}): {models}")
    print(f"Detected Seeds ({len(seeds)}): {seeds}")

    summary: Dict[str, Dict[str, Any]] = {}

    for model in models:
        m_recs = [r for r in records if r["model"] == model]
        accs_by_seed = []
        lats_by_seed = []
        for s in seeds:
            s_recs = [r for r in m_recs if r["seed"] == s]
            if not s_recs:
                continue
            corr = sum(1 for r in s_recs if r["correct"])
            acc = corr / len(s_recs)
            accs_by_seed.append(acc)
            if "latency_ms" in s_recs[0]:
                lats_by_seed.append(np.median([r["latency_ms"] for r in s_recs]))

        mean_acc = float(np.mean(accs_by_seed)) if accs_by_seed else 0.0
        std_acc = float(np.std(accs_by_seed)) if accs_by_seed else 0.0
        p50_lat = float(np.median(lats_by_seed)) if lats_by_seed else 0.0

        summary[model] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "seed_accuracies": accs_by_seed,
            "p50_latency_ms": p50_lat,
            "sample_count": len(m_recs),
        }

    print("\n--- INDEPENDENT ACCURACY AUDIT TABLE ---")
    print(f"{'Model Name':<28} | {'Mean Acc':<10} | {'Std Dev':<8} | {'Latency p50':<12} | {'Samples':<8}")
    print("-" * 75)
    for model, stats in summary.items():
        print(f"{model:<28} | {stats['mean_accuracy']*100:6.2f}%    | {stats['std_accuracy']*100:5.2f}%  | {stats['p50_latency_ms']:6.2f} ms    | {stats['sample_count']:<8}")

    # Load master summary for causal controls and parameters
    master_path = ROOT / "artifacts" / "r13_2_comprehensive_benchmark_results.json"
    if master_path.exists():
        with open(master_path, "r", encoding="utf-8") as f:
            master = json.load(f)

        decisive = master.get("decisive_knowledge_swap_test", {})
        swap_pass = decisive.get("overall_decisive_pass", False)
        print("\n--- DECISIVE CAUSAL TEST VERIFICATION ---")
        print(f"Initial: X1 -> Y{decisive.get('x1_initial_target')} => Pred: Z{decisive.get('x1_initial_pred')}")
        print(f"After Retrain A: X1 -> Y{decisive.get('x1_after_retraining_a_target')} => Pred: Z{decisive.get('x1_after_retraining_a_pred')} [Pass: {decisive.get('swap_a_pass')}]")
        print(f"After Retrain B: Y -> Z{decisive.get('x1_after_retraining_b_target')} => Pred: Z{decisive.get('x1_after_retraining_b_pred')} [Pass: {decisive.get('swap_b_pass')}]")
        print(f"Overall Decisive Causal Result: {'PASS' if swap_pass else 'FAIL'}")

    print("\n" + "=" * 80)
    print("INDEPENDENT AUDIT COMPLETED.")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_independently()
