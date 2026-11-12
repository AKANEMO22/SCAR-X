"""SCAR-X Round 8: Pre-Registered Development Screen for MTSG e-prop (Candidate 1).

Evaluates across pre-registered DEV seeds: 140011, 140017, 140023.
Tasks: strict_binding, distractor_binding, xor, order.
All causal controls: frozen_hidden, zero_signal, shuffled_signal, readout_only,
r35_baseline, ew_he_legacy, learned_mtsg, and causal revert.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np

from scarx.neural.core import Config, SpikingCore
from scarx.neural.r8_mtsg_eprop import MultiTimescaleSensoryGatedEProp
from scarx.evaluation.tasks21 import ridge_probe
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_distractor_binding,
    generate_xor_representation,
    generate_order_representation,
    assert_input_dimension,
    assert_spikes_binary,
)
from scarx.evaluation.r8_gate_evaluator import R8GateEvaluator

DEV_SEEDS = [140011, 140017, 140023]
TASKS = ["strict_binding", "distractor_binding", "xor", "order"]
CONDITIONS = [
    "frozen_hidden",
    "zero_signal",
    "shuffled_signal",
    "readout_only",
    "r35_baseline",
    "ew_he_legacy",
    "learned_mtsg",
]


def get_task_data(task: str, seed: int):
    if task == "strict_binding":
        tr_x, tr_y = generate_strict_binding(seed + 200, 1024)
        fit_x, fit_y = generate_strict_binding(seed + 300, 256)
        te_x, te_y = generate_strict_binding(seed + 400, 512)
        ood_x, ood_y = generate_strict_binding(seed + 500, 512, ood=True)
    elif task == "distractor_binding":
        tr_x, tr_y = generate_distractor_binding(seed + 200, 1024)
        fit_x, fit_y = generate_distractor_binding(seed + 300, 256)
        te_x, te_y = generate_distractor_binding(seed + 400, 512)
        ood_x, ood_y = generate_distractor_binding(seed + 500, 512, ood=True)
    elif task == "xor":
        tr_x, tr_y = generate_xor_representation(seed + 200, 1024)
        fit_x, fit_y = generate_xor_representation(seed + 300, 256)
        te_x, te_y = generate_xor_representation(seed + 400, 512)
        ood_x, ood_y = generate_xor_representation(seed + 500, 512, ood=True)
    elif task == "order":
        tr_x, tr_y = generate_order_representation(seed + 200, 1024)
        fit_x, fit_y = generate_order_representation(seed + 300, 256)
        te_x, te_y = generate_order_representation(seed + 400, 512)
        ood_x, ood_y = generate_order_representation(seed + 500, 512, ood=True)
    else:
        raise ValueError(f"Unknown task: {task}")
    return tr_x, tr_y, fit_x, fit_y, te_x, te_y, ood_x, ood_y


def run_dev_screen():
    print("=== SCAR-X ROUND 8: PRE-REGISTERED DEV SCREEN ===")
    results: dict[str, dict] = {"p1_results": {}, "trials": []}

    for task in TASKS:
        print(f"\n--- Testing Task: {task} ---")
        task_data = {c: {"direct_accs": [], "probe_accs": [], "ood_probe_accs": []} for c in CONDITIONS}
        causal_revert_drops = []
        revert_in_drops = []
        revert_rec_drops = []

        for seed in DEV_SEEDS:
            tr_x, tr_y, fit_x, fit_y, te_x, te_y, ood_x, ood_y = get_task_data(task, seed)
            assert_input_dimension(tr_x, 8)

            for cond in CONDITIONS:
                cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
                model = MultiTimescaleSensoryGatedEProp(cfg, condition=cond)
                model.calibrate(fit_x[:32])

                # Train
                for seq, y in zip(tr_x, tr_y):
                    model.learn_episode(seq, int(y))

                # Direct accuracy
                preds = model.predict_many(te_x)
                d_acc = float(np.mean(preds == te_y))

                # Ridge probe accuracy (frozen features representation)
                ff = model.features_many(fit_x)
                tf = model.features_many(te_x)
                ood_f = model.features_many(ood_x)
                p_acc = ridge_probe(ff, fit_y, tf, te_y)
                ood_p_acc = ridge_probe(ff, fit_y, ood_f, ood_y)

                task_data[cond]["direct_accs"].append(d_acc)
                task_data[cond]["probe_accs"].append(p_acc)
                task_data[cond]["ood_probe_accs"].append(ood_p_acc)

                # Causal revert check for learned_mtsg
                if cond == "learned_mtsg":
                    # Full revert
                    model_full_rev = MultiTimescaleSensoryGatedEProp.loads(model.dumps())
                    model_full_rev.revert_hidden_weights(revert_in=True, revert_rec=True)
                    p_rev = ridge_probe(
                        model_full_rev.features_many(fit_x),
                        fit_y,
                        model_full_rev.features_many(te_x),
                        te_y,
                    )
                    drop = p_acc - p_rev
                    causal_revert_drops.append(drop)

                    # Revert w_in only
                    model_in_rev = MultiTimescaleSensoryGatedEProp.loads(model.dumps())
                    model_in_rev.revert_hidden_weights(revert_in=True, revert_rec=False)
                    p_in_rev = ridge_probe(
                        model_in_rev.features_many(fit_x),
                        fit_y,
                        model_in_rev.features_many(te_x),
                        te_y,
                    )
                    revert_in_drops.append(p_acc - p_in_rev)

                    # Revert w_rec only
                    model_rec_rev = MultiTimescaleSensoryGatedEProp.loads(model.dumps())
                    model_rec_rev.revert_hidden_weights(revert_in=False, revert_rec=True)
                    p_rec_rev = ridge_probe(
                        model_rec_rev.features_many(fit_x),
                        fit_y,
                        model_rec_rev.features_many(te_x),
                        te_y,
                    )
                    revert_rec_drops.append(p_acc - p_rec_rev)

            mtsg_d = task_data["learned_mtsg"]["direct_accs"][-1]
            mtsg_p = task_data["learned_mtsg"]["probe_accs"][-1]
            fr_p = task_data["frozen_hidden"]["probe_accs"][-1]
            r35_p = task_data["r35_baseline"]["probe_accs"][-1]
            print(
                f"Seed {seed}: MTSG direct={mtsg_d*100:.2f}%, MTSG probe={mtsg_p*100:.2f}%, "
                f"Frozen probe={fr_p*100:.2f}%, R35 probe={r35_p*100:.2f}%"
            )

        # Summarize task results
        task_summary = {}
        for cond in CONDITIONS:
            d_accs = task_data[cond]["direct_accs"]
            p_accs = task_data[cond]["probe_accs"]
            ood_accs = task_data[cond]["ood_probe_accs"]
            task_summary[cond] = {
                "direct_acc_mean": float(np.mean(d_accs)),
                "direct_acc_sd": float(np.std(d_accs, ddof=1)),
                "probe_acc_mean": float(np.mean(p_accs)),
                "probe_acc_sd": float(np.std(p_accs, ddof=1)),
                "ood_probe_mean": float(np.mean(ood_accs)),
            }

        # Compute 95% CI for MTSG probe gain over frozen
        gains = np.array(task_data["learned_mtsg"]["probe_accs"]) - np.array(task_data["frozen_hidden"]["probe_accs"])
        n = len(gains)
        mean_g = float(np.mean(gains))
        se_g = float(np.std(gains, ddof=1) / np.sqrt(n))
        # t critical for df=2, 95% CI is approx 4.303
        ci_lower = mean_g - 4.303 * se_g
        ci_upper = mean_g + 4.303 * se_g
        task_summary["learned_mtsg"]["probe_gain_mean"] = mean_g
        task_summary["learned_mtsg"]["probe_gain_lower95"] = ci_lower
        task_summary["learned_mtsg"]["probe_gain_upper95"] = ci_upper

        task_summary["causal_revert"] = {
            "drop_mean": float(np.mean(causal_revert_drops)),
            "drop_sd": float(np.std(causal_revert_drops, ddof=1)),
            "revert_in_drop_mean": float(np.mean(revert_in_drops)),
            "revert_rec_drop_mean": float(np.mean(revert_rec_drops)),
        }

        results["p1_results"][task] = task_summary

    # Run programmatic gate evaluation
    evaluator = R8GateEvaluator(results)
    eval_res = evaluator.evaluate_full()
    results["gate_evaluation"] = eval_res

    # Save artifact
    out_path = Path("artifacts/gemini-r8/dev-screen.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== DEV SCREEN GATE EVALUATION SUMMARY ===")
    for t, t_eval in eval_res["p1_tasks"].items():
        print(f"Task {t}: Status = {t_eval['status']}, Notes = {t_eval['notes']}")
    print(f"\nOverall P1 Status: {eval_res['overall_status']}")
    print(f"Artifact written to: {out_path.resolve()}")


if __name__ == "__main__":
    run_dev_screen()
