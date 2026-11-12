"""SCAR-X Round 8: Pre-Registered Confirmation Suite.

Evaluates locked confirmation seeds: 129001, 129007, 129013, 129019, 129025.
No hyperparameter tuning permitted.
Evaluates:
- P1 MTSG e-prop across strict_binding, distractor_binding, xor, order.
- P2 BF-Cascade continual learning stream (order -> xor -> strict_binding).
- Statistical intervals: Mean, SD, 95% Confidence Intervals (t_4 = 2.776), worst seed.
- Programmatic gate evaluation via R8GateEvaluator.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from scarx.neural.core import Config
from scarx.neural.r8_mtsg_eprop import MultiTimescaleSensoryGatedEProp
from scarx.neural.r8_bf_cascade import BennaFusiMetaplasticCascade
from scarx.evaluation.tasks21 import ridge_probe
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_distractor_binding,
    generate_xor_representation,
    generate_order_representation,
    assert_input_dimension,
)
from scarx.evaluation.r8_gate_evaluator import R8GateEvaluator

CONFIRMATION_SEEDS = [129001, 129007, 129013, 129019, 129025]
TASKS = ["strict_binding", "distractor_binding", "xor", "order"]


def get_task_datasets(task: str, seed: int):
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


def compute_ci95(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    # t critical for df=n-1 (for n=5, df=4, t=2.776)
    t_crit = 2.776 if n == 5 else 4.303
    return mean, mean - t_crit * se, mean + t_crit * se


def run_confirmation():
    print("=== SCAR-X ROUND 8: PRE-REGISTERED CONFIRMATION RUN ===")
    results: dict[str, dict] = {"p1_results": {}, "p2_results": {}, "per_seed_records": []}

    # -----------------------------------------------------------------------
    # Part 1: P1 Confirmation across Tasks
    # -----------------------------------------------------------------------
    for task in TASKS:
        print(f"\n--- Running Confirmation for Task: {task} ---")
        task_data = {
            "learned_mtsg": {"direct": [], "probe": [], "ood_probe": []},
            "frozen_hidden": {"probe": []},
            "r35_baseline": {"probe": []},
            "shuffled_signal": {"probe": []},
            "causal_revert": {"drop": []},
        }

        for seed in CONFIRMATION_SEEDS:
            tr_x, tr_y, fit_x, fit_y, te_x, te_y, ood_x, ood_y = get_task_datasets(task, seed)
            assert_input_dimension(tr_x, 8)
            cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)

            # 1. Learned MTSG
            model = MultiTimescaleSensoryGatedEProp(cfg, condition="learned_mtsg")
            model.calibrate(fit_x[:32])
            for seq, y in zip(tr_x, tr_y):
                model.learn_episode(seq, int(y))

            d_acc = float(np.mean(model.predict_many(te_x) == te_y))
            ff = model.features_many(fit_x); tf = model.features_many(te_x); ood_f = model.features_many(ood_x)
            p_acc = ridge_probe(ff, fit_y, tf, te_y)
            ood_p_acc = ridge_probe(ff, fit_y, ood_f, ood_y)

            task_data["learned_mtsg"]["direct"].append(d_acc)
            task_data["learned_mtsg"]["probe"].append(p_acc)
            task_data["learned_mtsg"]["ood_probe"].append(ood_p_acc)

            # Causal revert
            model_rev = MultiTimescaleSensoryGatedEProp.loads(model.dumps())
            model_rev.revert_hidden_weights(revert_in=True, revert_rec=True)
            p_rev = ridge_probe(model_rev.features_many(fit_x), fit_y, model_rev.features_many(te_x), te_y)
            task_data["causal_revert"]["drop"].append(p_acc - p_rev)

            # 2. Frozen hidden
            model_fr = MultiTimescaleSensoryGatedEProp(cfg, condition="frozen_hidden")
            model_fr.calibrate(fit_x[:32])
            ff_fr = model_fr.features_many(fit_x); tf_fr = model_fr.features_many(te_x)
            p_fr = ridge_probe(ff_fr, fit_y, tf_fr, te_y)
            task_data["frozen_hidden"]["probe"].append(p_fr)

            # 3. R35 baseline
            model_r35 = MultiTimescaleSensoryGatedEProp(cfg, condition="r35_baseline")
            model_r35.calibrate(fit_x[:32])
            for seq, y in zip(tr_x, tr_y):
                model_r35.learn_episode(seq, int(y))
            ff_r35 = model_r35.features_many(fit_x); tf_r35 = model_r35.features_many(te_x)
            p_r35 = ridge_probe(ff_r35, fit_y, tf_r35, te_y)
            task_data["r35_baseline"]["probe"].append(p_r35)

            # 4. Shuffled signal control
            model_shuf = MultiTimescaleSensoryGatedEProp(cfg, condition="shuffled_signal")
            model_shuf.calibrate(fit_x[:32])
            for seq, y in zip(tr_x, tr_y):
                model_shuf.learn_episode(seq, int(y))
            ff_shuf = model_shuf.features_many(fit_x); tf_shuf = model_shuf.features_many(te_x)
            p_shuf = ridge_probe(ff_shuf, fit_y, tf_shuf, te_y)
            task_data["shuffled_signal"]["probe"].append(p_shuf)

            print(
                f"Seed {seed}: MTSG direct={d_acc*100:.2f}%, probe={p_acc*100:.2f}%, "
                f"Frozen probe={p_fr*100:.2f}%, Revert drop={(p_acc-p_rev)*100:+.2f}pp"
            )

        # Paired gain statistics
        gains = np.array(task_data["learned_mtsg"]["probe"]) - np.array(task_data["frozen_hidden"]["probe"])
        mean_g, low_g, high_g = compute_ci95(gains)

        direct_arr = np.array(task_data["learned_mtsg"]["direct"])
        probe_arr = np.array(task_data["learned_mtsg"]["probe"])

        results["p1_results"][task] = {
            "learned_mtsg": {
                "direct_acc_mean": float(np.mean(direct_arr)),
                "direct_acc_sd": float(np.std(direct_arr, ddof=1)),
                "direct_acc_worst": float(np.min(direct_arr)),
                "probe_acc_mean": float(np.mean(probe_arr)),
                "probe_acc_sd": float(np.std(probe_arr, ddof=1)),
                "probe_acc_worst": float(np.min(probe_arr)),
                "probe_gain_mean": mean_g,
                "probe_gain_lower95": low_g,
                "probe_gain_upper95": high_g,
                "ood_probe_mean": float(np.mean(task_data["learned_mtsg"]["ood_probe"])),
            },
            "frozen_hidden": {
                "probe_acc_mean": float(np.mean(task_data["frozen_hidden"]["probe"])),
                "probe_acc_sd": float(np.std(task_data["frozen_hidden"]["probe"], ddof=1)),
            },
            "r35_baseline": {
                "probe_acc_mean": float(np.mean(task_data["r35_baseline"]["probe"])),
                "probe_acc_sd": float(np.std(task_data["r35_baseline"]["probe"], ddof=1)),
            },
            "shuffled_signal": {
                "probe_acc_mean": float(np.mean(task_data["shuffled_signal"]["probe"])),
                "probe_acc_sd": float(np.std(task_data["shuffled_signal"]["probe"], ddof=1)),
            },
            "causal_revert": {
                "drop_mean": float(np.mean(task_data["causal_revert"]["drop"])),
                "drop_sd": float(np.std(task_data["causal_revert"]["drop"], ddof=1)),
            },
        }

    # -----------------------------------------------------------------------
    # Part 2: P2 Confirmation: Continual Learning Stream
    # -----------------------------------------------------------------------
    print("\n--- Running Confirmation for P2 Continual Stream ---")
    bf_acqs, bf_finals, bf_forgs = [], [], []
    unconsol_forgs = []

    for seed in CONFIRMATION_SEEDS:
        cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
        model_bf = BennaFusiMetaplasticCascade(cfg, use_cascade=True, kappa1=0.004, kappa2=0.001)
        model_un = BennaFusiMetaplasticCascade(cfg, use_cascade=False)

        o_tr = generate_order_representation(seed + 100, 512); o_te = generate_order_representation(seed + 1100, 256)
        x_tr = generate_xor_representation(seed + 200, 512); x_te = generate_xor_representation(seed + 1200, 256)
        b_tr = generate_strict_binding(seed + 300, 512); b_te = generate_strict_binding(seed + 1300, 256)
        skills = [("order", o_tr, o_te), ("xor", x_tr, x_te), ("strict_binding", b_tr, b_te)]

        cal = np.concatenate([o_tr[0][:16], x_tr[0][:16], b_tr[0][:16]], axis=0)
        model_bf.calibrate(cal); model_un.calibrate(cal)

        n_s = len(skills)
        mat_bf = np.zeros((n_s, n_s), dtype=np.float64)
        mat_un = np.zeros((n_s, n_s), dtype=np.float64)

        for s_idx, (_, (trx, try_), _) in enumerate(skills):
            for seq, y in zip(trx, try_):
                model_bf.learn_episode(seq, int(y))
                model_un.learn_episode(seq, int(y))
            for eval_idx in range(s_idx + 1):
                _, _, (tex, tey) = skills[eval_idx]
                mat_bf[s_idx, eval_idx] = float(np.mean(model_bf.predict_many(tex) == tey))
                mat_un[s_idx, eval_idx] = float(np.mean(model_un.predict_many(tex) == tey))

        diag_bf = np.diag(mat_bf)
        final_bf = mat_bf[-1, :]
        forg_bf = [float(max(mat_bf[t, j] for t in range(j, n_s)) - final_bf[j]) for j in range(n_s)]
        forg_un = [float(max(mat_un[t, j] for t in range(j, n_s)) - mat_un[-1, j]) for j in range(n_s)]

        bf_acqs.append(float(np.mean(diag_bf)))
        bf_finals.append(float(np.mean(final_bf)))
        bf_forgs.append(float(np.mean(forg_bf)))
        unconsol_forgs.append(float(np.mean(forg_un)))

        print(
            f"Seed {seed}: BF Acq={np.mean(diag_bf)*100:.2f}%, Final={np.mean(final_bf)*100:.2f}%, "
            f"MeanForg={np.mean(forg_bf)*100:.2f}% (vs Unconsol: {np.mean(forg_un)*100:.2f}%)"
        )

    results["p2_results"] = {
        "bf_cascade": {
            "acquisition_mean": float(np.mean(bf_acqs)),
            "acquisition_sd": float(np.std(bf_acqs, ddof=1)),
            "acquisition_worst": float(np.min(bf_acqs)),
            "final_mean": float(np.mean(bf_finals)),
            "final_sd": float(np.std(bf_finals, ddof=1)),
            "final_worst": float(np.min(bf_finals)),
            "mean_forgetting": float(np.mean(bf_forgs)),
            "mean_forgetting_sd": float(np.std(bf_forgs, ddof=1)),
        },
        "unconsolidated": {
            "mean_forgetting": float(np.mean(unconsol_forgs)),
            "mean_forgetting_sd": float(np.std(unconsol_forgs, ddof=1)),
        },
    }

    # Evaluate all gates programmatically
    evaluator = R8GateEvaluator(results)
    eval_res = evaluator.evaluate_full()
    results["gate_evaluation"] = eval_res

    out_path = Path("artifacts/gemini-r8/confirmation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== CONFIRMATION GATE VERDICTS ===")
    for t, t_eval in eval_res["p1_tasks"].items():
        print(f"Task {t:20s}: Status = {t_eval['status']}")
    print(f"P2 Stream Status    : {eval_res['p2_stream']['status']}")
    print(f"Overall Confirmation: {eval_res['overall_status']}")
    print(f"Artifact written to : {out_path.resolve()}")


if __name__ == "__main__":
    run_confirmation()
