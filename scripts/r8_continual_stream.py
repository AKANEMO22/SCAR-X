"""SCAR-X Round 8: Task-Agnostic Continual Learning Stream Evaluation.

Evaluates sequential skill learning (order -> xor -> strict_binding) in a single
SpikingCore(H=48) without task identity cues, without boundary snapshots, and
without replay buffers (B=0).

Compares:
1. unconsolidated baseline (standard MTSG e-prop, kappa1=0, kappa2=0)
2. bf_cascade (Benna-Fusi metaplastic cascade, kappa1=0.004, kappa2=0.001)
3. bf_cascade_slow (slower diffusion, kappa1=0.002, kappa2=0.0005)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from scarx.neural.core import Config
from scarx.neural.r8_bf_cascade import BennaFusiMetaplasticCascade
from scarx.evaluation.r8_benchmark_integrity import (
    generate_order_representation,
    generate_xor_representation,
    generate_strict_binding,
    assert_input_dimension,
)

DEV_SEEDS = [140011, 140017, 140023]


def run_continual_stream_for_seed(seed: int, mode: str) -> dict[str, float | list]:
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)

    if mode == "unconsolidated":
        model = BennaFusiMetaplasticCascade(cfg, use_cascade=False)
    elif mode == "bf_cascade":
        model = BennaFusiMetaplasticCascade(cfg, use_cascade=True, kappa1=0.004, kappa2=0.001)
    elif mode == "bf_cascade_slow":
        model = BennaFusiMetaplasticCascade(cfg, use_cascade=True, kappa1=0.002, kappa2=0.0005)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Generate clean datasets (strictly D=8)
    order_tr = generate_order_representation(seed + 100, 512)
    order_te = generate_order_representation(seed + 1100, 256)
    xor_tr = generate_xor_representation(seed + 200, 512)
    xor_te = generate_xor_representation(seed + 1200, 256)
    bind_tr = generate_strict_binding(seed + 300, 512)
    bind_te = generate_strict_binding(seed + 1300, 256)

    skills = [("order", order_tr, order_te), ("xor", xor_tr, xor_te), ("strict_binding", bind_tr, bind_te)]

    # Calibrate on initial sensory samples
    cal_seqs = np.concatenate([order_tr[0][:16], xor_tr[0][:16], bind_tr[0][:16]], axis=0)
    model.calibrate(cal_seqs)

    n_skills = len(skills)
    matrix = np.zeros((n_skills, n_skills), dtype=np.float64)

    # Continual stream loop (zero task cues, zero task boundary metadata)
    for s_idx, (name, (tr_x, tr_y), _) in enumerate(skills):
        assert_input_dimension(tr_x, 8)
        for seq, y in zip(tr_x, tr_y):
            model.learn_episode(seq, int(y))

        # Evaluate on all tasks seen so far
        for eval_idx in range(s_idx + 1):
            _, _, (te_x, te_y) = skills[eval_idx]
            preds = model.predict_many(te_x)
            matrix[s_idx, eval_idx] = float(np.mean(preds == te_y))

    diag = np.diag(matrix).tolist()
    final = matrix[-1, :].tolist()
    forg = [float(max(matrix[t, j] for t in range(j, n_skills)) - final[j]) for j in range(n_skills)]

    return {
        "matrix": matrix.tolist(),
        "acquisition": diag,
        "acquisition_mean": float(np.mean(diag)),
        "final": final,
        "final_mean": float(np.mean(final)),
        "forgetting": forg,
        "mean_forgetting": float(np.mean(forg)),
        "worst_forgetting": float(np.max(forg)),
    }


def run_all_continual_streams():
    print("=== SCAR-X ROUND 8: TASK-AGNOSTIC CONTINUAL STREAM EVALUATION ===")
    modes = ["unconsolidated", "bf_cascade", "bf_cascade_slow"]
    summary: dict[str, dict] = {}

    for mode in modes:
        print(f"\n--- Testing Mode: {mode} ---")
        mode_runs = []
        for seed in DEV_SEEDS:
            run_res = run_continual_stream_for_seed(seed, mode)
            mode_runs.append(run_res)
            print(
                f"Seed {seed}: Acq={run_res['acquisition_mean']*100:.2f}%, "
                f"Final={run_res['final_mean']*100:.2f}%, "
                f"MeanForg={run_res['mean_forgetting']*100:.2f}%, "
                f"WorstForg={run_res['worst_forgetting']*100:.2f}%"
            )

        acqs = [r["acquisition_mean"] for r in mode_runs]
        finals = [r["final_mean"] for r in mode_runs]
        mean_forgs = [r["mean_forgetting"] for r in mode_runs]
        worst_forgs = [r["worst_forgetting"] for r in mode_runs]

        summary[mode] = {
            "acquisition_mean": float(np.mean(acqs)),
            "acquisition_sd": float(np.std(acqs, ddof=1)),
            "final_mean": float(np.mean(finals)),
            "final_sd": float(np.std(finals, ddof=1)),
            "mean_forgetting": float(np.mean(mean_forgs)),
            "mean_forgetting_sd": float(np.std(mean_forgs, ddof=1)),
            "worst_forgetting_mean": float(np.mean(worst_forgs)),
            "per_seed": mode_runs,
        }

    # Paired retention improvement of BF-Cascade over unconsolidated
    base_forgs = np.array([r["mean_forgetting"] for r in summary["unconsolidated"]["per_seed"]])
    bf_forgs = np.array([r["mean_forgetting"] for r in summary["bf_cascade"]["per_seed"]])
    forg_reductions = base_forgs - bf_forgs
    summary["retention_contrast"] = {
        "mean_reduction_pp": float(np.mean(forg_reductions) * 100),
        "min_reduction_pp": float(np.min(forg_reductions) * 100),
        "max_reduction_pp": float(np.max(forg_reductions) * 100),
    }

    out_path = Path("artifacts/gemini-r8/continual-stream.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CONTINUAL STREAM SUMMARY ===")
    for m in modes:
        print(
            f"Mode {m:15s}: Acq={summary[m]['acquisition_mean']*100:.2f}%, "
            f"Final={summary[m]['final_mean']*100:.2f}%, "
            f"MeanForg={summary[m]['mean_forgetting']*100:.2f}%"
        )
    print(f"Forgetting Reduction: {summary['retention_contrast']['mean_reduction_pp']:.2f} pp")
    print(f"Artifact written to: {out_path.resolve()}")


if __name__ == "__main__":
    run_all_continual_streams()
