"""SCAR-X Round 7: 18-Cell Paired-Across-Orders Continual Retention Benchmark.

Evaluates 3 DEV seeds x 6 skill permutations = 18 paired cells:
Arm A: unconsolidated_no_replay (online EW-HE without consolidation or replay)
Arm B: ew_he_ca_msc (EW-HE + Activity-Gated Synaptic Consolidation CA-MSC + ReplayBuffer)

Directly evaluates the order-variance hypothesis and exchange rate
between forgetting reduction and acquisition cost across all 18 paired cells.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import time
import numpy as np

from scarx.neural.core import Config
from scarx.learning.replay import ReplayBuffer
from scarx.evaluation.tasks21 import representation
from scarx.evaluation.binding_audit21 import strict_binding
from scarx.evaluation.r7_hidden_plasticity import (
    EventWindowedHomeostaticEligibility,
    wrap_skill_pulse,
)

DEV_SEEDS = (140011, 140017, 140023)
SKILL_NAMES = ("order", "xor", "strict_binding")


def bootstrap_ci(values: list[float] | np.ndarray, seed: int = 42, n_boot: int = 10000) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "lower95": float(np.quantile(boot_means, 0.025)),
        "upper95": float(np.quantile(boot_means, 0.975)),
    }


def run_single_cell(
    seed: int,
    order_indices: tuple[int, ...],
    mode: str,
) -> dict[str, object]:
    n_skills = 3
    d_in = 8 + n_skills
    cfg = Config(inputs=d_in, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0.0)
    cond = "ew_he_ca_msc" if mode == "ew_he_ca_msc" else "learned_ew_he"
    model = EventWindowedHomeostaticEligibility(cfg, cond)

    # 1. Calibrate initial homeostatic statistics across all 3 skills
    cal_batches = [
        wrap_skill_pulse(representation("order", seed + 10, 48)[0], 0, n_skills),
        wrap_skill_pulse(representation("xor", seed + 20, 48)[0], 1, n_skills),
        wrap_skill_pulse(strict_binding(seed + 30, 48)[0], 2, n_skills),
    ]
    model.calibrate(np.concatenate(cal_batches, axis=0))

    # 2. Prepare raw datasets for all 3 skills
    raw_data = {
        0: (
            "order",
            wrap_skill_pulse(representation("order", seed + 100, 768)[0], 0, n_skills),
            representation("order", seed + 100, 768)[1],
            wrap_skill_pulse(representation("order", seed + 1100, 256)[0], 0, n_skills),
            representation("order", seed + 1100, 256)[1],
        ),
        1: (
            "xor",
            wrap_skill_pulse(representation("xor", seed + 200, 1024)[0], 1, n_skills),
            representation("xor", seed + 200, 1024)[1],
            wrap_skill_pulse(representation("xor", seed + 1200, 256)[0], 1, n_skills),
            representation("xor", seed + 1200, 256)[1],
        ),
        2: (
            "strict_binding",
            wrap_skill_pulse(strict_binding(seed + 300, 1024)[0], 2, n_skills),
            strict_binding(seed + 300, 1024)[1],
            wrap_skill_pulse(strict_binding(seed + 1300, 256)[0], 2, n_skills),
            strict_binding(seed + 1300, 256)[1],
        ),
    }

    # Order the stages according to order_indices
    stages = [raw_data[idx] for idx in order_indices]

    policy = "balanced" if mode == "ew_he_ca_msc" else "none"
    buffer = ReplayBuffer(256, 12, d_in, policy, seed + 7000)

    matrix = np.zeros((n_skills, n_skills), dtype=np.float64)
    for s_idx, (sname, trx, try_, _, _) in enumerate(stages):
        for seq, y in zip(trx, try_):
            model.learn_episode(seq, int(y), stat_rate=0.005)
            if mode == "ew_he_ca_msc" and buffer.size > 0:
                for r_seq, r_y in buffer.sample(1):
                    model.learn_episode(r_seq, int(r_y), stat_rate=0.005)
        if mode == "ew_he_ca_msc":
            model.snapshot_consolidation_anchors()
            for seq, y in zip(trx, try_):
                buffer.add(seq, int(y), s_idx)

        for eval_idx in range(s_idx + 1):
            _, _, _, tx, ty = stages[eval_idx]
            matrix[s_idx, eval_idx] = float(np.mean(model.predict_many(tx) == ty))

    diag = np.diag(matrix)
    final = matrix[-1, :]
    forg = [max(matrix[t, j] for t in range(j, n_skills)) - final[j] for j in range(n_skills)]
    return {
        "acquisition_accuracy": float(diag.mean()),
        "final_accuracy": float(final.mean()),
        "mean_forgetting": float(np.mean(forg)),
        "worst_forgetting": float(np.max(forg)),
        "stage_matrix": [[round(float(x), 4) for x in row] for row in matrix],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art_dir = root / "artifacts" / "gemini-r7"
    art_dir.mkdir(parents=True, exist_ok=True)

    permutations = list(itertools.permutations(range(3)))  # 6 orders
    print(f"Starting 18-cell paired-across-orders study: {len(DEV_SEEDS)} seeds x {len(permutations)} orders = 18 cells")

    cells_data: list[dict[str, object]] = []
    delta_forg_list: list[float] = []
    delta_acq_list: list[float] = []
    delta_final_list: list[float] = []

    start_t = time.perf_counter()

    for seed in DEV_SEEDS:
        for order_idx, p_tuple in enumerate(permutations):
            order_names = [SKILL_NAMES[i] for i in p_tuple]
            t0 = time.perf_counter()

            # Run Arm A: unconsolidated_no_replay
            res_unconsol = run_single_cell(seed, p_tuple, "unconsolidated_no_replay")
            # Run Arm B: ew_he_ca_msc
            res_ca_msc = run_single_cell(seed, p_tuple, "ew_he_ca_msc")

            elapsed = time.perf_counter() - t0

            d_forg = float(res_ca_msc["mean_forgetting"]) - float(res_unconsol["mean_forgetting"])
            d_acq = float(res_ca_msc["acquisition_accuracy"]) - float(res_unconsol["acquisition_accuracy"])
            d_final = float(res_ca_msc["final_accuracy"]) - float(res_unconsol["final_accuracy"])

            delta_forg_list.append(d_forg)
            delta_acq_list.append(d_acq)
            delta_final_list.append(d_final)

            cell_record = {
                "cell_id": f"seed_{seed}_order_{order_idx}",
                "seed": seed,
                "order": order_names,
                "order_tuple": list(p_tuple),
                "unconsolidated_no_replay": res_unconsol,
                "ew_he_ca_msc": res_ca_msc,
                "delta_mean_forgetting": d_forg,
                "delta_acquisition": d_acq,
                "delta_final_accuracy": d_final,
                "seconds": round(elapsed, 2),
            }
            cells_data.append(cell_record)

            print(
                f"[Cell {len(cells_data):2d}/18] seed={seed} order={'->'.join(order_names):35s} | "
                f"forg: {res_unconsol['mean_forgetting']:.4f} -> {res_ca_msc['mean_forgetting']:.4f} (d={d_forg:+.4f}) | "
                f"acq: {res_unconsol['acquisition_accuracy']:.4f} -> {res_ca_msc['acquisition_accuracy']:.4f} (d={d_acq:+.4f}) | "
                f"final: {res_unconsol['final_accuracy']:.4f} -> {res_ca_msc['final_accuracy']:.4f} (d={d_final:+.4f})",
                flush=True,
            )

    total_time = time.perf_counter() - start_t

    ci_forg = bootstrap_ci(delta_forg_list)
    ci_acq = bootstrap_ci(delta_acq_list)
    ci_final = bootstrap_ci(delta_final_list)

    ci_unconsol_forg = bootstrap_ci([c["unconsolidated_no_replay"]["mean_forgetting"] for c in cells_data])
    ci_ca_msc_forg = bootstrap_ci([c["ew_he_ca_msc"]["mean_forgetting"] for c in cells_data])
    ci_unconsol_acq = bootstrap_ci([c["unconsolidated_no_replay"]["acquisition_accuracy"] for c in cells_data])
    ci_ca_msc_acq = bootstrap_ci([c["ew_he_ca_msc"]["acquisition_accuracy"] for c in cells_data])
    ci_unconsol_final = bootstrap_ci([c["unconsolidated_no_replay"]["final_accuracy"] for c in cells_data])
    ci_ca_msc_final = bootstrap_ci([c["ew_he_ca_msc"]["final_accuracy"] for c in cells_data])

    cells_forg_improved = sum(1 for d in delta_forg_list if d < 0.0)
    cells_final_improved = sum(1 for d in delta_final_list if d > 0.0)
    cells_acq_improved = sum(1 for d in delta_acq_list if d > 0.0)

    # Order variance in baseline and consolidated
    within_seed_order_sd_unconsol = float(
        np.mean([np.std([c["unconsolidated_no_replay"]["mean_forgetting"] for c in cells_data if c["seed"] == s], ddof=1) for s in DEV_SEEDS])
    )
    within_seed_order_sd_ca_msc = float(
        np.mean([np.std([c["ew_he_ca_msc"]["mean_forgetting"] for c in cells_data if c["seed"] == s], ddof=1) for s in DEV_SEEDS])
    )

    summary = {
        "n_cells": 18,
        "dev_seeds": list(DEV_SEEDS),
        "total_seconds": round(total_time, 2),
        "within_seed_order_sd_unconsolidated": within_seed_order_sd_unconsol,
        "within_seed_order_sd_ca_msc": within_seed_order_sd_ca_msc,
        "paired_contrasts": {
            "mean_forgetting_delta": {
                **ci_forg,
                "cells_improved": f"{cells_forg_improved}/18",
                "excludes_zero": bool(ci_forg["upper95"] < 0.0),
            },
            "acquisition_delta": {
                **ci_acq,
                "cells_improved": f"{cells_acq_improved}/18",
                "excludes_zero": bool(ci_acq["lower95"] > 0.0 or ci_acq["upper95"] < 0.0),
            },
            "final_accuracy_delta": {
                **ci_final,
                "cells_improved": f"{cells_final_improved}/18",
                "excludes_zero": bool(ci_final["lower95"] > 0.0),
            },
        },
        "absolute_levels": {
            "unconsolidated_no_replay": {
                "acquisition": ci_unconsol_acq,
                "mean_forgetting": ci_unconsol_forg,
                "final_accuracy": ci_unconsol_final,
            },
            "ew_he_ca_msc": {
                "acquisition": ci_ca_msc_acq,
                "mean_forgetting": ci_ca_msc_forg,
                "final_accuracy": ci_ca_msc_final,
            },
        },
        "cells": cells_data,
    }

    out_json = art_dir / "paired-orders-18cell.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote 18-cell paired results -> {out_json}")
    print(f"Mean Forgetting Delta: {ci_forg['mean']:+.4f} (95% CI [{ci_forg['lower95']:+.4f}, {ci_forg['upper95']:+.4f}]) | {cells_forg_improved}/18 cells")
    print(f"Acquisition Delta    : {ci_acq['mean']:+.4f} (95% CI [{ci_acq['lower95']:+.4f}, {ci_acq['upper95']:+.4f}]) | {cells_acq_improved}/18 cells")
    print(f"Final Accuracy Delta : {ci_final['mean']:+.4f} (95% CI [{ci_final['lower95']:+.4f}, {ci_final['upper95']:+.4f}]) | {cells_final_improved}/18 cells")


if __name__ == "__main__":
    main()
