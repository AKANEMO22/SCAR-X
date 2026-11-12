"""SCAR-X Round 8: Multi-Key Associative Memory & Context-Gated Readout Experiment.

Role: Sub-Agent C (Readout, Associative Memory & Retention)
Mandate:
1. Test associative scaling across n_keys in {2, 4, 8}.
2. Test under delays {4, 8 ticks} and mid-delay distractors.
3. Compare:
   - Standard Linear Readout (shared linear projection)
   - Context-Gated Readout (query context-conditioned projection)
   - CKCER-style key-conditioned gating
under matched parameter and compute budgets.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from scarx.neural.core import Config
from scarx.neural.r8_bf_cascade import BennaFusiMetaplasticCascade
from scarx.evaluation.r8_benchmark_integrity import generate_multikey_associative, assert_input_dimension

DEV_SEEDS = [140011, 140017, 140023]
KEY_COUNTS = [2, 4, 8]
DELAYS = [4, 8]


def run_multikey_experiment():
    print("=== SCAR-X ROUND 8: MULTI-KEY ASSOCIATIVE MEMORY SCALING ===")
    results: dict[str, dict] = {}

    for n_keys in KEY_COUNTS:
        for delay in DELAYS:
            for distractor in (False, True):
                case_name = f"keys_{n_keys}_delay_{delay}_distr_{distractor}"
                print(f"\n--- Testing Case: {case_name} ---")

                standard_accs = []
                context_accs = []

                for seed in DEV_SEEDS:
                    tr_x, tr_y = generate_multikey_associative(seed + 100, 512, n_keys=n_keys, delay=delay, distractor=distractor)
                    te_x, te_y = generate_multikey_associative(seed + 200, 256, n_keys=n_keys, delay=delay, distractor=distractor)
                    assert_input_dimension(tr_x, 8)

                    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)

                    # Arm 1: Standard Linear Readout (Context Gating OFF)
                    model_std = BennaFusiMetaplasticCascade(cfg, use_cascade=True, use_context_gating=False)
                    model_std.calibrate(tr_x[:16])
                    for seq, y in zip(tr_x, tr_y):
                        model_std.learn_episode(seq, int(y))
                    p_std = float(np.mean(model_std.predict_many(te_x) == te_y))
                    standard_accs.append(p_std)

                    # Arm 2: Context-Gated Readout (Query-Conditioned ON)
                    model_ctx = BennaFusiMetaplasticCascade(cfg, use_cascade=True, use_context_gating=True)
                    model_ctx.calibrate(tr_x[:16])
                    for seq, y in zip(tr_x, tr_y):
                        model_ctx.learn_episode(seq, int(y))
                    p_ctx = float(np.mean(model_ctx.predict_many(te_x) == te_y))
                    context_accs.append(p_ctx)

                    print(f"Seed {seed}: Standard={p_std*100:.2f}%, ContextGated={p_ctx*100:.2f}%")

                results[case_name] = {
                    "n_keys": n_keys,
                    "delay": delay,
                    "distractor": distractor,
                    "standard_mean": float(np.mean(standard_accs)),
                    "standard_sd": float(np.std(standard_accs, ddof=1)),
                    "context_mean": float(np.mean(context_accs)),
                    "context_sd": float(np.std(context_accs, ddof=1)),
                    "context_gain_pp": float((np.mean(context_accs) - np.mean(standard_accs)) * 100),
                }

    out_path = Path("artifacts/gemini-r8/multikey-associative.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== MULTI-KEY ASSOCIATIVE EXPERIMENT SUMMARY ===")
    for case, data in results.items():
        print(
            f"{case:32s}: Standard={data['standard_mean']*100:.2f}%, "
            f"ContextGated={data['context_mean']*100:.2f}%, "
            f"Gain={data['context_gain_pp']:+.2f} pp"
        )
    print(f"Artifact written to: {out_path.resolve()}")


if __name__ == "__main__":
    run_multikey_experiment()
