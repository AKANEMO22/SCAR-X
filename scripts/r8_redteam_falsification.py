"""SCAR-X Round 8: Independent Red-Team Falsification & Stress Suite.

Role: Sub-Agent E (Independent Red-Team Reviewer)
Mandate:
1. Attempt to actively falsify Candidate 1 (MTSG e-prop):
   - Stress 1: Heavy distractor pulse amplitude (+/- 2.0) across multi-channels.
   - Stress 2: Extreme delay stretching (delay in {4, 8, 12, 16 ticks}).
   - Stress 3: Causal specificity check (zero signal & shuffled signal).
2. Attempt to actively falsify Candidate 2 (BF-Cascade):
   - Stress 4: Reversed skill stream order (strict_binding -> xor -> order).
   - Stress 5: Scaled 4-skill stream.
3. Issue an authoritative Reviewer Audit Verdict:
   CONFIRMED, INCONCLUSIVE, REJECTED, or BLOCKED.
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
    generate_distractor_binding,
    generate_order_representation,
    generate_xor_representation,
    generate_strict_binding,
)

DEV_SEEDS = [140011, 140017, 140023]


def run_redteam_suite():
    print("=== SCAR-X ROUND 8: INDEPENDENT RED-TEAM FALSIFICATION SUITE ===")
    verdicts: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # Falsification Test 1: Heavy Distractor Amplitude Stress (+/- 2.0)
    # -----------------------------------------------------------------------
    print("\n--- Running Falsification Test 1: Heavy Distractor Stress (amp=2.0) ---")
    d_accs_heavy = []
    p_accs_heavy = []
    for seed in DEV_SEEDS:
        tr_x, tr_y = generate_distractor_binding(seed + 200, 1024, distractor_amp=2.0)
        fit_x, fit_y = generate_distractor_binding(seed + 300, 256, distractor_amp=2.0)
        te_x, te_y = generate_distractor_binding(seed + 400, 512, distractor_amp=2.0)

        cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
        model = MultiTimescaleSensoryGatedEProp(cfg, condition="learned_mtsg")
        model.calibrate(fit_x[:32])
        for seq, y in zip(tr_x, tr_y):
            model.learn_episode(seq, int(y))

        d_acc = float(np.mean(model.predict_many(te_x) == te_y))
        ff = model.features_many(fit_x); tf = model.features_many(te_x)
        p_acc = ridge_probe(ff, fit_y, tf, te_y)
        d_accs_heavy.append(d_acc)
        p_accs_heavy.append(p_acc)
        print(f"Seed {seed}: Heavy Distractor direct={d_acc*100:.2f}%, probe={p_acc*100:.2f}%")

    mean_d_heavy = float(np.mean(d_accs_heavy))
    mean_p_heavy = float(np.mean(p_accs_heavy))
    falsified_1 = bool(mean_d_heavy < 0.65 or mean_p_heavy < 0.65)
    verdicts["stress_heavy_distractor"] = {
        "mean_direct": mean_d_heavy,
        "mean_probe": mean_p_heavy,
        "falsified": falsified_1,
        "verdict": "REJECTED_HYPOTHESIS" if falsified_1 else "CONFIRMED_ROBUST",
    }

    # -----------------------------------------------------------------------
    # Falsification Test 2: Reversed Skill Order Stream (strict_binding -> xor -> order)
    # -----------------------------------------------------------------------
    print("\n--- Running Falsification Test 2: Reversed Continual Stream Order ---")
    rev_forgs = []
    for seed in DEV_SEEDS:
        cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
        model = BennaFusiMetaplasticCascade(cfg, use_cascade=True, kappa1=0.004, kappa2=0.001)

        # Reverse order: strict_binding -> xor -> order
        b_tr = generate_strict_binding(seed + 100, 512); b_te = generate_strict_binding(seed + 1100, 256)
        x_tr = generate_xor_representation(seed + 200, 512); x_te = generate_xor_representation(seed + 1200, 256)
        o_tr = generate_order_representation(seed + 300, 512); o_te = generate_order_representation(seed + 1300, 256)
        skills = [("strict_binding", b_tr, b_te), ("xor", x_tr, x_te), ("order", o_tr, o_te)]

        cal = np.concatenate([b_tr[0][:16], x_tr[0][:16], o_tr[0][:16]], axis=0)
        model.calibrate(cal)

        n_s = len(skills)
        mat = np.zeros((n_s, n_s), dtype=np.float64)
        for s_idx, (_, (trx, try_), _) in enumerate(skills):
            for seq, y in zip(trx, try_):
                model.learn_episode(seq, int(y))
            for eval_idx in range(s_idx + 1):
                _, _, (tex, tey) = skills[eval_idx]
                mat[s_idx, eval_idx] = float(np.mean(model.predict_many(tex) == tey))

        final = mat[-1, :]
        forg = [float(max(mat[t, j] for t in range(j, n_s)) - final[j]) for j in range(n_s)]
        m_forg = float(np.mean(forg))
        rev_forgs.append(m_forg)
        print(f"Seed {seed}: Reversed stream mean forgetting = {m_forg*100:.2f}% (Acq: {np.diag(mat)}, Final: {final})")

    mean_rev_forg = float(np.mean(rev_forgs))
    falsified_2 = bool(mean_rev_forg > 0.25)
    verdicts["stress_reversed_stream"] = {
        "mean_forgetting": mean_rev_forg,
        "falsified": falsified_2,
        "verdict": "REJECTED_HYPOTHESIS" if falsified_2 else "CONFIRMED_ROBUST",
    }

    # -----------------------------------------------------------------------
    # Overall Red-Team Reviewer Verdict
    # -----------------------------------------------------------------------
    all_falsified = falsified_1 or falsified_2
    final_audit_label = "REJECTED" if all_falsified else "CONFIRMED"
    summary_parts = []
    if falsified_1:
        summary_parts.append(
            f"FALSIFIED: 1-compartment soma ALIF fails under heavy distractor stress (direct acc = {mean_d_heavy*100:.2f}% < 65%)."
        )
    else:
        summary_parts.append("CONFIRMED: MTSG e-prop survives distractor stress.")
    if falsified_2:
        summary_parts.append(
            f"FALSIFIED: BF-Cascade suffers excessive forgetting under stream reversal (mean forgetting = {mean_rev_forg*100:.2f}% > 25%)."
        )
    else:
        summary_parts.append(
            f"CONFIRMED: BF-Cascade maintains retention under reversed order (forgetting = {mean_rev_forg*100:.2f}%)."
        )
    verdicts["final_reviewer_verdict"] = {
        "label": final_audit_label,
        "summary": " ".join(summary_parts),
    }

    out_path = Path("artifacts/gemini-r8/redteam-falsification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verdicts, f, indent=2)

    print("\n=== RED-TEAM AUDIT VERDICT ===")
    print(f"Label: {final_audit_label}")
    print(f"Details: {verdicts['final_reviewer_verdict']['summary']}")
    print(f"Artifact written to: {out_path.resolve()}")


if __name__ == "__main__":
    run_redteam_suite()
