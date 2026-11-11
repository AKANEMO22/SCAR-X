"""R7 round 36 — is the T=6 wall FEATURE COLLAPSE? Measure task separability directly.

ROUND 34-35 CONCLUDED "features"
--------------------------------
    T=6: every head form pays ~31pp; solo 0.97 -> joint 0.66
Round 35 tested the effective-rank path and it failed (rank rises while joint falls).
So "the features" is a placeholder, not a mechanism. This round measures the features
themselves.

THE COMPETING EXPLANATIONS
--------------------------
H-COLLAPSE   The 48 neurons produce nearly the SAME feature vector for different tasks, so
              one boundary set cannot separate them. Predicts: between-task feature distance
              is small relative to within-task spread, and a classifier trained on one task's
              features can partially answer another task's labels above chance by accident
              (because the codes overlap).
H-SCATTER    The tasks produce well-separated but HIGH-DIMENSIONAL, overlapping codes — each
              task is separable alone but their regions interpenetrate in the union.
              Predicts: large between-task centroid distance, but heavy overlap of per-class
              distributions.
H-LABEL      The tasks are individually learnable (solo 0.97) yet their LABEL functions
              conflict on the shared features. Predicts: two tasks have near-identical
              feature distributions but opposite labels on the same points.

MEASUREMENTS (frozen features, 8 seeds, T=6)
    centroid distance      ||mean(F_a) - mean(F_b)|| per task pair
    normalized separation  centroid distance / pooled within-task sd, per pair
    label agreement        on nearest-neighbour pairs ACROSS tasks, how often do labels
                           agree? 0.5 = no conflict, 1.0 = identical function,
                           0.0 = perfectly opposed
    per-task 2-NN purity   does each task occupy its own region of feature space?

H-LABEL is the sharpest: if two tasks are individually learnable but their 2-NN label
agreement is far from 0.5, their label functions are in conflict on shared features and no
single head can hold both.

Cheap: reuses round 32's features. 8 seeds, no head training needed except one probe.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scarx.evaluation import r7_secs as S  # noqa: E402
import r7_r32_tier3_six as R32  # noqa: E402

SEEDS = tuple(320001 + 6 * k for k in range(8))
TASKS = R32.TASKS
N_TR, N_TE = 320, 200


def main():
    t0 = time.time()
    print("=" * 98)
    print("R7 ROUND 36 — is the T=6 wall feature COLLAPSE, SCATTER, or LABEL CONFLICT?")
    print("=" * 98)
    print(f"  6 tasks, 8 seeds, frozen features, no head training\n")

    pair_stats = {}
    label_agree = {}
    for s_i, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        F, Y = {}, {}
        for tk in TASKS:
            trx, try_ = R32.GENS[tk](seed + 200_000, N_TR)
            F[tk] = R32.feats(core, trx); Y[tk] = try_

        for a, b in itertools.combinations(TASKS, 2):
            Fa, Fb = F[a], F[b]
            mu_a, mu_b = Fa.mean(0), Fb.mean(0)
            sd = np.sqrt(0.5 * (Fa.var(0).mean() + Fb.var(0).mean())) + 1e-9
            dist = float(np.linalg.norm(mu_a - mu_b))
            sep = dist / sd
            key = f"{a}|{b}"
            pair_stats.setdefault(key, []).append((dist, sep))

            # cross-task 2-NN label agreement: for each point in a, find its nearest
            # neighbour in b and compare LABELS
            d2 = ((Fa[:, None, :] - Fb[None, :, :]) ** 2).sum(-1)
            nn = d2.argmin(1)
            agree = float(np.mean(Y[a] == Y[b][nn]))
            label_agree.setdefault(key, []).append(agree)
        if (s_i + 1) % 4 == 0:
            print(f"  seed {s_i+1}/8 done ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 98)
    print("PAIRWISE FEATURE SEPARATION (mean over 8 seeds)")
    print("=" * 98)
    print(f"  {'pair':28s} {'centroid dist':>14} {'normalised':>11} {'2NN label agree':>16}")
    rows = []
    for key in sorted(pair_stats, key=lambda k: -np.mean([s for _, s in pair_stats[k]])):
        d = np.mean([x for x, _ in pair_stats[key]])
        s = np.mean([x for _, x in pair_stats[key]])
        ag = np.mean(label_agree[key])
        rows.append({"pair": key, "dist": float(d), "sep": float(s), "agree": float(ag)})
        print(f"  {key:28s} {d:14.4f} {s:11.4f} {ag:16.3f}")

    seps = np.array([r["sep"] for r in rows])
    ags = np.array([r["agree"] for r in rows])
    print(f"\n  normalised separation: mean {seps.mean():.3f}  "
          f"min {seps.min():.3f}  max {seps.max():.3f}")
    print(f"  2-NN label agreement : mean {ags.mean():.3f}  "
          f"min {ags.min():.3f}  max {ags.max():.3f}")
    print(f"  (0.5 = no conflict, 1.0 = identical function, 0.0 = perfectly opposed)")

    print("\n" + "=" * 98)
    print("DIAGNOSIS")
    print("=" * 98)
    collapsed = seps.mean() < 0.5
    conflicting = np.abs(ags - 0.5).mean() > 0.15
    print(f"  separation {'COLLAPSED' if collapsed else 'PRESENT'} "
          f"(mean {seps.mean():.3f})")
    print(f"  label functions {'CONFLICTING' if conflicting else 'COMPATIBLE'} "
          f"(mean |agree-0.5| = {np.abs(ags-0.5).mean():.3f})")

    if conflicting:
        print(f"\n  => LABEL CONFLICT. Individually learnable tasks (solo 0.97) have label")
        print(f"     functions that disagree on the SAME feature points. Nearest-neighbour")
        print(f"     pairs across tasks agree on their labels only {ags.mean()*100:.1f}% of")
        print(f"     the time against 50% for no relationship. A single head on these")
        print(f"     features must therefore fit contradictory constraints.")
        verdict = "label_conflict"
    elif collapsed:
        print(f"\n  => FEATURE COLLAPSE. The population produces similar codes for")
        print(f"     different tasks; the readout has nothing to separate.")
        verdict = "collapse"
    else:
        print(f"\n  => SCATTER. Tasks are separated but their regions interpenetrate in the")
        print(f"     union. A head with enough capacity should manage; none did, so this")
        print(f"     points at the union geometry rather than at individual task codes.")
        verdict = "scatter"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t6-feature-diagnosis.json").write_text(json.dumps({
        "tasks": list(TASKS), "seeds": list(SEEDS),
        "pairs": rows,
        "separation_mean": float(seps.mean()),
        "label_agreement_mean": float(ags.mean()),
        "verdict": verdict,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: t6-feature-diagnosis.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
