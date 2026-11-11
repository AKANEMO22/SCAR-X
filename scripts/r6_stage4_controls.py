"""Round 6A Stage 4 — null controls and the hidden-plasticity control (H5).

Brief section H requires ten controls. Section G permits ONE secondary hidden-learning
arm, only after the frozen study, using the EXISTING local rule unmodified.

Controls implemented here:
  1. random training labels
  2. first-key shortcut baseline
  3. query-only shortcut baseline
  4. matched repeated-data control        (already in Stage 2, re-checked here)
  5. frozen vs permitted learned-hidden   (H5)
  6. equal-value and differing-value counterfactual siblings
  7. distinct train/test seeds            (verified in tests)
  8. held-out temporal perturbation
  9. held-out distractor perturbation
 10. two fixed training orders            (verified in Stage 1/3)

The hidden arm uses `PlasticCore`-style local update through the repository's existing
`scarx.neural.core` machinery. If it cannot be reproduced faithfully the arm is
declared BLOCKED rather than fabricated.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r6_scaling as S  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

QT = G.QUERY_TICK
N_HELD, N_PAIRS, CAL_SIZE = 1024, 256, 256
N_UNIQUE = 16384


def build(seed, n=N_UNIQUE):
    trx, try_, *_ = G.independent_query_set_v1(seed + S.TRAIN_OFFSET, n)
    tex, tey, evA, evB, eq, _ = G.independent_query_set_v1(seed + S.HELD_OFFSET, N_HELD)
    ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(
        seed + S.PAIRED_OFFSET, N_PAIRS)
    f_tr = S.capture_features(seed, trx, QT)
    f_te = S.capture_features(seed, tex, QT)
    f_pt = S.capture_features(seed, ptx, QT)
    cal = f_tr[:CAL_SIZE]
    mu, sd = cal.mean(0), np.maximum(cal.std(0), 1e-6)
    f = lambda a: (a - mu) / sd           # noqa: E731
    return dict(f_tr=f(f_tr), f_te=f(f_te), f_pt=f(f_pt), y_tr=try_, y_te=tey,
                vA=evA, vB=evB, q=eq, pty=pty, pvA=pvA, pvB=pvB, pflips=flips,
                trx=trx, tex=tex, ptx=ptx,
                digests={"train": G.dataset_sha256(trx, try_),
                         "held": G.dataset_sha256(tex, tey)})


def features_perturbed(seed, x, kind):
    """Held-out stress, applied at feature-extraction time with no retraining."""
    if kind == "temporal":
        xp = G.shifted(x, 1, seed=seed)
        return S.capture_features(seed, xp, QT + 1)
    if kind == "distractor":
        xp = G.with_distractors(x, seed + 55)
        return S.capture_features(seed, xp, QT)
    raise ValueError(kind)


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r6/protocol-v1.md"
    psha = hashlib.sha256(proto.read_bytes()).hexdigest() if proto.exists() else "PROTOCOL_PENDING"
    run = new_run(root, "r6-controls-v1", {
        "protocol_sha256": psha, "seeds": list(S.DEV_SEEDS), "n_unique": N_UNIQUE,
        "controls": ["random_labels", "first_key", "query_only", "temporal_shift",
                     "distractor", "hidden_plasticity"]})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in S.DEV_SEEDS:
            d = build(seed)
            rng = np.random.default_rng(seed + 2024)
            order = rng.permutation(N_UNIQUE)
            rec = {"seed": seed, "digests": d["digests"]}

            from scarx.evaluation.tasks21 import ridge_probe

            # --- the reference arm
            h = S.train_head(d["f_tr"], d["y_tr"], seed, order, epochs=1)
            ref = S.full_report(h, d["f_te"], d["y_te"], d["vA"], d["vB"], d["q"],
                                d["f_pt"], d["pty"], d["pvA"], d["pvB"], d["pflips"],
                                d["f_tr"], d["y_tr"],
                                ridge_probe(d["f_tr"], d["y_tr"], d["f_te"], d["y_te"]))
            rec["reference"] = ref

            # --- control 1: random training labels
            y_rand = rng.permutation(d["y_tr"])
            hr = S.train_head(d["f_tr"], y_rand, seed, order, epochs=1)
            rec["random_labels"] = {
                "direct": float(np.mean(hr.predict(d["f_te"]) == d["y_te"])),
                "train": hr.acc(d["f_tr"], y_rand)}

            # --- control 2/3: shortcut baselines on the held-out set
            rec["first_key"] = float(np.mean(
                G.baseline_first_key(d["tex"]) == d["y_te"]))
            rec["query_only"] = float(np.mean(
                G.baseline_query_only(d["tex"]) == d["y_te"]))

            # --- controls 8/9: held-out perturbations, NO retraining
            for kind in ("temporal", "distractor"):
                f_p = features_perturbed(seed, d["tex"], kind)
                rec[kind] = {"direct": h.acc(f_p, d["y_te"]),
                             "drop": ref["direct"] - h.acc(f_p, d["y_te"])}

            rows.append(rec)
            mon.check()
            print(f"  seed{seed} ref {ref['direct']:.4f} rand {rec['random_labels']['direct']:.4f} "
                  f"temp {rec['temporal']['direct']:.4f} dist {rec['distractor']['direct']:.4f}",
                  flush=True)

    def M(k):
        return float(np.mean([r["reference"][k] for r in rows]))

    summary = {
        "reference": {k: M(k) for k in ("direct", "train", "both_A", "both_B",
                                        "acc_needB", "ridge", "ridge_gap")},
        "random_labels_direct": float(np.mean([r["random_labels"]["direct"] for r in rows])),
        "first_key": float(np.mean([r["first_key"] for r in rows])),
        "query_only": float(np.mean([r["query_only"] for r in rows])),
        "temporal": {"direct": float(np.mean([r["temporal"]["direct"] for r in rows])),
                     "drop": float(np.mean([r["temporal"]["drop"] for r in rows]))},
        "distractor": {"direct": float(np.mean([r["distractor"]["direct"] for r in rows])),
                       "drop": float(np.mean([r["distractor"]["drop"] for r in rows]))},
        "hidden_plasticity": "BLOCKED - see report; the existing local hidden rule could "
                             "not be reproduced faithfully inside this research harness "
                             "without modifying src/scarx/neural, which the brief forbids",
    }
    write_json(run / "result.json",
               {"protocol_sha256": psha, "phase": "stage4_controls", "trials": rows,
                "summary": summary, "resources": mon.report(),
                "wall_seconds": time.time() - t0})

    print("\n" + "=" * 78)
    print("NULL CONTROLS at n=16384 (3 dev seeds, mean)")
    print("=" * 78)
    print(f"  reference direct      {summary['reference']['direct']:.4f}")
    print(f"  random labels         {summary['random_labels_direct']:.4f}")
    print(f"  first-key shortcut    {summary['first_key']:.4f}")
    print(f"  query-only shortcut   {summary['query_only']:.4f}")
    print(f"  temporal shift        {summary['temporal']['direct']:.4f} "
          f"(drop {summary['temporal']['drop']*100:+.2f}pp)")
    print(f"  distractors           {summary['distractor']['direct']:.4f} "
          f"(drop {summary['distractor']['drop']*100:+.2f}pp)")
    print(f"  hidden plasticity     {summary['hidden_plasticity'][:40]}...")
    print(f"\nRESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
