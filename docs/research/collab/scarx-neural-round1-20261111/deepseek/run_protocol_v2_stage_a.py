"""Stage A of protocol v2, all registered dev seeds. H1/H3.

Registered seeds: 71001, 71007, 71013. 5 arms (raw, zscore, rms + two frozen
controls) x 3 seeds. Gate G1: conditioned head within 5pp of the ridge on the
same frozen hidden, on every dev seed.

Nothing here is tuned. If G1 fails, Stage B is not run (protocol section 9).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation.conditioned_head import ConditionedCore, sim_xor
from scarx.evaluation.tasks21 import ridge_probe

SEEDS = (71001, 71007, 71013)      # registered in protocol v2 section 5
HEADS = ("raw", "zscore", "rms")
N_TRAIN = 400
PASSES = 3


def run_arm(seed, head, hidden_learns):
    fx, fy = sim_xor(seed + 100_000, 128)
    trx, try_ = sim_xor(seed + 200_000, N_TRAIN)
    tx, ty = sim_xor(seed + 300_000, 300)

    m = ConditionedCore(seed, head=head, signal="scalar" if hidden_learns else "frozen")
    ridge_same = ridge_probe(m.frozen_features(fx), fy, m.frozen_features(tx), ty)
    w_before = m.w_in.copy()

    losses = []
    for _ in range(PASSES):
        for seq, lab in zip(trx, try_):
            m.run(seq, track=hidden_learns)
            l, _ = m.learn(int(lab))
            losses.append(l)

    head_acc = m.head_accuracy(tx, ty)
    moved = not np.array_equal(m.w_in, w_before)
    probe = ridge_probe(m.frozen_features(fx), fy, m.frozen_features(tx), ty)
    return {"seed": seed, "head": head, "hidden_learns": hidden_learns,
            "ridge_same_features": ridge_same, "head_acc": head_acc,
            "gap": head_acc - ridge_same, "hidden_moved": bool(moved),
            "probe_after": probe, "mean_loss": float(np.mean(losses))}


def main():
    print("=" * 78)
    print("STAGE A — protocol v2, registered dev seeds 71001/71007/71013")
    print("G1: conditioned head within 5pp of ridge on the SAME frozen hidden")
    print("=" * 78)
    rows = []
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        for head in HEADS:
            r = run_arm(seed, head, hidden_learns=False)
            rows.append(r)
            print(f"  {head:8s} frozen-hidden  ridge={r['ridge_same_features']:.4f} "
                  f"head={r['head_acc']:.4f}  gap={r['gap']*100:+6.2f}pp")
        # H3: does conditioning itself move the hidden?
        for head in ("raw", "zscore"):
            r = run_arm(seed, head, hidden_learns=True)
            rows.append(r)
            print(f"  {head:8s} hidden-learn  head={r['head_acc']:.4f} "
                  f"hidden_moved={r['hidden_moved']} loss={r['mean_loss']:.4f}")

    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    gaps = {}
    for head in HEADS:
        v = [r["gap"] for r in rows if r["head"] == head and not r["hidden_learns"]]
        gaps[head] = {"mean_pp": float(np.mean(v) * 100),
                      "worst_pp": float(np.min(v) * 100),
                      "per_seed": [round(x * 100, 2) for x in v]}
        print(f"  {head:8s} gap per seed {gaps[head]['per_seed']}  "
              f"worst {gaps[head]['worst_pp']:+.2f}pp")

    g1 = any(gaps[h]["worst_pp"] >= -5.0 for h in ("zscore", "rms"))
    print(f"\n  G1 (H1): conditioned head within 5pp of ridge on every seed -> {g1}")

    frozen_probe = [r["probe_after"] for r in rows
                    if r["head"] == "zscore" and not r["hidden_learns"]]
    learn_probe = [r["probe_after"] for r in rows
                   if r["head"] == "zscore" and r["hidden_learns"]]
    print(f"  probe frozen-head arm : {np.mean(frozen_probe):.4f}")
    print(f"  probe learning arm    : {np.mean(learn_probe):.4f}")

    out = {"protocol": "v2", "stage": "A", "seeds": list(SEEDS),
           "trials": rows, "gaps": gaps, "G1_pass": bool(g1),
           "stage_b_run": False,
           "note": ("Stage B not run: G1 failed. Protocol section 9 requires "
                    "stopping when H1 fails.")}
    p = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r1"
    p.mkdir(parents=True, exist_ok=True)
    (p / "protocol-v2-stage-a.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nwritten: {p / 'protocol-v2-stage-a.json'}")
    print(f"VERDICT: H1 {'PASS' if g1 else 'FAIL'} -> Stage B "
          f"{'opened' if g1 else 'NOT run'}")


if __name__ == "__main__":
    main()
