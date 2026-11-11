"""R7 round 23 — what distinguishes the 8 cells that reach 3pp? (analysis of existing data)

NO NEW RUNS. This analyses the 60 cells already produced by round 22
(`artifacts/deepseek-r7/gate-reachability-60.json`) to find a predictor of which cells meet
the mission's mean-forgetting gate.

Round 22 found 8/60 cells at or below 3pp under interleaved replay. If those 8 share a
structural property, the variance is controllable and the gate is a solvable engineering
problem. If they do not, 8/60 is a draw from a continuous distribution and nothing short of
more data or a different mechanism will change it.

CANDIDATE PREDICTORS (all available per cell, none requires a new run)
  order            which task comes first / last
  seed             the initialisation
  no_replay_level  how much forgetting the SAME cell has without replay (difficulty proxy)
  base_acquisition how well the cell learns at all
  base_final       absolute final accuracy

TEST
  For each candidate: mean interleaved forgetting conditional on each value, plus the
  overlap of the <=3pp set with each level.

DISCRIMINATION
  If a single order achieves the gate across most seeds -> ORDER is the lever and a fixed
  protocol order is justified.
  If it is scattered across orders and seeds -> the gate is a tail event.
  If no_replay_level correlates strongly -> the mechanism's benefit is largest where the
  baseline is worst, and the gate is reached where the baseline happens to be benign.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "deepseek-r7" / "gate-reachability-60.json"


def corr(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xs.std() < 1e-12 or ys.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    cells = d["cells"]
    print("=" * 96)
    print("R7 ROUND 23 — what distinguishes the 8 gate-reaching cells?")
    print("=" * 96)
    print(f"  source: {SRC.name}  ({len(cells)} cells, no new runs)\n")

    reached = [c for c in cells if c["interleaved"]["meanF"] <= 3.0]
    missed = [c for c in cells if c["interleaved"]["meanF"] > 3.0]
    print(f"  cells at or below 3pp: {len(reached)}/60")
    print(f"  their interleaved meanF: "
          + ", ".join(f"{c['interleaved']['meanF']:+.2f}" for c in
                      sorted(reached, key=lambda c: c["interleaved"]["meanF"])))

    # ---- predictor 1: order
    print("\n" + "-" * 96)
    print("PREDICTOR 1 — task order")
    print("-" * 96)
    by_order = defaultdict(list)
    for c in cells:
        by_order[" > ".join(o[:4] for o in c["order"])].append(c["interleaved"]["meanF"])
    print(f"  {'order':32s} {'mean':>8} {'sd':>7} {'min':>8} {'<=3pp':>7}")
    order_stats = {}
    for k in sorted(by_order, key=lambda k: np.mean(by_order[k])):
        v = np.array(by_order[k])
        n = int((v <= 3).sum())
        order_stats[k] = {"mean": float(v.mean()), "n_reached": n, "n": len(v)}
        print(f"  {k:32s} {v.mean():8.2f} {v.std(ddof=1):7.2f} "
              f"{v.min():8.2f} {n:3d}/{len(v):<3d}")
    best_order = min(order_stats, key=lambda k: order_stats[k]["mean"])
    print(f"\n  best order: {best_order} "
          f"(mean {order_stats[best_order]['mean']:.2f}pp, "
          f"{order_stats[best_order]['n_reached']}/10 seeds reach 3pp)")
    concentrated = order_stats[best_order]["n_reached"] >= 5
    print(f"  order concentrates the gate: {concentrated}")

    # ---- predictor 2: seed
    print("\n" + "-" * 96)
    print("PREDICTOR 2 — seed")
    print("-" * 96)
    by_seed = defaultdict(list)
    for c in cells:
        by_seed[c["seed"]].append(c["interleaved"]["meanF"])
    seeds_ok = [s for s in by_seed if sum(1 for v in by_seed[s] if v <= 3) >= 3]
    print(f"  seeds reaching the gate in >=3 of their 6 orders: {len(seeds_ok)}/10")
    for s in sorted(by_seed):
        v = np.array(by_seed[s])
        n = int((v <= 3).sum())
        print(f"    seed {s}  mean {v.mean():7.2f}  min {v.min():7.2f}  "
              f"<=3pp {n}/6")

    # ---- predictor 3: baseline difficulty
    print("\n" + "-" * 96)
    print("PREDICTOR 3 — does the mechanism help most where the baseline is worst?")
    print("-" * 96)
    base = [c["no_replay"]["meanF"] for c in cells]
    inter = [c["interleaved"]["meanF"] for c in cells]
    delta = [c["d_meanF"] for c in cells]
    print(f"  corr(no_replay meanF, interleaved meanF) = {corr(base, inter):+.4f}")
    print(f"  corr(no_replay meanF, delta)             = {corr(base, delta):+.4f}")
    rb = np.mean([c["no_replay"]["meanF"] for c in reached])
    mb = np.mean([c["no_replay"]["meanF"] for c in missed])
    print(f"  mean no_replay level | reached cells : {rb:7.2f}pp")
    print(f"  mean no_replay level | missed cells  : {mb:7.2f}pp")
    print(f"  difference: {rb - mb:+.2f}pp  "
          f"({'reached cells had EASIER baselines' if rb < mb else 'reached cells had HARDER baselines'})")

    # ---- predictor 4: acquisition / final
    print("\n" + "-" * 96)
    print("PREDICTOR 4 — acquisition and final accuracy")
    print("-" * 96)
    for key, label in (("acq", "acquisition"), ("final", "final accuracy")):
        vals = [c["interleaved"][key] for c in cells]
        print(f"  corr({label}, interleaved meanF) = {corr(vals, inter):+.4f}")
        vr = np.mean([c["interleaved"][key] for c in reached])
        vm = np.mean([c["interleaved"][key] for c in missed])
        print(f"    reached {vr:.4f}   missed {vm:.4f}   diff {vr-vm:+.4f}")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    spread_reached = sorted(set(c["seed"] for c in reached))
    orders_reached = sorted(set(" > ".join(o[:4] for o in c["order"]) for c in reached))
    print(f"  reached cells span {len(spread_reached)} distinct seeds and "
          f"{len(orders_reached)} distinct orders")
    print(f"  order that concentrates most: {best_order} "
          f"({order_stats[best_order]['n_reached']}/10 seeds)")
    print(f"  baseline difficulty difference: {rb - mb:+.2f}pp")

    if concentrated:
        print("\n  => ORDER is a controllable lever. A fixed protocol putting "
              f"{best_order.split(' > ')[0]} first reaches the gate on "
              f"{order_stats[best_order]['n_reached']}/10 seeds.")
        verdict = "order_lever"
    elif abs(corr(base, delta)) > 0.5:
        print("\n  => The benefit concentrates where the baseline is worst, so the "
              "gate is reached\n     where the baseline happens to be benign. Not a "
              "controllable property.")
        verdict = "baseline_tail"
    else:
        print("\n  => The 8 reaching cells are scattered across seeds and orders with")
        print("     no structural predictor. 8/60 is a tail draw from a continuous")
        print("     distribution, and no protocol choice concentrates it.")
        verdict = "tail_event"

    out = ROOT / "artifacts" / "deepseek-r7" / "gate-predictors.json"
    out.write_text(json.dumps({
        "n_cells": len(cells), "n_reached": len(reached),
        "reached_values": sorted(round(c["interleaved"]["meanF"], 2) for c in reached),
        "order_stats": order_stats,
        "seeds_with_3plus": seeds_ok,
        "corr_base_inter": corr(base, inter),
        "corr_base_delta": corr(base, delta),
        "baseline_reached_mean": float(rb), "baseline_missed_mean": float(mb),
        "verdict": verdict,
        "analysis_of": SRC.name, "new_runs": 0,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: {out.name}")


if __name__ == "__main__":
    sys.exit(main())
