"""R7 round 25 — which TASK collapses, and is it position or difficulty?

NO NEW RUNS. Analyses the 15 fresh-seed R matrices from round 24
(`artifacts/deepseek-r7/prereg-order-fresh15.json`).

ROUND 24 LEFT THIS
    meanF 3.96pp (8/15 seeds at gate) but worstF 5.86-24.22pp (0/15 seeds at gate).
So one task loses far more than the others inside each run. The mechanism is an
approximation of joint training; it protects the average and lets one task go.

TWO COMPETING EXPLANATIONS
  H-POSITION  The FIRST task learned absorbs the most subsequent updates, so its features and
              its decision region drift furthest. Predicts the worst task is always the same
              one — the first in the order, i.e. `order_short`.

  H-DIFFICULTY The task hardest to re-fit is the worst. Predicts the worst task VARIES by
              seed and tracks that task's own acquisition accuracy (the least well-fitted
              task is the one whose boundary is most fragile).

Both are readable from the data already stored: each cell has an R matrix, the order, and the
diagonal (per-task acquisition).

DISCRIMINATION
  Count how often each POSITION (1st/2nd/3rd) is the worst task, and how often each TASK is.
  Then correlate per-task forgetting with per-task acquisition.
  If position 1 dominates -> H-POSITION and the fix is positional (e.g. a short
  re-consolidation pass on the first task).
  If it tracks task identity or low acquisition -> H-DIFFICULTY and the fix is per-task
  weighting.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "deepseek-r7" / "prereg-order-fresh15.json"
ORDER = ("order_short", "parity3", "sim_xor")


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    cells = d["cells"]
    print("=" * 96)
    print("R7 ROUND 25 — which task collapses: position or difficulty?")
    print("=" * 96)
    print(f"  source: {SRC.name}  ({len(cells)} fresh seeds, no new runs)")
    print(f"  order : {' > '.join(ORDER)}\n")

    # R is indexed [trained_through][evaluated_on], both in ORDER
    worst_pos = defaultdict(int)
    worst_task = defaultdict(int)
    forget_by_pos = defaultdict(list)
    forget_by_task = defaultdict(list)
    acq_by_task = defaultdict(list)

    for c in cells:
        R = np.array(c["interleaved"].get("R", []), float) if "R" in c["interleaved"] \
            else None
        if R is None or R.size == 0:
            # R not stored per-cell in this artifact; recompute from stored fields
            R = None
        if R is None:
            continue

    if not worst_pos:
        print("  NOTE: this artifact does not carry per-cell R matrices; only aggregate")
        print("  meanF/worstF/acq/final. Re-deriving what IS available.\n")

    # What IS available per cell: meanF, worstF, acq (mean over tasks), final.
    # That is enough to test H-DIFFICULTY partially: does worstF track the cell's own acq?
    acq = np.array([c["interleaved"]["acq"] for c in cells])
    wf = np.array([c["interleaved"]["worstF"] for c in cells])
    mf = np.array([c["interleaved"]["meanF"] for c in cells])
    fin = np.array([c["interleaved"]["final"] for c in cells])

    print("-" * 96)
    print("AVAILABLE EVIDENCE (per-seed aggregates)")
    print("-" * 96)
    print(f"  corr(acquisition, worstF) = {corr(acq, wf):+.4f}")
    print(f"  corr(acquisition, meanF)  = {corr(acq, mf):+.4f}")
    print(f"  corr(final,       worstF) = {corr(fin, wf):+.4f}")
    print(f"  corr(meanF,       worstF) = {corr(mf, wf):+.4f}")

    print(f"\n  worstF - meanF gap (how concentrated the damage is):")
    gap = wf - mf
    print(f"    mean {gap.mean():.2f}pp   sd {gap.std(ddof=1):.2f}   "
          f"min {gap.min():.2f}   max {gap.max():.2f}")
    print(f"    With 3 tasks, forgetting_j = [f1, f2, 0] and")
    print(f"    meanF = (f1+f2)/3, worstF = max(f1,f2).")
    print(f"    So worstF - meanF = max(f1,f2) - (f1+f2)/3, which is")
    print(f"    >= (2/3)*max(f1,f2) >= (2/3)*worstF by construction.")

    print("\n" + "-" * 96)
    print("CAN worstF <= 5pp COEXIST WITH meanF <= 3pp IN THIS METRIC?")
    print("-" * 96)
    # Task 3 (last) has forgetting exactly 0 by construction. So with 2 measurable tasks:
    #   meanF = (f1+f2)/3 ; worstF = max(f1,f2)
    # meanF <= 3 => f1+f2 <= 9.  worstF <= 5 => max(f1,f2) <= 5.
    # Both are satisfiable: e.g. f1=f2=4.5 gives meanF=3.0, worstF=4.5. FEASIBLE.
    f1 = f2 = 4.5
    print(f"  Feasibility check with 2 measurable tasks out of 3:")
    print(f"    forgetting_j = [f1, f2, 0]   (last task is 0 BY CONSTRUCTION)")
    print(f"    meanF  = (f1+f2)/3")
    print(f"    worstF = max(f1, f2)")
    print(f"    meanF<=3 requires f1+f2<=9 ; worstF<=5 requires max<=5")
    print(f"    f1=f2={f1} gives meanF={(f1+f2)/3:.2f} and worstF={f2} -> "
          f"{'FEASIBLE' if (f1+f2)/3 <= 3 and f2 <= 5 else 'INFEASIBLE'}")
    print(f"\n  BUT the divisor is 3 while only 2 values can be non-zero.")
    print(f"  That is a metric inconsistency worth stating: with T tasks, the last")
    print(f"  contributes a structural zero to the mean but is never the max.")
    print(f"  My meanF is therefore understated by a factor of T/(T-1) = 1.5x")
    print(f"  relative to 'mean over measurable tasks'.")

    print("\n" + "-" * 96)
    print("RECOMPUTING meanF OVER MEASURABLE TASKS ONLY (x1.5)")
    print("-" * 96)
    mf_hon = mf * 3.0 / 2.0
    print(f"  {'seed':>8} {'meanF(÷3)':>11} {'meanF(÷2)':>11} {'worstF':>9} "
          f"{'gap':>8}")
    for c, a, b in zip(cells, mf, mf_hon):
        print(f"  {c['seed']:>8} {a:10.2f} {b:10.2f} {c['interleaved']['worstF']:8.2f} "
              f"{c['interleaved']['worstF']-b:+7.2f}")

    n_gate_div3 = int((mf <= 3).sum())
    n_gate_div2 = int((mf_hon <= 3).sum())
    print(f"\n  seeds at 3pp with divisor 3 (last task counted): {n_gate_div3}/15")
    print(f"  seeds at 3pp with divisor 2 (measurable only)  : {n_gate_div2}/15")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  The mean/worst gap averages {gap.mean():.2f}pp and is STRUCTURAL:")
    print(f"  two non-zero forgettings are averaged over three slots, so the mean")
    print(f"  is mechanically about 2/3 of the worst.")
    print(f"  With that geometry, meanF<=3 and worstF<=5 are simultaneously FEASIBLE")
    print(f"  (f1=f2=4.5pp does it), so the mission's pair of gates is not self-")
    print(f"  contradictory.")
    print(f"\n  What IS true: only {n_gate_div2}/15 seeds meet 3pp on the honest divisor,")
    print(f"  and 0/15 meet 5pp worst-case. The mechanism is further from the gate")
    print(f"  than the divisor-3 numbers suggest.")

    out = ROOT / "artifacts" / "deepseek-r7" / "worst-case-analysis.json"
    out.write_text(json.dumps({
        "n_seeds": len(cells), "order": list(ORDER),
        "corr_acq_worstF": corr(acq, wf), "corr_acq_meanF": corr(acq, mf),
        "corr_final_worstF": corr(fin, wf),
        "gap_mean_pp": float(gap.mean()), "gap_sd_pp": float(gap.std(ddof=1)),
        "seeds_at_3pp_div3": n_gate_div3, "seeds_at_3pp_div2": n_gate_div2,
        "feasible_both": True,
        "metric_note": "with T=3 the last task contributes a structural zero to meanF; "
                       "the mean is understated by T/(T-1)=1.5x vs mean over measurable",
        "analysis_of": SRC.name, "new_runs": 0,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: {out.name}")


if __name__ == "__main__":
    sys.exit(main())
