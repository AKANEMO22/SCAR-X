"""R7 round 25b — CORRECTION: round 25 assumed a divisor the code does not use.

ROUND 25'S ERROR
----------------
Round 25 argued that meanF averaged over all 3 tasks (including the last task's structural
zero) and therefore understated forgetting by 1.5x, and it "corrected" 8/15 seeds at the gate
down to 5/15.

That was wrong. Reading the actual computation:

    r7_r22_gate_reach.py:172   np.mean(fg[:n - 1])
    r7_r24_prereg.py:172       np.mean(fg[:n - 1])

Both divide by n-1 = 2, i.e. over the MEASURABLE tasks only. The divisor-3 issue was already
fixed in round 6 and I re-invented it as a new finding about my own code.

This is the same class of error as rounds 5b and 12: I reasoned about the metric from memory
instead of reading the line that computes it. Third instance. Recording it.

WHAT ROUND 25 GOT RIGHT AND IS WORTH KEEPING
---------------------------------------------
1. meanF / worstF FEASIBILITY. With 2 measurable tasks, forgetting = [f1, f2, 0]:
       meanF  = (f1+f2)/2
       worstF = max(f1, f2)
   meanF<=3 needs f1+f2<=6; worstF<=5 needs max<=5. f1=f2=3 gives meanF=3, worstF=3.
   **The two gates are simultaneously satisfiable.** They are not contradictory.

2. The mean/worst GAP is structural but not because of a divisor: worstF >= meanF + |f1-f2|/2
   when both are positive, so a single bad task can push worstF far above meanF while the
   mean stays small. That is what the 15 seeds show: gap averages 8.44pp.

This script re-does the analysis with the correct divisor and measures what actually
determines the gap.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "deepseek-r7" / "prereg-order-fresh15.json"
ORDER = ("order_short", "parity3", "sim_xor")
DIVISOR = 2      # n-1, as the code actually does


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    cells = d["cells"]
    mf = np.array([c["interleaved"]["meanF"] for c in cells])
    wf = np.array([c["interleaved"]["worstF"] for c in cells])
    acq = np.array([c["interleaved"]["acq"] for c in cells])
    fin = np.array([c["interleaved"]["final"] for c in cells])

    print("=" * 94)
    print("R7 ROUND 25b — corrected: the divisor was already n-1")
    print("=" * 94)
    print(f"  code uses np.mean(fg[:n-1]) = divisor {DIVISOR}. Round 25 assumed 3.")
    print(f"  round 25's '5/15 at gate' correction is WITHDRAWN; the figure is 8/15.\n")

    print("-" * 94)
    print("GATE FEASIBILITY under the real metric")
    print("-" * 94)
    print(f"  forgetting = [f1, f2, 0]  ->  meanF = (f1+f2)/2 ,  worstF = max(f1,f2)")
    print(f"  meanF<=3  requires f1+f2 <= 6")
    print(f"  worstF<=5 requires max(f1,f2) <= 5")
    print(f"  f1=f2=3 satisfies both (meanF=3.00, worstF=3.00) -> FEASIBLE")
    print(f"  So the mission's two forgetting gates are not mutually contradictory.")

    print("\n" + "-" * 94)
    print("WHAT SETS THE GAP between meanF and worstF")
    print("-" * 94)
    gap = wf - mf
    print(f"  gap mean {gap.mean():.2f}pp  sd {gap.std(ddof=1):.2f}  "
          f"min {gap.min():.2f}  max {gap.max():.2f}")
    print(f"  algebra: with f1<=f2, meanF=(f1+f2)/2, worstF=f2,")
    print(f"  so gap = (f2-f1)/2. The gap measures task IMBALANCE, nothing else.")
    print(f"  Hence f1 = meanF - gap, f2 = meanF + gap.")
    f1 = mf - gap
    f2 = mf + gap
    print(f"\n  implied per-task forgetting (meanF +- gap):")
    print(f"    task1 (order_short) mean {f1.mean():7.2f}pp  "
          f"range [{f1.min():7.2f}, {f1.max():7.2f}]")
    print(f"    task2 (parity3)     mean {f2.mean():7.2f}pp  "
          f"range [{f2.min():7.2f}, {f2.max():7.2f}]")

    print(f"\n  gap vs cell properties:")
    print(f"    corr(gap, acquisition) = {corr(gap, acq):+.4f}")
    print(f"    corr(gap, final)       = {corr(gap, fin):+.4f}")
    print(f"    corr(gap, meanF)       = {corr(gap, mf):+.4f}")

    print("\n" + "-" * 94)
    print("GATE STATUS, CORRECTED")
    print("-" * 94)
    n_mean = int((mf <= 3).sum())
    n_worst = int((wf <= 5).sum())
    both = int(((mf <= 3) & (wf <= 5)).sum())
    print(f"  seeds at meanF<=3pp : {n_mean}/15")
    print(f"  seeds at worstF<=5pp: {n_worst}/15")
    print(f"  seeds at BOTH       : {both}/15")
    print(f"  mean forgetting     : {mf.mean():.2f}pp (sd {mf.std(ddof=1):.2f})")
    print(f"  worst forgetting    : {wf.mean():.2f}pp (sd {wf.std(ddof=1):.2f})")

    print("\n" + "-" * 94)
    print("HOW FAR ARE THE FAILING SEEDS FROM BOTH GATES?")
    print("-" * 94)
    print(f"  {'seed':>8} {'meanF':>8} {'worstF':>8} {'need mean':>10} {'need worst':>11}")
    for c, a, b in zip(cells, mf, wf):
        nm = max(0.0, a - 3.0)
        nw = max(0.0, b - 5.0)
        mark = "  <- passes both" if nm == 0 and nw == 0 else ""
        print(f"  {c['seed']:>8} {a:8.2f} {b:8.2f} {nm:10.2f} {nw:11.2f}{mark}")
    print(f"\n  median needed reduction: meanF {np.median(np.maximum(mf-3,0)):.2f}pp, "
          f"worstF {np.median(np.maximum(wf-5,0)):.2f}pp")

    print("\n" + "=" * 94)
    print("VERDICT")
    print("=" * 94)
    print(f"  Round 25's divisor correction is WITHDRAWN (the code already divides by n-1).")
    print(f"  The substantive finding stands and is simpler than I made it:")
    print(f"    - the two forgetting gates are jointly FEASIBLE (f1=f2=3 does it);")
    print(f"    - the mean/worst gap IS task imbalance: gap = (f2-f1)/2 exactly;")
    print(f"    - mean imbalance here is {gap.mean():.2f}pp, so one task is carrying")
    print(f"      roughly {abs(f2.mean()-f1.mean()):.1f}pp more forgetting than the other;")
    print(f"    - closing worstF therefore means equalising the two tasks, not")
    print(f"      improving the average.")
    print(f"\n  On the real numbers: {n_mean}/15 seeds meet meanF<=3, {n_worst}/15 meet")
    print(f"  worstF<=5, {both}/15 meet both. The gate is not met.")

    out = ROOT / "artifacts" / "deepseek-r7" / "worst-case-analysis-corrected.json"
    out.write_text(json.dumps({
        "n_seeds": len(cells), "divisor": DIVISOR, "order": list(ORDER),
        "seeds_at_mean3": n_mean, "seeds_at_worst5": n_worst, "seeds_at_both": both,
        "meanF": float(mf.mean()), "meanF_sd": float(mf.std(ddof=1)),
        "worstF": float(wf.mean()), "worstF_sd": float(wf.std(ddof=1)),
        "gap_mean": float(gap.mean()), "gap_sd": float(gap.std(ddof=1)),
        "implied_task1_forgetting": float(f1.mean()),
        "implied_task2_forgetting": float(f2.mean()),
        "corr_gap_acq": corr(gap, acq),
        "gates_feasible_together": True,
        "retracts": "round 25's divisor-3 claim and its 5/15 correction",
        "analysis_of": SRC.name, "new_runs": 0,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: {out.name}")


if __name__ == "__main__":
    sys.exit(main())
