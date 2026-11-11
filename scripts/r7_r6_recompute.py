"""R7 round 6 — independent recomputation of the retention metrics from raw R matrices.

The mission requires (Step 8) recalculating important metrics from raw data rather than
trusting an automatic summary. Doing that on round 5b immediately exposes a defect in my
own gate arithmetic.

THE DEFECT. With T tasks, `forgetting_j = R[j][j] - R[T-1][j]`. For the LAST task,
j = T-1, so `forgetting = R[T-1][T-1] - R[T-1][T-1] = 0` IDENTICALLY. It is zero by
construction, not by measurement: the last task is never evaluated again after anything
else is learned.

Round 5b averaged over all T tasks including that structural zero, which DILUTES the
mean by a factor of (T-1)/T. At T=3 that is a 33% understatement.

This script reports:
  - `meanF_diluted`   average over all T incl. the structural zero   (what 5b printed)
  - `meanF_honest`    average over the T-1 tasks where forgetting is measurable
  - `worstF`          unchanged, but with the task identified

It then re-evaluates the mission's gates against the honest figure, because the mission's
thresholds were written for real forgetting, not for an average diluted by a constant.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("sim_xor", "parity3", "order_short")
GATES = {"acq": .65, "final": .65, "mean_f_pp": 3.0, "worst_f_pp": 5.0}


def load(run_name_part):
    for d in (ROOT / "runs").iterdir():
        if d.is_dir() and run_name_part in d.name:
            return json.loads((d / "result.json").read_text(encoding="utf-8")), d.name
    raise FileNotFoundError(run_name_part)


def recompute(trials, head_key):
    """Per-seed R matrix -> acquisition, final, diluted and honest forgetting."""
    out = []
    n = len(TASKS)
    for r in trials:
        if r.get(head_key) != trials[0].get(head_key):
            pass
        R = np.array(r["R"], dtype=float)
        diag = np.diag(R)
        final = R[n - 1]
        forget = diag - final
        out.append({
            "seed": r["seed"],
            "forget_per_task": {TASKS[j]: float(forget[j] * 100) for j in range(n)},
            "meanF_diluted": float(np.mean(forget) * 100),
            "meanF_honest": float(np.mean(forget[:n - 1]) * 100),
            "worstF": float(np.max(forget) * 100),
            "worst_task": TASKS[int(np.argmax(forget))],
            "acq": float(np.mean(diag)),
            "final": float(np.mean(final)),
        })
    return out


def main():
    print("=" * 90)
    print("R7 ROUND 6 — independent recomputation of retention metrics")
    print("=" * 90)

    data, runname = load("r7-r5b-capacity-diagnostic")
    print(f"  source: {runname}")
    print(f"  tasks : {TASKS}   (T={len(TASKS)})")
    print(f"\n  With T={len(TASKS)}, task '{TASKS[-1]}' is the LAST learned, so")
    print(f"  forgetting_{TASKS[-1]} = R[T-1][T-1] - R[T-1][T-1] = 0 identically.")
    print(f"  Averaging over all {len(TASKS)} tasks therefore dilutes the mean by "
          f"{(len(TASKS)-1)}/{len(TASKS)}.")

    rows_by_head = {}
    for head in ("linear_2logit", "linear_wide_32", "mlp32_tanh"):
        subset = [r for r in data["trials"] if r["head"] == head]
        rows_by_head[head] = recompute(subset, "head")

    print("\n" + "=" * 90)
    print("PER-SEED, HONEST DECOMPOSITION")
    print("=" * 90)
    for head, rows in rows_by_head.items():
        print(f"\n  {head}")
        print(f"    {'seed':>8} {'acq':>7} {'final':>7} {'meanF_dil':>10} "
              f"{'meanF_hon':>10} {'worstF':>8} {'worst task':>12}")
        for r in rows:
            print(f"    {r['seed']:>8} {r['acq']:7.4f} {r['final']:7.4f} "
                  f"{r['meanF_diluted']:9.2f}p {r['meanF_honest']:9.2f}p "
                  f"{r['worstF']:7.2f}p {r['worst_task']:>12}")

    print("\n" + "=" * 90)
    print("GATE RE-EVALUATION WITH THE HONEST MEAN")
    print("=" * 90)
    print(f"  {'head':16s} {'acq':>7} {'final':>7} {'meanF_hon':>10} {'worstF':>8} "
          f"{'gate(honest)':>13} {'gate(5b)':>9}")
    summary = {}
    for head, rows in rows_by_head.items():
        acq = float(np.mean([r["acq"] for r in rows]))
        final = float(np.mean([r["final"] for r in rows]))
        mh = float(np.mean([r["meanF_honest"] for r in rows]))
        wf = float(np.mean([r["worstF"] for r in rows]))
        gwf = float(np.max([r["worstF"] for r in rows]))
        passes_honest = bool(acq >= GATES["acq"] and final >= GATES["final"]
                             and mh <= GATES["mean_f_pp"] and gwf <= GATES["worst_f_pp"])
        md = float(np.mean([r["meanF_diluted"] for r in rows]))
        passes_5b = bool(acq >= GATES["acq"] and final >= GATES["final"]
                         and md <= GATES["mean_f_pp"] and gwf <= GATES["worst_f_pp"])
        summary[head] = {"acq": acq, "final": final, "meanF_diluted": md,
                         "meanF_honest": mh, "worstF_mean": wf, "worstF_max": gwf,
                         "passes_honest": passes_honest, "passes_5b": passes_5b}
        print(f"  {head:16s} {acq:7.4f} {final:7.4f} {mh:9.2f}p {gwf:7.2f}p "
              f"{('PASS' if passes_honest else 'fail'):>13} "
              f"{('PASS' if passes_5b else 'fail'):>9}")

    print("\n" + "=" * 90)
    print("VERDICT ON MY OWN ROUND-5b CLAIM")
    print("=" * 90)
    m = summary["mlp32_tanh"]
    print(f"  Round 5b said: meanF 2.86pp, under the 3pp gate.")
    print(f"  Diluted mean : {m['meanF_diluted']:.2f}pp  (includes the structural zero)")
    print(f"  Honest mean  : {m['meanF_honest']:.2f}pp  (over the 2 measurable tasks)")
    print(f"  => the 5b statement was INFLATED by the diluted denominator.")
    print(f"  Honest gate outcome for mlp32_tanh: "
          f"{'PASS' if m['passes_honest'] else 'FAIL'}")
    print(f"  worst-case forgetting (max over seeds): {m['worstF_max']:.2f}pp "
          f"vs {GATES['worst_f_pp']}pp gate")

    (ROOT / "artifacts" / "deepseek-r7").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "deepseek-r7" / "retention-recompute.json").write_text(
        json.dumps({"source_run": runname, "tasks": TASKS,
                    "note": "meanF_honest excludes the last task, whose forgetting is "
                            "zero by construction, not by measurement",
                    "per_seed": rows_by_head, "summary": summary},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/retention-recompute.json")


if __name__ == "__main__":
    main()
