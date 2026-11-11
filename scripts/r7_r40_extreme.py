"""R7 round 40 — is the worst-case gate an EXTREME-VALUE artefact of task count?

THE HYPOTHESIS ROUND 39 POINTED AT
----------------------------------
Round 39 showed the damage RELOCATES when you move a task: worstF 8.54 (xnor first) ->
17.83 (xnor last), with parity3 taking the loss instead. No positional law, no template law.

That is the signature of an extreme-value statistic rather than a mechanism:

    worstF = max_j (delta_j)

The maximum of T correlated draws grows with T EVEN IF every individual delta is unchanged.
If the mechanism is fixed and only the number of tasks changes, worstF must rise mechanically.
The mission's gate holds worstF <= 5pp FIXED while task count varies — which is a different
kind of requirement from a fixed mean.

TEST — using ONLY stored artifacts, no new runs
    T=3 (nonlinear-90cells.json, 90 cells, mlp32, replay_tail): per-cell 'delta' vector exists
    T=6 (t6-weighted-replay.json + t6-position-vs-template.json): per-task deltas exist

    For each T, build the empirical distribution of per-task deltas, then compute the
    EXPECTED MAXIMUM of T draws from that same distribution (bootstrap, 2000 resamples).

    If E[max of T draws] >= measured worstF at both T, then worstF is fully explained by the
    extreme-value effect of task count and the mechanism adds nothing.

    Also: recompute worstF over only the first 3 tasks at T=6 (a matched-T comparison). If the
    T=6 first-3 worst is comparable to the T=3 worst, the "damage" is entirely the extra 3
    tasks' maxima, not a degradation of the mechanism.

This is the cleanest available decomposition and it needs no new compute.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"


def load(f):
    return json.loads((ROOT / f).read_text(encoding="utf-8"))


def expected_max(pool, T, rng, n=4000):
    pool = np.asarray(pool, float)
    return float(np.mean([rng.choice(pool, T, replace=True).max() for _ in range(n)]))


def main():
    print("=" * 96)
    print("R7 ROUND 40 — is worstF an extreme-value artefact of task count?")
    print("=" * 96)

    # ---------------- T=3 pool
    d90 = load("nonlinear-90cells.json")
    t3_pool = []
    t3_worst = []
    for c in d90["cells"]:
        delta = np.array(c["arms"]["replay_tail"]["delta"], float)
        meas = delta[:-1]                       # last task is a structural zero
        t3_pool.extend((meas * 100).tolist())   # delta is a FRACTION; convert to pp
        t3_worst.append(c["arms"]["replay_tail"]["worstF"])
    t3_pool = np.array(t3_pool)

    # ---------------- T=6 pool (aged arm, the best)
    d38 = load("t6-weighted-replay.json")
    t6_pool, t6_worst = [], []
    for c in d38["cells"]:
        pt = c["arms"]["aged"]["per_task"]
        vals = list(pt.values())
        # per_task includes the last task at 0.0; keep it, it is a real draw of 0
        t6_pool.extend(vals)
        t6_worst.append(c["arms"]["aged"]["worstF"])
    t6_pool = np.array(t6_pool)

    print(f"\n  per-task delta distributions (percent)")
    print(f"  {'T':>4} {'n draws':>8} {'mean':>8} {'sd':>7} {'max':>8}")
    for name, pool in (("T=3", t3_pool), ("T=6", t6_pool)):
        print(f"  {name:>4} {len(pool):8d} {pool.mean():8.2f} {pool.std(ddof=1):7.2f} "
              f"{pool.max():8.2f}")

    rng = np.random.default_rng(7)
    print(f"\n  EXPECTED MAXIMUM of T draws from the T=3 pool")
    print(f"  {'T':>4} {'E[max]':>9}")
    emax = {}
    for T in (3, 4, 5, 6):
        emax[T] = expected_max(t3_pool, T, rng)
        print(f"  {T:>4} {emax[T]:9.2f}")

    print(f"\n  measured worstF: T=3 {np.mean(t3_worst):.2f}pp, "
          f"T=6 {np.mean(t6_worst):.2f}pp")
    print(f"  E[max] if the T=3 mechanism simply had 6 tasks: {emax[6]:.2f}pp")
    print(f"  observed rise  : {np.mean(t6_worst) - np.mean(t3_worst):+.2f}pp")
    print(f"  predicted rise : {emax[6] - emax[3]:+.2f}pp")

    # ---------------- matched-T comparison: first 3 tasks at T=6
    print("\n" + "=" * 96)
    print("MATCHED-T CHECK — worstF over only the first 3 tasks, at T=6")
    print("=" * 96)
    d39 = load("t6-position-vs-template.json")
    first3_worst = {n: [] for n in d39["orders"]}
    all6_worst = {n: [] for n in d39["orders"]}
    for c in d39["cells"]:
        for n, order in d39["orders"].items():
            pt = c["orders"][n]["per_task"]
            vals = np.array([pt[tk] for tk in order], float)
            first3_worst[n].append(float(vals[:3].max()))
            all6_worst[n].append(float(vals.max()))
    print(f"  {'order':12s} {'worstF(first 3)':>16} {'worstF(all 6)':>14}")
    for n in d39["orders"]:
        print(f"  {n:12s} {np.mean(first3_worst[n]):16.2f} "
              f"{np.mean(all6_worst[n]):14.2f}")
    f3 = np.mean([np.mean(first3_worst[n]) for n in d39["orders"]])
    print(f"\n  mean worstF over first 3 tasks at T=6 : {f3:6.2f}pp")
    print(f"  mean worstF over all 3 tasks at T=3   : {np.mean(t3_worst):6.2f}pp")
    print(f"  difference                            : {f3 - np.mean(t3_worst):+6.2f}pp")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    pred_rise = emax[6] - emax[3]
    obs_rise = np.mean(t6_worst) - np.mean(t3_worst)
    matched_diff = f3 - np.mean(t3_worst)
    print(f"  observed worstF rise T=3 -> T=6 : {obs_rise:+.2f}pp")
    print(f"  extreme-value predicted rise    : {pred_rise:+.2f}pp")
    print(f"  matched-T (first 3 at T=6 vs T=3): {matched_diff:+.2f}pp")

    explained = obs_rise <= pred_rise + 0.5 and abs(matched_diff) <= 2.0
    if explained:
        print(f"\n  => EXTREME-VALUE EXPLANATION HOLDS. The worst-case gate rises because it is")
        print(f"     a maximum over more tasks, not because the mechanism degrades. At matched")
        print(f"     task count the damage is {matched_diff:+.2f}pp, i.e. unchanged.")
        print(f"     IMPLICATION: a FIXED worstF threshold is not scale-invariant. It must be")
        print(f"     quoted with its task count, or the mission gate is unsatisfiable as T grows")
        print(f"     for ANY mechanism with non-zero per-task variance.")
        verdict = "extreme_value_explained"
    elif matched_diff <= 2.0:
        print(f"\n  => PARTIAL. Matched-T difference is small ({matched_diff:+.2f}pp) so the")
        print(f"     mechanism does not degrade with task count, but the observed rise")
        print(f"     ({obs_rise:+.2f}pp) exceeds the extreme-value prediction "
              f"({pred_rise:+.2f}pp) by {obs_rise-pred_rise:+.2f}pp.")
        verdict = "partial"
    else:
        print(f"\n  => NOT EXPLAINED. Matched-T damage rises {matched_diff:+.2f}pp, so the")
        print(f"     mechanism genuinely degrades with task count beyond the maximum effect.")
        verdict = "not_explained"

    out = ROOT / "worst-case-extreme-value.json"
    out.write_text(json.dumps({
        "t3_pool_n": int(len(t3_pool)), "t6_pool_n": int(len(t6_pool)),
        "t3_mean": float(t3_pool.mean()), "t6_mean": float(t6_pool.mean()),
        "t3_sd": float(t3_pool.std(ddof=1)), "t6_sd": float(t6_pool.std(ddof=1)),
        "expected_max": {str(k): v for k, v in emax.items()},
        "measured_worstF_t3": float(np.mean(t3_worst)),
        "measured_worstF_t6": float(np.mean(t6_worst)),
        "observed_rise": float(obs_rise), "predicted_rise": float(pred_rise),
        "matched_T_first3_at_T6": float(f3),
        "matched_T_diff": float(matched_diff),
        "verdict": verdict, "new_runs": 0,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: {out.name}")


if __name__ == "__main__":
    main()
