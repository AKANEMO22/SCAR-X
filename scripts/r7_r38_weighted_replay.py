"""R7 round 38 — which task carries the T=6 gap, and can a targeted mechanism close it?

ROUND 37 LEFT THIS
    T=6 corrected: meanF 3.84pp (gate 3, over by 0.84), worstF 9.12pp (gate 5, over by 4.12),
    0/12 seeds, acquisition 0.9359.
The mean is nearly there; the worst case is not. Round 25 showed the mean/worst gap is exactly
task imbalance, so the question is WHICH task and WHY.

PART A (no new runs) — read the 36 stored R matrices from `t6-corrected.json`
    per-task forgetting, de-biased by each task's own floor where available
    is the worst task POSITIONAL (the first, absorbing five later updates) or TASK-SPECIFIC?

PART B (one targeted run) — a mechanism matched to the answer
    If POSITIONAL: replay the earliest task MORE OFTEN as it ages. A geometrically decaying
    weight over the buffer (most recent task 1.0, oldest 0.4) keeps the earliest task in the
    mix without increasing buffer size or total budget.
    If TASK-SPECIFIC: weight replay toward the task with the lowest running accuracy, which
    is computable online from the buffer with no task ID beyond the replay row's own label.

    Both are the same implementation with different weights, so one run tests both:
        uniform    round 37's setting (each buffered row equally likely)
        aged       weight_i = decay ** (number of tasks learned since row i's task)
        adaptive   weight_i proportional to (1 - running accuracy on that task's buffer rows)

    decay = 0.75 over 6 tasks gives the earliest task 0.75^5 = 0.24 of a fresh row.

REGISTERED PREDICTION
    worstF falls by >= 2pp against uniform, meanF stays <= 4pp, and acquisition does not fall
    more than 0.01. Replay weights change only WHICH rows are drawn, not how many steps run,
    so budget is matched by construction.

FALSIFIER
    If neither weighting moves worstF by >= 1pp, the remaining gap is not replay allocation and
    must be attacked elsewhere.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import r7_r32_tier3_six as R32  # noqa: E402
import r7_r37_t6_fixed as R37  # noqa: E402  (tasks + assertions already validated)

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = R37.SEEDS
TASKS = R37.TASKS
ORDERS = R37.ORDERS
N_TR, N_TE = R37.N_TR, R37.N_TE
BUF = R37.BUF
EPOCHS = R37.EPOCHS
TAIL = R37.TAIL
BEST_ORDER = ("xnor_delay", "maj3", "and3", "order_short", "parity3", "sim_xor")
ARMS = ("uniform", "aged", "adaptive")
DECAY = 0.75


def part_a():
    src = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7" \
        / "t6-corrected.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    cells = [c for c in d["cells"] if tuple(c["order"]) == BEST_ORDER]
    print("=" * 96)
    print("PART A — which task carries the gap? (best order, replay_tail, no new runs)")
    print("=" * 96)
    print(f"  order: {' > '.join(BEST_ORDER)}")
    print(f"  {len(cells)} seeds\n")

    if "R" not in cells[0]["arms"]["replay_tail"]:
        print("  NOTE: t6-corrected.json stored only aggregates, not R matrices.")
        print("  Part A therefore reports what IS recoverable: meanF and worstF per seed,")
        print("  which bound the imbalance but do not name the task. The per-task answer")
        print("  comes from Part B, which stores R.")
        wf = np.array([c["arms"]["replay_tail"]["worstF"] for c in cells])
        mf = np.array([c["arms"]["replay_tail"]["meanF"] for c in cells])
        print(f"\n  worstF mean {wf.mean():.2f}  meanF (loss) mean {mf.mean():.2f}")
        print(f"  implied imbalance: with 5 measurable tasks and one worst,")
        print(f"  worstF - meanF = {wf.mean() - mf.mean():.2f}pp of concentrated damage")
        print(f"  seeds where worstF > 5pp: {int((wf > 5).sum())}/12")
        return None, None, {}

    # delta per task across seeds
    agg = {tk: [] for tk in BEST_ORDER}
    pos = {i: [] for i in range(len(BEST_ORDER))}
    for c in cells:
        R = np.array(c["arms"]["replay_tail"]["R"], float)
        delta = np.diag(R) - R[-1]
        for j, tk in enumerate(BEST_ORDER):
            agg[tk].append(delta[j] * 100)
            pos[j].append(delta[j] * 100)

    print(f"  {'position':>8} {'task':>14} {'mean delta':>11} {'worst':>8} {'n>5pp':>7}")
    for j, tk in enumerate(BEST_ORDER):
        v = np.array(agg[tk])
        print(f"  {j+1:>8} {tk:>14} {v.mean():11.2f} {v.max():8.2f} "
              f"{int((v > 5).sum()):5d}/12")

    print(f"\n  {'position (pooled)':>18} {'mean':>9} {'max':>8}")
    for j in range(len(BEST_ORDER)):
        v = np.array(pos[j])
        print(f"  {j+1:>18} {v.mean():9.2f} {v.max():8.2f}")

    means = {tk: float(np.mean(agg[tk])) for tk in BEST_ORDER}
    worst_task = max(means, key=means.get)
    first_task = BEST_ORDER[0]
    print(f"\n  worst TASK by mean delta: {worst_task} ({means[worst_task]:.2f}pp)")
    print(f"  first position is    : {first_task} "
          f"({means[first_task]:.2f}pp)")
    positional = worst_task == first_task
    print(f"  => {'POSITIONAL' if positional else 'TASK-SPECIFIC'} pattern")
    return worst_task, positional, means


def run(seed, order, Ftr, ytr, Fte, yte, arm):
    head = R32.MLP(Ftr[order[0]].shape[1], seed, 32)
    n = len(order)
    R = np.full((n, n), np.nan)
    seq_idx, seq_lab, seq_tk = [], [], []
    acc_hist = {}                       # per-task running accuracy on its buffer rows
    rng = np.random.default_rng(seed + 909)

    for i_t, tk in enumerate(order):
        A_c = head.z(Ftr[tk]); y_c = ytr[tk]
        k = min(BUF, N_TR)
        idx = np.random.default_rng(seed + i_t).choice(N_TR, k, replace=False)
        seq_idx.extend(int(j) for j in idx)
        seq_lab.extend(int(ytr[tk][j]) for j in idx)
        seq_tk.extend([tk] * len(idx))
        A_r = np.stack([head.z(Ftr[seq_tk[q]])[seq_idx[q]]
                        for q in range(len(seq_idx))])
        y_r = np.array(seq_lab, np.int32)

        n_seen = i_t + 1
        if arm == "uniform":
            w = np.ones(len(y_r))
        elif arm == "aged":
            age = np.array([n_seen - 1 - order.index(t) for t in seq_tk])
            w = DECAY ** age
        else:                           # adaptive: favour worst-performing buffered tasks
            acc_hist[tk] = head.acc(A_r[[q for q, t in enumerate(seq_tk) if t == tk]],
                                    y_r[[q for q, t in enumerate(seq_tk) if t == tk]])
            w = np.array([1.0 - acc_hist.get(t, 0.5) + 1e-3 for t in seq_tk])
        p = w / w.sum()

        hc = max(1, len(y_c) // 2); hr = max(1, len(y_r) // 2)
        for _ in range(2 * EPOCHS):
            ic = rng.choice(len(y_c), hc, replace=False)
            ir = rng.choice(len(y_r), hr, replace=False, p=p)
            head.step(np.concatenate([A_c[ic], A_r[ir]]),
                      np.concatenate([y_c[ic], y_r[ir]]))
        for j_t, tk2 in enumerate(order):
            R[i_t, j_t] = head.acc(head.z(Fte[tk2]), yte[tk2])

    nb = min(BUF, len(seq_idx))
    A1 = head.z(Ftr[order[0]])[seq_idx[:nb]]
    y1 = np.array(seq_lab[:nb], np.int32)
    Al = head.z(Ftr[order[-1]]); yl = ytr[order[-1]]
    h1 = max(1, len(y1) // 2); hl = max(1, len(yl) // 2)
    for _ in range(TAIL):
        i1 = rng.choice(len(y1), h1, replace=False)
        il = rng.choice(len(yl), hl, replace=False)
        head.step(np.concatenate([A1[i1], Al[il]]),
                  np.concatenate([y1[i1], yl[il]]))
    for j_t, tk2 in enumerate(order):
        R[n - 1, j_t] = head.acc(head.z(Fte[tk2]), yte[tk2])

    diag = np.diag(R); final = R[n - 1]; delta = diag - final
    meas = delta[:n - 1]
    return {"meanF": float(np.mean(np.maximum(meas, 0)) * 100),
            "worstF": float(np.max(delta) * 100),
            "acq": float(np.mean(diag)),
            "per_task": {tk: float(delta[j] * 100)
                         for j, tk in enumerate(order)}}


def main():
    t0 = time.time()
    worst_task, positional, means = part_a()
    if worst_task is None:
        worst_task, positional, means = "(not stored)", None, {}

    print("\n" + "=" * 96)
    print("PART B — targeted replay weighting (best order, 12 seeds, budget matched)")
    print("=" * 96)
    print(f"  arms {ARMS}; aging decay {DECAY} -> earliest task gets "
          f"{DECAY**5:.2f} of a fresh row's weight\n")

    cells = []
    for s_i, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in TASKS:
            trx, try_ = R37.GENS[tk](seed + 200_000, N_TR)
            tex, tey = R37.GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = R32.feats(core, trx); ytr[tk] = try_
            Fte[tk] = R32.feats(core, tex); yte[tk] = tey
        rec = {"seed": seed, "arms": {}}
        for arm in ARMS:
            rec["arms"][arm] = run(seed, BEST_ORDER, Ftr, ytr, Fte, yte, arm)
        cells.append(rec)
        if (s_i + 1) % 4 == 0:
            print(f"  seed {s_i+1}/12 ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 96)
    print("RESULTS")
    print("=" * 96)
    print(f"  {'arm':12s} {'meanF':>8} {'worstF':>8} {'acq':>8} {'seeds OK':>9}")
    summary = {}
    for arm in ARMS:
        mf = np.array([c["arms"][arm]["meanF"] for c in cells])
        wf = np.array([c["arms"][arm]["worstF"] for c in cells])
        aq = np.array([c["arms"][arm]["acq"] for c in cells])
        ok = int(((mf <= 3) & (wf <= 5) & (aq >= .65)).sum())
        summary[arm] = {"meanF": float(mf.mean()), "worstF": float(wf.mean()),
                        "acq": float(aq.mean()), "seeds_ok": ok}
        print(f"  {arm:12s} {mf.mean():8.2f} {wf.mean():8.2f} {aq.mean():8.4f} "
              f"{ok:6d}/12")

    print("\n" + "=" * 96)
    print("PER-TASK DELTA (mean over 12 seeds)")
    print("=" * 96)
    print(f"  {'task':>14} " + " ".join(f"{a:>10}" for a in ARMS))
    for tk in BEST_ORDER:
        vals = [np.mean([c["arms"][a]["per_task"][tk] for c in cells]) for a in ARMS]
        print(f"  {tk:>14} " + " ".join(f"{v:10.2f}" for v in vals))

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    base = summary["uniform"]
    for arm in ("aged", "adaptive"):
        s = summary[arm]
        print(f"  {arm:9s} vs uniform: worstF {s['worstF']-base['worstF']:+6.2f}pp  "
              f"meanF {s['meanF']-base['meanF']:+6.2f}pp  "
              f"acq {s['acq']-base['acq']:+.4f}  seeds {s['seeds_ok']}/12")
    best = min(ARMS, key=lambda a: summary[a]["worstF"])
    drop = base["worstF"] - summary[best]["worstF"]
    if drop >= 2.0:
        print(f"\n  => CONFIRMED. {best} weighting cuts worstF by {drop:.2f}pp.")
        verdict = "weighting_helps"
    elif drop >= 1.0:
        print(f"\n  => PARTIAL. {best} cuts worstF {drop:.2f}pp, below the registered 2pp.")
        verdict = "weighting_partial"
    else:
        print(f"\n  => REJECTED. No weighting moves worstF by 1pp. The remaining gap is not")
        print(f"     replay allocation.")
        verdict = "weighting_fails"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t6-weighted-replay.json").write_text(json.dumps({
        "order": list(BEST_ORDER), "seeds": list(SEEDS), "arms": list(ARMS),
        "decay": DECAY, "summary": summary, "verdict": verdict,
        "part_a": {"worst_task": worst_task, "positional": bool(positional),
                   "per_task_mean": means},
        "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: t6-weighted-replay.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
