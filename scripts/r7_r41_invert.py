"""R7 round 41 — the registered test: invert the age decay.

ROUND 40 REGISTERED THIS, BEFORE THE RUN
    At matched task count, a task in a six-task run is already 5.41pp more damaged than the same
    task in a three-task run (worstF first-3 at T=6 = 6.88pp vs T=3 = 1.46pp). So the damage is
    caused by the LATER tasks trained before the measurement, not by how many tasks are
    maximised over.

    Prediction: replaying the LATER tasks more heavily should reduce the EARLIER tasks' damage.
    That is the INVERSE of round 38's age decay, which favoured the oldest.

ARMS (same six tasks, same seeds, same budget, only the draw probability changes)
    old_first   w = 0.75 ** age      favour the OLDEST task   (round 38's winner)
    uniform     w = 1                no weighting
    recent_last w = 0.75 ** (T-1-age) favour the NEWEST task  (the round-40 prediction)
    both_ends   w = 0.75 ** min(age, T-1-age)   U-shaped: protect both extremes

WHY both_ends IS INCLUDED
    Round 38's ageing helped (worstF -1.38pp) and round 40 says the later tasks cause the
    damage. Both could be partly right: the earliest task is the most VULNERABLE and the
    latest is the most AGGRESSIVE. A U-shaped weight protects the vulnerable one while
    letting the aggressive one be represented, which is a different prediction from either
    single-ended scheme.

REGISTERED PREDICTION
    recent_last beats old_first on worstF by >= 2pp. If so, round 40's mechanism reading is
    confirmed and round 38's interpretation was the wrong sign on a real effect.

FALSIFIER
    If recent_last is not better than old_first, the "later tasks cause the damage" reading is
    wrong and the interference is not directional.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import r7_r32_tier3_six as R32  # noqa: E402
import r7_r37_t6_fixed as R37  # noqa: E402

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = R37.SEEDS
TASKS = R37.TASKS
N_TR, N_TE = R37.N_TR, R37.N_TE
BUF, EPOCHS, TAIL = R37.BUF, R37.EPOCHS, R37.TAIL
ORDER = ("xnor_delay", "maj3", "and3", "order_short", "parity3", "sim_xor")
DECAY = 0.75
ARMS = ("old_first", "uniform", "recent_last", "both_ends")


def weights(arm, seq_tk, order, n_seen):
    age = np.array([n_seen - 1 - order.index(t) for t in seq_tk], float)
    span = n_seen - 1
    if arm == "old_first":
        w = DECAY ** age
    elif arm == "uniform":
        w = np.ones(len(age))
    elif arm == "recent_last":
        w = DECAY ** (span - age)
    elif arm == "both_ends":
        w = DECAY ** np.minimum(age, span - age)
    else:
        raise ValueError(arm)
    return w / w.sum()


def run(seed, order, Ftr, ytr, Fte, yte, arm):
    head = R32.MLP(Ftr[order[0]].shape[1], seed, 32)
    n = len(order)
    R = np.full((n, n), np.nan)
    seq_idx, seq_lab, seq_tk = [], [], []
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
        p = weights(arm, seq_tk, order, i_t + 1)

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
            "first3_worst": float(np.max(delta[:3]) * 100),
            "per_task": {tk: float(delta[j] * 100) for j, tk in enumerate(order)}}


def main():
    t0 = time.time()
    print("=" * 98)
    print("R7 ROUND 41 — invert the age decay: do the LATER tasks cause the damage?")
    print("=" * 98)
    print(f"  order: {' > '.join(ORDER)}")
    print(f"  12 seeds, 4 weighting schemes, budget matched\n")

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
            rec["arms"][arm] = run(seed, ORDER, Ftr, ytr, Fte, yte, arm)
        cells.append(rec)
        if (s_i + 1) % 4 == 0:
            print(f"  seed {s_i+1}/12 ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 98)
    print("RESULTS")
    print("=" * 98)
    print(f"  {'arm':12s} {'meanF':>8} {'worstF':>8} {'first3':>8} {'acq':>8} "
          f"{'seeds OK':>9}")
    summary = {}
    for arm in ARMS:
        mf = np.array([c["arms"][arm]["meanF"] for c in cells])
        wf = np.array([c["arms"][arm]["worstF"] for c in cells])
        f3 = np.array([c["arms"][arm]["first3_worst"] for c in cells])
        aq = np.array([c["arms"][arm]["acq"] for c in cells])
        ok = int(((mf <= 3) & (wf <= 5) & (aq >= .65)).sum())
        summary[arm] = {"meanF": float(mf.mean()), "worstF": float(wf.mean()),
                        "first3": float(f3.mean()), "acq": float(aq.mean()),
                        "seeds_ok": ok}
        print(f"  {arm:12s} {mf.mean():8.2f} {wf.mean():8.2f} {f3.mean():8.2f} "
              f"{aq.mean():8.4f} {ok:6d}/12")

    print("\n" + "=" * 98)
    print("PER-TASK DELTA")
    print("=" * 98)
    print(f"  {'task':>14} " + " ".join(f"{a:>12}" for a in ARMS))
    for tk in ORDER:
        vals = [np.mean([c["arms"][a]["per_task"][tk] for c in cells]) for a in ARMS]
        print(f"  {tk:>14} " + " ".join(f"{v:12.2f}" for v in vals))

    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    of = summary["old_first"]
    rl = summary["recent_last"]
    be = summary["both_ends"]
    print(f"  registered: recent_last beats old_first on worstF by >= 2pp")
    print(f"  old_first  worstF {of['worstF']:6.2f}  first3 {of['first3']:6.2f}")
    print(f"  recent_last worstF {rl['worstF']:6.2f}  first3 {rl['first3']:6.2f}")
    print(f"  both_ends  worstF {be['worstF']:6.2f}  first3 {be['first3']:6.2f}")
    gain = of["worstF"] - rl["worstF"]
    print(f"\n  recent_last - old_first = {gain:+.2f}pp on worstF, "
          f"{of['first3'] - rl['first3']:+.2f}pp on first3")

    best = min(ARMS, key=lambda a: summary[a]["worstF"])
    b = summary[best]
    print(f"\n  best arm overall: {best}  worstF {b['worstF']:.2f}  meanF {b['meanF']:.2f}")

    if gain >= 2.0:
        print(f"\n  => CONFIRMED. Favour-the-recent cuts worstF by {gain:.2f}pp over")
        print(f"     favour-the-old. Round 40's mechanism reading holds and round 38's")
        print(f"     improvement came from the wrong sign of a real effect.")
        verdict = "recent_confirmed"
    elif gain >= 1.0:
        print(f"\n  => PARTIAL. {gain:.2f}pp, below the registered 2pp.")
        verdict = "recent_partial"
    else:
        print(f"\n  => REJECTED. Favour-the-recent does not beat favour-the-old. The")
        print(f"     interference is not directional in the way round 40 predicted.")
        verdict = "recent_rejected"

    if b["seeds_ok"] > 0:
        print(f"\n  {best} puts {b['seeds_ok']}/12 seeds inside ALL THREE gates.")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t6-inverted-decay.json").write_text(json.dumps({
        "order": list(ORDER), "seeds": list(SEEDS), "arms": list(ARMS),
        "decay": DECAY, "summary": summary, "verdict": verdict,
        "recent_vs_old_worstF": float(gain), "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: t6-inverted-decay.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
