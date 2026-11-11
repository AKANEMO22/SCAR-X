"""R7 round 39 — position or template? Put xnor_delay LAST.

REGISTERED IN ROUND 38, BEFORE THIS RUN
    `xnor_delay` is simultaneously the OLDEST task (position 1, absorbs five later updates)
    and the SPARSEST template (bits at ticks 2 and 4 on channels 7 and 0). It degrades +8.50pp
    while two other tasks IMPROVE (maj3 -9.46pp, order_short -5.29pp).

    Prediction: placing it LAST stops the degradation if the cause is POSITION.
    Falsifier : if it still degrades when last, the cause is the TEMPLATE and position is not
                the lever.

DESIGN — same six tasks, same seeds, same arm (aged replay), only the ORDER changes
    xnor_first   xnor_delay > maj3 > and3 > order_short > parity3 > sim_xor   (round 38's order)
    xnor_last    maj3 > and3 > order_short > parity3 > sim_xor > xnor_delay
    Also a control order where a DIFFERENT sparsish task sits first, to check the effect is
    about xnor_delay specifically rather than about "whatever is first":
    ctrl_first   sim_xor > maj3 > and3 > order_short > parity3 > xnor_delay   (sim_xor first)

    If POSITION is the cause: the FIRST task in every order should be the worst.
    If TEMPLATE is the cause: xnor_delay should be worst even when it is last, and sim_xor
    first should degrade only mildly.

REPORTED: per-task delta for each order, and which task is worst in each.
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
DECAY = 0.75

ORDERS = {
    "xnor_first": ("xnor_delay", "maj3", "and3", "order_short", "parity3", "sim_xor"),
    "xnor_last": ("maj3", "and3", "order_short", "parity3", "sim_xor", "xnor_delay"),
    "simx_first": ("sim_xor", "maj3", "and3", "order_short", "parity3", "xnor_delay"),
}


def run(seed, order, Ftr, ytr, Fte, yte):
    """Aged replay + tail, identical to round 38's winning arm."""
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

        n_seen = i_t + 1
        age = np.array([n_seen - 1 - order.index(t) for t in seq_tk])
        w = DECAY ** age
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
            "per_task": {tk: float(delta[j] * 100) for j, tk in enumerate(order)},
            "R": R.tolist()}


def main():
    t0 = time.time()
    print("=" * 98)
    print("R7 ROUND 39 — position or template? xnor_delay moved to LAST")
    print("=" * 98)
    print(f"  12 seeds, aged replay (round 38's arm), budget matched, only ORDER changes\n")
    for name, o in ORDERS.items():
        print(f"  {name:11s}: {' > '.join(o)}")
    print()

    cells = []
    for s_i, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in TASKS:
            trx, try_ = R37.GENS[tk](seed + 200_000, N_TR)
            tex, tey = R37.GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = R32.feats(core, trx); ytr[tk] = try_
            Fte[tk] = R32.feats(core, tex); yte[tk] = tey
        rec = {"seed": seed, "orders": {}}
        for name, order in ORDERS.items():
            rec["orders"][name] = run(seed, order, Ftr, ytr, Fte, yte)
        cells.append(rec)
        if (s_i + 1) % 4 == 0:
            print(f"  seed {s_i+1}/12 ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 98)
    print("AGGREGATE")
    print("=" * 98)
    print(f"  {'order':12s} {'meanF':>8} {'worstF':>8} {'acq':>8} {'seeds OK':>9}")
    summary = {}
    for name in ORDERS:
        mf = np.array([c["orders"][name]["meanF"] for c in cells])
        wf = np.array([c["orders"][name]["worstF"] for c in cells])
        aq = np.array([c["orders"][name]["acq"] for c in cells])
        ok = int(((mf <= 3) & (wf <= 5) & (aq >= .65)).sum())
        summary[name] = {"meanF": float(mf.mean()), "worstF": float(wf.mean()),
                         "acq": float(aq.mean()), "seeds_ok": ok}
        print(f"  {name:12s} {mf.mean():8.2f} {wf.mean():8.2f} {aq.mean():8.4f} "
              f"{ok:6d}/12")

    print("\n" + "=" * 98)
    print("PER-TASK DELTA — the discriminating table")
    print("=" * 98)
    print(f"  {'task':>14} " + " ".join(f"{n:>12}" for n in ORDERS))
    for tk in TASKS:
        vals = [np.mean([c["orders"][n]["per_task"][tk] for c in cells])
                for n in ORDERS]
        print(f"  {tk:>14} " + " ".join(f"{v:12.2f}" for v in vals))

    print("\n" + "=" * 98)
    print("WHICH POSITION IS WORST, IN EACH ORDER")
    print("=" * 98)
    for name, order in ORDERS.items():
        worst_pos, worst_val = None, -1e9
        for j, tk in enumerate(order):
            v = np.mean([c["orders"][name]["per_task"][tk] for c in cells])
            if v > worst_val:
                worst_val, worst_pos = v, j
        print(f"  {name:12s} first={order[0]:12s} worst position {worst_pos+1} "
              f"= {order[worst_pos]:12s} ({worst_val:+.2f}pp)")

    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    xn_first = np.mean([c["orders"]["xnor_first"]["per_task"]["xnor_delay"]
                        for c in cells])
    xn_last = np.mean([c["orders"]["xnor_last"]["per_task"]["xnor_delay"]
                       for c in cells])
    sx_first = np.mean([c["orders"]["simx_first"]["per_task"]["sim_xor"]
                        for c in cells])
    xn_in_simx = np.mean([c["orders"]["simx_first"]["per_task"]["xnor_delay"]
                          for c in cells])
    print(f"  xnor_delay when FIRST : {xn_first:+7.2f}pp")
    print(f"  xnor_delay when LAST  : {xn_last:+7.2f}pp")
    print(f"  recovery from moving it: {xn_first - xn_last:+7.2f}pp")
    print(f"\n  sim_xor when FIRST    : {sx_first:+7.2f}pp   (control: easy task first)")
    print(f"  xnor_delay when 2nd-last: {xn_in_simx:+7.2f}pp")

    if xn_last <= 2.0 and xn_first - xn_last >= 4.0:
        print(f"\n  => POSITION IS THE CAUSE. Moving xnor_delay from first to last removes")
        print(f"     {xn_first - xn_last:.1f}pp of its degradation. The mechanism must be")
        print(f"     consolidation of the earliest task, not its template.")
        verdict = "positional"
    elif xn_last >= 5.0:
        print(f"\n  => TEMPLATE IS THE CAUSE. xnor_delay still loses {xn_last:.1f}pp when it")
        print(f"     is last and absorbs no later updates, so position is not the lever.")
        print(f"     The mechanism must be representational.")
        verdict = "template"
    else:
        print(f"\n  => MIXED. {xn_first:.1f}pp first vs {xn_last:.1f}pp last: position explains")
        print(f"     {xn_first - xn_last:.1f}pp, the remainder is template.")
        verdict = "mixed"

    print(f"\n  SEED-LEVEL: seeds passing all gates")
    for name in ORDERS:
        print(f"    {name:12s} {summary[name]['seeds_ok']}/12")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t6-position-vs-template.json").write_text(json.dumps({
        "seeds": list(SEEDS), "orders": {k: list(v) for k, v in ORDERS.items()},
        "summary": summary, "verdict": verdict,
        "xnor_delay_first_pp": float(xn_first),
        "xnor_delay_last_pp": float(xn_last),
        "sim_xor_first_pp": float(sx_first),
        "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: t6-position-vs-template.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
