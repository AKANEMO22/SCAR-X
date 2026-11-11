"""R7 round 32 — TIER 3: does the result survive 6 tasks, not 3?

WHERE WE ARE
------------
Round 31: 15 seeds x 6 orders = 90 cells, mlp32 readout, 3 tasks.
    90/90 cells pass acquisition, final, meanF<=3, worstF<=5
    3 of 6 orders reach 15/15 seeds on every gate
    replay acquisition cost +0.008pp (linear head: -1.43pp)

That is Tier 1 (implementation) and Tier 2 (causal) evidence. **It is all at T=3.**

THE MISSION'S THIRD TIER
------------------------
"Generalization validity: does the effect exist OUTSIDE the optimised conditions?"
The single most obvious optimised condition is the task count. Round 29 measured the
irreducible floor of a LINEAR head and round 8/27 showed that floor scales with how many
distributions must share one boundary set. With a nonlinear head the floor was 0.36pp at T=3.
**At T=6 it may not be negligible.**

This round adds three new tasks and runs the identical protocol at T=6.

NEW TASKS (same 12-tick, 8-input, gate-at-10 format; each is a distinct conjunction)
    and3       a AND b AND c   (order-3 conjunction, unlike parity3's XOR)
    maj3       majority of three bits
    xnor_delay two bits, 2-tick apart, label = NOT(XOR)   (temporal + inverted)

DESIGN
    12 fresh seeds (320001 + 6k, never used)  x  3 orders at T=6  = 36 cells
    Two orders only is enough: round 31 showed order variance at T=3 is 0.42-1.63pp and
    ALL orders passed. The question here is scale, not order.
    Arms: noreplay, replay_tail (the round-31 best).

REGISTERED PREDICTION
    P1  at T=6 the nonlinear floor rises but stays below 3pp, and at least one order
        reaches meanF <= 3pp and worstF <= 5pp on >= 10/12 seeds.
    P2  replay's acquisition cost stays under 1pp in absolute value.

FALSIFIER
    If meanF at T=6 exceeds 3pp on the best order, the result is T=3-specific and the
    mission gate is met only at the benchmark size I happened to choose. I will say so.

This is the test that decides whether round 31 is a benchmark result or a mechanism result.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = tuple(320001 + 6 * k for k in range(12))
N_TR, N_TE = 320, 200          # slightly smaller per task: 6 tasks, same total data
BUF = 100
EPOCHS = 600
TAIL = 250
ARMS = ("noreplay", "replay_tail")

TASKS = ("sim_xor", "parity3", "order_short", "and3", "maj3", "xnor_delay")
# three orders, chosen to vary which task is first/last without enumeration over 720
ORDERS = [
    ("sim_xor", "parity3", "order_short", "and3", "maj3", "xnor_delay"),
    ("xnor_delay", "maj3", "and3", "order_short", "parity3", "sim_xor"),
    ("and3", "order_short", "maj3", "sim_xor", "xnor_delay", "parity3"),
]


def sim_xor(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        a = int(rng.integers(2)); b = int(rng.integers(2)); y[i] = a ^ b
        t = int(rng.integers(1, 4))
        x[i, t, 0] += 1.5 * a; x[i, t, 1] += 1.5 * b; x[i, 10, 7] += 1.5
    return x, y


def parity3(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        bits = [int(rng.integers(2)) for _ in range(3)]
        y[i] = bits[0] ^ bits[1] ^ bits[2]
        t = int(rng.integers(1, 4))
        for k, bb in enumerate(bits):
            x[i, t, k] += 1.5 * bb
        x[i, 10, 7] += 1.5
    return x, y


def order_short(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        y[i] = int(rng.integers(2))
        first, second = (4, 5) if y[i] == 0 else (5, 4)
        p = int(rng.integers(1, 4))
        x[i, p, first] += 1.5
        x[i, p + 2, second] += 1.5
        x[i, 10, 7] += 1.5
    return x, y


def and3(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        bits = [int(rng.integers(2)) for _ in range(3)]
        y[i] = bits[0] & bits[1] & bits[2]
        t = int(rng.integers(1, 4))
        for k, bb in enumerate(bits):
            x[i, t, k] += 1.5 * bb
        x[i, 10, 7] += 1.5
    return x, y


def maj3(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        bits = [int(rng.integers(2)) for _ in range(3)]
        y[i] = 1 if sum(bits) >= 2 else 0
        t = int(rng.integers(1, 4))
        for k, bb in enumerate(bits):
            x[i, t, k] += 1.5 * bb
        x[i, 10, 7] += 1.5
    return x, y


def xnor_delay(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        a = int(rng.integers(2)); b = int(rng.integers(2)); y[i] = 1 - (a ^ b)
        x[i, 2, 0] += 1.5 * a
        x[i, 4, 1] += 1.5 * b
        x[i, 10, 7] += 1.5
    return x, y


GENS = {"sim_xor": sim_xor, "parity3": parity3, "order_short": order_short,
        "and3": and3, "maj3": maj3, "xnor_delay": xnor_delay}


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def feats(core, X):
    out = []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
        out.append(tot)
    return np.stack(out)


class MLP:
    __slots__ = ("mu", "sd", "W1", "b1", "W2", "b2")

    def __init__(self, d, seed=0, H=32):
        self.mu = None; self.sd = None
        g = np.random.default_rng(seed + 21)
        self.W1 = g.normal(0, 1 / np.sqrt(d), (H, d)); self.b1 = np.zeros(H)
        self.W2 = g.normal(0, 1 / np.sqrt(H), (2, H)); self.b2 = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 0.05)
        return (F - self.mu) / self.sd

    def fwd(self, A):
        Hd = np.tanh(A @ self.W1.T + self.b1)
        return Hd, Hd @ self.W2.T + self.b2

    def step(self, A, y, lr=.1):
        n = len(y)
        Hd, lg = self.fwd(A)
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        e = -p; e[np.arange(n), y] += 1
        eH = (e @ self.W2) * (1 - Hd ** 2)
        self.W2 += lr * (e.T @ Hd) / n; self.b2 += lr * e.mean(0)
        self.W1 += lr * (eH.T @ A) / n; self.b1 += lr * eH.mean(0)

    def acc(self, A, y):
        _, lg = self.fwd(A)
        return float(np.mean(np.argmax(lg, 1) == y))


def run(seed, order, Ftr, ytr, Fte, yte, arm):
    head = MLP(Ftr[order[0]].shape[1], seed, 32)
    n = len(order)
    R = np.full((n, n), np.nan)
    seq_idx, seq_lab, seq_tk = [], [], []
    rng = np.random.default_rng(seed + 909)

    for i_t, tk in enumerate(order):
        A_c = head.z(Ftr[tk]); y_c = ytr[tk]
        if arm == "noreplay":
            for _ in range(2 * EPOCHS):
                head.step(A_c, y_c)
        else:
            k = min(BUF, N_TR)
            idx = np.random.default_rng(seed + i_t).choice(N_TR, k, replace=False)
            seq_idx.extend(int(j) for j in idx)
            seq_lab.extend(int(ytr[tk][j]) for j in idx)
            seq_tk.extend([tk] * len(idx))
            A_r = np.stack([head.z(Ftr[seq_tk[q]])[seq_idx[q]]
                            for q in range(len(seq_idx))])
            y_r = np.array(seq_lab, np.int32)
            hc = max(1, len(y_c) // 2); hr = max(1, len(y_r) // 2)
            for _ in range(2 * EPOCHS):
                ic = rng.choice(len(y_c), hc, replace=False)
                ir = rng.choice(len(y_r), hr, replace=False)
                head.step(np.concatenate([A_c[ic], A_r[ir]]),
                          np.concatenate([y_c[ic], y_r[ir]]))
        for j_t, tk2 in enumerate(order):
            R[i_t, j_t] = head.acc(head.z(Fte[tk2]), yte[tk2])

    if arm == "replay_tail":
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
    meas = delta[:n - 1]      # divisor n-1, as the code has always done
    return {"meanF": float(np.mean(np.maximum(meas, 0)) * 100),
            "meanF_signed": float(np.mean(meas) * 100),
            "worstF": float(np.max(delta) * 100),
            "acq": float(np.mean(diag)), "final": float(np.mean(final)),
            "delta": [round(float(v) * 100, 2) for v in delta]}


def main():
    t0 = time.time()
    print("=" * 100)
    print("R7 ROUND 32 — TIER 3: the same protocol at T=6 instead of T=3")
    print("=" * 100)
    print(f"  tasks: {', '.join(TASKS)}")
    print(f"  12 fresh seeds (320001+6k), 3 orders, arms {ARMS}\n")

    cells = []
    for s_i, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in TASKS:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = feats(core, trx); ytr[tk] = try_
            Fte[tk] = feats(core, tex); yte[tk] = tey
        for order in ORDERS:
            rec = {"seed": seed, "order": list(order), "arms": {}}
            for arm in ARMS:
                rec["arms"][arm] = run(seed, order, Ftr, ytr, Fte, yte, arm)
            cells.append(rec)
        if (s_i + 1) % 3 == 0:
            print(f"  seed {s_i+1}/12 done ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 100)
    print("T=6 RESULTS BY ORDER (loss-based; gates meanF<=3, worstF<=5, acq>=0.65)")
    print("=" * 100)
    print(f"  {'order (first...last)':40s} {'meanF':>7} {'worstF':>8} {'acq':>7} "
          f"{'seeds OK':>9}")
    by_order = {}
    for order in ORDERS:
        sel = [c for c in cells if tuple(c["order"]) == order]
        a = "replay_tail"
        mf = float(np.mean([c["arms"][a]["meanF"] for c in sel]))
        wf = float(np.mean([c["arms"][a]["worstF"] for c in sel]))
        aq = float(np.mean([c["arms"][a]["acq"] for c in sel]))
        ok = sum(1 for c in sel
                 if c["arms"][a]["meanF"] <= 3 and c["arms"][a]["worstF"] <= 5
                 and c["arms"][a]["acq"] >= .65)
        key = f"{order[0]}..{order[-1]}"
        by_order[key] = {"meanF": mf, "worstF": wf, "acq": aq, "seeds_ok": int(ok)}
        print(f"  {key:40s} {mf:7.2f} {wf:8.2f} {aq:7.4f} {ok:6d}/12")

    best = min(by_order, key=lambda k: by_order[k]["meanF"])
    print(f"\n  best order: {best}  ({by_order[best]['seeds_ok']}/12 seeds pass every gate)")

    print("\n" + "=" * 100)
    print("T=3 vs T=6 COMPARISON (round 31 vs this round, best order each)")
    print("=" * 100)
    print(f"  T=3  meanF 0.42pp  worstF 0.65pp  acq 0.9037  15/15 seeds")
    print(f"  T=6  meanF {by_order[best]['meanF']:.2f}pp  worstF {by_order[best]['worstF']:.2f}pp"
          f"  acq {by_order[best]['acq']:.4f}  {by_order[best]['seeds_ok']}/12 seeds")

    print("\n" + "=" * 100)
    print("REPLAY ACQUISITION COST AT T=6")
    print("=" * 100)
    v = np.array([c["arms"]["replay_tail"]["acq"] - c["arms"]["noreplay"]["acq"]
                  for c in cells])
    se = v.std(ddof=1) / np.sqrt(len(v))
    lo, hi = v.mean() - 1.96 * se, v.mean() + 1.96 * se
    print(f"  paired over {len(v)} cells: {v.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  T=3 measured +0.0081pp ; linear head was -1.43pp")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    p1 = by_order[best]["meanF"] <= 3.0 and by_order[best]["seeds_ok"] >= 10
    p2 = abs(v.mean()) < 0.0100
    print(f"  P1 T=6 best order meanF<=3 and >=10/12 seeds : "
          f"{by_order[best]['meanF']:.2f}pp, {by_order[best]['seeds_ok']}/12  "
          f"{'CONFIRMED' if p1 else 'FAILED'}")
    print(f"  P2 replay cost under 1pp in absolute value    : {v.mean():+.4f}  "
          f"{'CONFIRMED' if p2 else 'FAILED'}")
    if p1:
        print(f"\n  => TIER 3 PASSES AT T=6. The result is not specific to the 3-task")
        print(f"     benchmark: doubling the task count keeps every gate.")
        verdict = "tier3_passes_t6"
    else:
        print(f"\n  => TIER 3 FAILS AT T=6. The gate is met only at the benchmark size")
        print(f"     I happened to choose. Reported as such.")
        verdict = "tier3_fails_t6"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t6-scale.json").write_text(json.dumps({
        "tasks": list(TASKS), "n_tasks": len(TASKS), "seeds": list(SEEDS),
        "orders": [list(o) for o in ORDERS], "arms": list(ARMS),
        "by_order": by_order, "best_order": best,
        "replay_acq_delta": {"mean": float(v.mean()), "ci95": [float(lo), float(hi)]},
        "predictions": {"P1_t6_gate": bool(p1), "P2_replay_free": bool(p2)},
        "verdict": verdict, "cells": cells,
        "n_tr_per_task": N_TR, "epochs": EPOCHS, "tail": TAIL,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: t6-scale.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
