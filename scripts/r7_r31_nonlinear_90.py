"""R7 round 31 — the two open questions from round 30, with the design round 22 taught.

ROUND 30 LEFT TWO THINGS UNTESTED
---------------------------------
Q1  NO-TRADE-OFF. `mlp32_noreplay` had acquisition 0.9642 vs `mlp32` 0.9635 — a 0.07pp gap
    that cannot resolve anything. Round 22 showed a 1-2pp effect needs the paired-across-orders
    design, not 3 seeds and one order. So: is interleaved replay free on a nonlinear head, or
    does it cost acquisition like it did on the linear head (-1.43pp over 60 cells)?

Q2  THE FAILING SEED. 240099 missed by 0.12pp on meanF and 0.08pp on worstF, with parity3 as
    the degraded task. Is that (a) a genuine tail event, or (b) an artefact of the single
    fixed order `order_short > parity3 > sim_xor`? Round 23/24 established order is worth
    ~7pp on the linear head; it has never been tested on a nonlinear one.

DESIGN (both questions with one run)
    15 fresh seeds x 6 task orders = 90 cells, mlp32 head, three arms at matched budget:
        noreplay   current task only, 1600 epochs
        replay     interleaved current+replay, 1600 epochs
        replay_tail  replay + 300-epoch tail re-consolidation of the earliest task

    Q1 is answered by the paired `replay - noreplay` acquisition delta over 90 cells.
    Q2 is answered by the distribution of meanF across the 6 orders for each seed.

REGISTERED PREDICTIONS
    P1  replay's acquisition cost on the nonlinear head is smaller than the linear head's
        -1.43pp, because a head that fits to 0.96 has more slack.
    P2  the best order reaches the gate on >= 12 of 15 seeds (round 30's 14/15 was one order).
    P3  seed 240099 is NOT a fixed property: under a different order it falls inside the gate.

FALSIFIERS
    If the acquisition delta CI excludes zero and is <= -1.0pp, replay is NOT free and the
    no-trade-off rule fails on this head too.
    If no order reaches the gate on more than 10/15 seeds, round 30's 14/15 was favourable.

All reporting uses the LOSS-based aggregation (rounds 25/26), both gates, every seed.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = tuple(240001 + 7 * k for k in range(15))
BASE = ("sim_xor", "parity3", "order_short")
ORDERS = list(itertools.permutations(BASE))
N_TR, N_TE = 384, 256
BUF = 128
EPOCHS = 800
TAIL = 300
ARMS = ("noreplay", "replay", "replay_tail")


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


GENS = {"sim_xor": sim_xor, "parity3": parity3, "order_short": order_short}


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
    rng = np.random.default_rng(seed + 4242)

    for i_t, tk in enumerate(order):
        A_c = head.z(Ftr[tk]); y_c = ytr[tk]
        if arm == "noreplay":
            for _ in range(2 * EPOCHS):
                head.step(A_c, y_c)
        else:
            idx = np.random.default_rng(seed + i_t).choice(N_TR, BUF, replace=False)
            seq_idx.extend(int(j) for j in idx)
            seq_lab.extend(int(ytr[tk][j]) for j in idx)
            seq_tk.extend([tk] * len(idx))
            A_r = np.stack([head.z(Ftr[seq_tk[k]])[seq_idx[k]]
                            for k in range(len(seq_idx))])
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
        A1 = head.z(Ftr[order[0]])[seq_idx[:BUF]]
        y1 = np.array(seq_lab[:BUF], np.int32)
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
            "acq": float(np.mean(diag)), "final": float(np.mean(final)),
            "delta": delta.tolist(), "order": list(order)}


def ci95(v, tcrit):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - tcrit * se), float(v.mean() + tcrit * se)


def main():
    t0 = time.time()
    print("=" * 100)
    print("R7 ROUND 31 — 15 seeds x 6 orders = 90 cells, mlp32, three arms")
    print("=" * 100)
    print(f"  Q1: is interleaved replay free on a nonlinear head?")
    print(f"  Q2: is round 30's 14/15 order-specific, and is seed 240099 a tail event?\n")

    cells = []
    for s_i, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in BASE:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = feats(core, trx); ytr[tk] = try_
            Fte[tk] = feats(core, tex); yte[tk] = tey
        for order in ORDERS:
            rec = {"seed": seed, "order": list(order), "arms": {}}
            for arm in ARMS:
                rec["arms"][arm] = run(seed, order, Ftr, ytr, Fte, yte, arm)
            cells.append(rec)
        print(f"  seed {s_i+1}/15 {seed} done ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 100)
    print("Q1 — ACQUISITION COST OF REPLAY ON A NONLINEAR HEAD (paired, 90 cells)")
    print("=" * 100)
    for arm in ("replay", "replay_tail"):
        v = np.array([c["arms"][arm]["acq"] - c["arms"]["noreplay"]["acq"]
                      for c in cells])
        m, lo, hi = ci95(v, 1.96)
        better = int((v > 0).sum())
        print(f"  {arm:12s} vs noreplay: {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{better}/90 cells higher")
    print(f"  round 22's linear-head cost was -1.43pp (CI [-2.35, -0.50])")

    print("\n" + "=" * 100)
    print("Q2 — GATE REACH BY ORDER (meanF<=3 and worstF<=5 and acq>=0.65)")
    print("=" * 100)
    print(f"  {'order':36s} {'meanF':>7} {'worstF':>8} {'acq':>7} {'seeds OK':>9}")
    by_order = {}
    for order in ORDERS:
        sel = [c for c in cells if tuple(c["order"]) == order]
        for arm in ("replay_tail",):
            mf = np.mean([c["arms"][arm]["meanF"] for c in sel])
            wf = np.mean([c["arms"][arm]["worstF"] for c in sel])
            aq = np.mean([c["arms"][arm]["acq"] for c in sel])
            ok = sum(1 for c in sel
                     if c["arms"][arm]["meanF"] <= 3
                     and c["arms"][arm]["worstF"] <= 5
                     and c["arms"][arm]["acq"] >= .65)
            by_order[" > ".join(o[:4] for o in order)] = {
                "meanF": float(mf), "worstF": float(wf), "acq": float(aq),
                "seeds_ok": int(ok)}
            print(f"  {' > '.join(o[:4] for o in order):36s} {mf:7.2f} {wf:8.2f} "
                  f"{aq:7.4f} {ok:6d}/15")

    best_order = min(by_order, key=lambda k: by_order[k]["meanF"])
    print(f"\n  best order: {best_order}  "
          f"({by_order[best_order]['seeds_ok']}/15 seeds pass all three gates)")

    print("\n" + "=" * 100)
    print("SEED 240099 ACROSS ITS SIX ORDERS")
    print("=" * 100)
    for c in cells:
        if c["seed"] == 240099:
            a = c["arms"]["replay_tail"]
            ok = (a["meanF"] <= 3 and a["worstF"] <= 5 and a["acq"] >= .65)
            print(f"  {' > '.join(o[:4] for o in c['order']):36s} "
                  f"meanF {a['meanF']:6.2f}  worstF {a['worstF']:6.2f}  "
                  f"{'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    v = np.array([c["arms"]["replay_tail"]["acq"] - c["arms"]["noreplay"]["acq"]
                  for c in cells])
    m, lo, hi = ci95(v, 1.96)
    p1 = not (hi < 0 and m <= -0.010)
    p2 = by_order[best_order]["seeds_ok"] >= 12
    n99 = sum(1 for c in cells if c["seed"] == 240099
              and c["arms"]["replay_tail"]["meanF"] <= 3
              and c["arms"]["replay_tail"]["worstF"] <= 5)
    p3 = n99 >= 1
    print(f"  P1 replay cost smaller than -1.0pp : measured {m:+.4f} CI [{lo:+.4f},{hi:+.4f}]"
          f"  {'CONFIRMED' if p1 else 'FAILED'}")
    print(f"  P2 best order reaches gate >=12/15 : measured "
          f"{by_order[best_order]['seeds_ok']}/15  {'CONFIRMED' if p2 else 'FAILED'}")
    print(f"  P3 seed 240099 passes under some order: {n99}/6 orders "
          f"{'CONFIRMED' if p3 else 'FAILED'}")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "nonlinear-90cells.json").write_text(json.dumps({
        "seeds": list(SEEDS), "orders": [list(o) for o in ORDERS],
        "arms": list(ARMS), "by_order": by_order,
        "best_order": best_order,
        "replay_acq_delta": {"mean": m, "ci95": [lo, hi]},
        "predictions": {"P1_free_replay": bool(p1), "P2_order_ge12": bool(p2),
                        "P3_seed240099_movable": bool(p3)},
        "seed240099_orders_ok": int(n99),
        "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: nonlinear-90cells.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
