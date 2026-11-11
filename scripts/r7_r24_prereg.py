"""R7 round 24 — PRE-REGISTERED test of the order lever on 15 FRESH seeds.

WHY THIS ROUND EXISTS
---------------------
Round 23 found that the task order `order_short > parity3 > sim_xor` reached the mission's
3pp mean-forgetting gate on 5 of 10 seeds, versus 8/60 cells overall. But that order was
chosen by looking at the same 60 cells it was measured on, so 5/10 is a SELECTION, not
evidence.

The brief's rule for a protocol change is: register it, then test on data not already seen.

So this round writes down the prediction BEFORE running, and uses seeds that have never
appeared in any round of this session.

PRE-REGISTERED PREDICTION
  Seeds: 240001 + 7k for k in 0..14   (15 fresh seeds, none used before)
  Order: order_short > parity3 > sim_xor   (fixed, chosen by round 23, not re-tuned)
  Mechanism: interleaved replay, buffer 128/task, 1600 total epochs (round 21/22 setting)
  Baseline: no_replay, same order, same seeds

  Prediction A: paired forgetting improvement replicates at approximately -17pp
  Prediction B: >= 5 of 15 seeds reach meanF <= 3pp  (round 23 saw 5/10)
  Prediction C: acquisition cost replicates at approximately -1.4pp (round 22's 60-cell value)

FALSIFIERS
  If the paired improvement collapses toward zero -> round 22's effect was also
    sample-specific and the mechanism line is wrong.
  If 0-2 of 15 seeds reach 3pp -> round 23's 5/10 was order-selection on the same data,
    and the gate is NOT controllable by order. I will report that.
  If acquisition cost disappears -> round 22's retraction was itself underpowered.

Reported regardless of which falsifier fires.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = tuple(240001 + 7 * k for k in range(15))
ORDER = ("order_short", "parity3", "sim_xor")     # PRE-REGISTERED
N_TR, N_TE = 384, 256
BUF = 128
EPOCHS = 800
GATE_MEAN = 3.0
GATE_WORST = 5.0
PRIOR_SEEDS = {140011 + 6 * k for k in range(10)} | {140011, 140017, 140023}

assert not (set(SEEDS) & PRIOR_SEEDS), "fresh seeds overlap with earlier rounds"


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


class Head:
    __slots__ = ("mu", "sd", "w", "b")

    def __init__(self, d):
        self.mu = None; self.sd = None
        self.w = np.zeros((2, d)); self.b = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 0.05)
        return (F - self.mu) / self.sd

    def _step(self, A, y, lr):
        p = softmax(A @ self.w.T + self.b)
        e = -p; e[np.arange(len(y)), y] += 1
        self.w += lr * (e.T @ A) / len(y)
        self.b += lr * e.mean(0)

    def acc(self, A, y):
        return float(np.mean(np.argmax(A @ self.w.T + self.b, 1) == y))


def run(seed, Ftr, ytr, Fte, yte, arm):
    head = Head(Ftr[ORDER[0]].shape[1])
    seq_idx, seq_lab, seq_tk = [], [], []
    n = len(ORDER)
    R = np.full((n, n), np.nan)
    rng = np.random.default_rng(seed + 4242)

    for i_t, tk in enumerate(ORDER):
        A_c = head.z(Ftr[tk]); y_c = ytr[tk]
        if arm == "no_replay":
            for _ in range(2 * EPOCHS):
                head._step(A_c, y_c, .5)
        else:
            idx = np.random.default_rng(seed + i_t).choice(
                N_TR, min(BUF, N_TR), replace=False)
            seq_idx.extend(int(j) for j in idx)
            seq_lab.extend(int(ytr[tk][j]) for j in idx)
            seq_tk.extend([tk] * len(idx))
            A_r = np.stack([head.z(Ftr[seq_tk[k]])[seq_idx[k]]
                            for k in range(len(seq_idx))])
            y_r = np.array(seq_lab, np.int32)
            half_c = max(1, len(y_c) // 2); half_r = max(1, len(y_r) // 2)
            for _ in range(2 * EPOCHS):
                ic = rng.choice(len(y_c), half_c, replace=False)
                ir = rng.choice(len(y_r), half_r, replace=False)
                head._step(np.concatenate([A_c[ic], A_r[ir]]),
                           np.concatenate([y_c[ic], y_r[ir]]), .5)

        for j_t, tk2 in enumerate(ORDER):
            R[i_t, j_t] = head.acc(head.z(Fte[tk2]), yte[tk2])

    diag = np.diag(R); final = R[n - 1]; fg = diag - final
    return {"meanF": float(np.mean(fg[:n - 1]) * 100),
            "worstF": float(np.max(fg) * 100),
            "acq": float(np.mean(diag)), "final": float(np.mean(final))}


def main():
    t0 = time.time()
    print("=" * 98)
    print("R7 ROUND 24 — PRE-REGISTERED order test on 15 FRESH seeds")
    print("=" * 98)
    print(f"  order  : {' > '.join(ORDER)}   (fixed by round 23, NOT re-tuned)")
    print(f"  seeds  : {SEEDS[0]} .. {SEEDS[-1]}  (none used in any earlier round)")
    print(f"  predicted: paired dF ~ -17pp; >=5/15 seeds reach 3pp; acq cost ~ -1.4pp\n")

    rows = []
    for k, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in ORDER:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = feats(core, trx); ytr[tk] = try_
            Fte[tk] = feats(core, tex); yte[tk] = tey
        a = run(seed, Ftr, ytr, Fte, yte, "no_replay")
        b = run(seed, Ftr, ytr, Fte, yte, "interleaved")
        rows.append({"seed": seed, "no_replay": a, "interleaved": b,
                     "dF": b["meanF"] - a["meanF"],
                     "dAcq": b["acq"] - a["acq"]})
        flag = "  <=GATE" if b["meanF"] <= GATE_MEAN else ""
        print(f"  {k+1:2d}/15 seed{seed}  no_rep {a['meanF']:7.2f}  "
              f"inter {b['meanF']:7.2f}  dF {b['meanF']-a['meanF']:+7.2f}  "
              f"worst {b['worstF']:6.2f}  acq {b['acq']:.4f}{flag}", flush=True)

    inter = np.array([r["interleaved"]["meanF"] for r in rows])
    dF = np.array([r["dF"] for r in rows])
    dA = np.array([r["dAcq"] for r in rows])
    worst = np.array([r["interleaved"]["worstF"] for r in rows])
    n_gate = int((inter <= GATE_MEAN).sum())
    n_worst = int((worst <= GATE_WORST).sum())
    both = int(((inter <= GATE_MEAN) & (worst <= GATE_WORST)).sum())

    def ci(x):
        se = x.std(ddof=1) / np.sqrt(len(x))
        # t(14) 95% = 2.145
        return x.mean(), x.mean() - 2.145 * se, x.mean() + 2.145 * se

    print("\n" + "=" * 98)
    print("PRE-REGISTERED PREDICTIONS vs RESULT")
    print("=" * 98)
    m, lo, hi = ci(dF)
    print(f"  A  paired forgetting: predicted ~-17pp | measured {m:+.2f}pp "
          f"CI [{lo:+.2f}, {hi:+.2f}]  "
          f"{'CONFIRMED' if hi < 0 else 'FAILED'}")
    print(f"  B  seeds reaching 3pp: predicted >=5/15 | measured {n_gate}/15  "
          f"{'CONFIRMED' if n_gate >= 5 else 'FAILED'}")
    ma, la, ha = ci(dA)
    print(f"  C  acquisition cost: predicted ~-1.4pp | measured {ma:+.4f} "
          f"CI [{la:+.4f}, {ha:+.4f}]  "
          f"{'CONFIRMED' if ha < 0 else 'FAILED (cost not detected on fresh seeds)'}")

    print(f"\n  absolute interleaved forgetting: mean {inter.mean():.2f}pp  "
          f"sd {inter.std(ddof=1):.2f}  min {inter.min():.2f}  max {inter.max():.2f}")
    print(f"  meanF<=3pp: {n_gate}/15   worstF<=5pp: {n_worst}/15   BOTH: {both}/15")

    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    if n_gate >= 5 and hi < 0:
        print(f"  The ORDER LEVER REPLICATES on fresh seeds: {n_gate}/15 reach 3pp and the")
        print(f"  paired improvement holds at {m:+.2f}pp. Round 23's 5/10 was not")
        print(f"  order-selection.")
        verdict = "order_lever_replicates"
    elif n_gate <= 2:
        print(f"  The ORDER LEVER DOES NOT REPLICATE: only {n_gate}/15 fresh seeds reach")
        print(f"  3pp. Round 23's 5/10 was order-selection on the same 60 cells, and the")
        print(f"  gate is NOT controllable by order.")
        verdict = "order_lever_fails"
    else:
        print(f"  PARTIAL: {n_gate}/15 fresh seeds reach 3pp, between round 23's rate")
        print(f"  and chance. The lever is weaker than the in-sample estimate.")
        verdict = "partial"
    print(f"\n  gate status: {'meanF met on ' + str(n_gate) + '/15 seeds, NOT every seed' if n_gate else 'not met'}")
    print(f"  wall {time.time()-t0:.0f}s")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prereg-order-fresh15.json").write_text(json.dumps({
        "preregistered_order": list(ORDER), "seeds": list(SEEDS),
        "predicted": {"dF_pp": -17, "seeds_at_3pp": ">=5/15", "acq_cost_pp": -1.4},
        "measured": {"dF": {"mean": float(m), "ci95": [float(lo), float(hi)]},
                     "acq": {"mean": float(ma), "ci95": [float(la), float(ha)]},
                     "n_at_3pp": n_gate, "n_worst_5pp": n_worst, "n_both": both,
                     "interleaved_mean": float(inter.mean()),
                     "interleaved_sd": float(inter.std(ddof=1))},
        "cells": rows, "verdict": verdict,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  written: prerad-order-fresh15.json" if False else
          f"  written: prereg-order-fresh15.json")


if __name__ == "__main__":
    main()
