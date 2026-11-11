"""R7 round 20 — is the replay effect REAL? 18 paired comparisons instead of 3.

ROUND 19 LEFT ME WITH A WEAK CLAIM
----------------------------------
Replay's improvement (31.90 -> 7.68pp) was a paired comparison on 3 seeds and ONE task
order. Round 19 then showed within-seed order variance has sd 3-6pp, which means a 3-point
paired difference is not evidence.

But round 19 also produced the tool to fix that: 18 cells (3 seeds x 6 orders). Running
replay and no-replay on ALL 18 gives 18 paired differences under identical seeds and
identical orders, which is exactly the design needed to separate the mechanism from noise.

DESIGN
  For each (seed, order) pair, run BOTH arms:
      no_replay      head retrained per task, no buffer
      replay         head retrained per task + replay buffer (128/task, weight 1)
  Identical data, identical initialisation, identical head algorithm, identical budget.

  Reports: paired difference per cell, mean, sd, and a paired t-interval.
  Also the same for the two ACCURACY gates, which round 19 showed are far outside noise.

FALSIFIER (registered): if the 18-cell paired mean difference is not significantly below
zero at 95%, or if the CI includes zero, then the replay effect is not established and
round 17's -24.22pp was a favourable draw.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
BASE = ("sim_xor", "parity3", "order_short")
BUF, WT = 128, 1
ORDERS = list(itertools.permutations(BASE))


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
    def __init__(self, d):
        self.mu = None; self.sd = None
        self.w = np.zeros((2, d)); self.b = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 0.05)
        return (F - self.mu) / self.sd

    def train(self, F, y, epochs=800, lr=.5):
        A = self.z(F)
        for _ in range(epochs):
            p = softmax(A @ self.w.T + self.b)
            e = -p; e[np.arange(len(y)), y] += 1
            self.w += lr * (e.T @ A) / len(y)
            self.b += lr * e.mean(0)

    def acc(self, F, y):
        return float(np.mean(np.argmax(self.z(F) @ self.w.T + self.b, 1) == y))


def run(seed, order, data, replay):
    core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
    head = Head(S.cfg(seed).hidden)
    buf_seq, buf_lab = [], []
    n = len(order)
    R = np.full((n, n), np.nan)

    for i_t, tk in enumerate(order):
        trx, try_, tex, tey = data[tk]
        head.train(feats(core, trx), try_)
        if replay:
            idx = np.random.default_rng(seed + i_t).choice(N_TR, min(BUF, N_TR),
                                                          replace=False)
            buf_seq.extend(trx[j].copy() for j in idx)
            buf_lab.extend(int(try_[j]) for j in idx)
            head.train(feats(core, np.stack(buf_seq)), np.array(buf_lab, np.int32),
                       epochs=800 * WT)
        for j_t, tk2 in enumerate(order):
            _, _, tex2, tey2 = data[tk2]
            R[i_t, j_t] = head.acc(feats(core, tex2), tey2)

    diag = np.diag(R); final = R[n - 1]; fg = diag - final
    return {"acq": float(np.mean(diag)), "final": float(np.mean(final)),
            "meanF_pp": float(np.mean(fg[:n - 1]) * 100),
            "worstF_pp": float(np.max(fg) * 100)}


def t_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    s = float(x.std(ddof=1))
    se = s / np.sqrt(n)
    # 95% two-sided, n-1 dof; n=18 -> t ~ 2.110
    tcrit = {3: 3.182, 6: 2.571, 12: 2.201, 18: 2.110}.get(n, 2.0)
    return m, s, se, (m - tcrit * se, m + tcrit * se)


def main():
    print("=" * 96)
    print("R7 ROUND 20 — replay vs no_replay, 18 PAIRED cells (3 seeds x 6 orders)")
    print("=" * 96)

    cells = []
    for seed in DEV:
        data = {}
        for tk in BASE:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            data[tk] = (trx, try_, tex, tey)
        for order in ORDERS:
            a = run(seed, order, data, replay=False)
            b = run(seed, order, data, replay=True)
            cells.append({"seed": seed, "order": list(order),
                          "no_replay": a, "replay": b,
                          "d_meanF": b["meanF_pp"] - a["meanF_pp"],
                          "d_acq": b["acq"] - a["acq"],
                          "d_final": b["final"] - a["final"]})
            print(f"  seed{seed} {'>'.join(o[:4] for o in order):22s} "
                  f"no_rep {a['meanF_pp']:6.2f}  rep {b['meanF_pp']:6.2f}  "
                  f"d {b['meanF_pp']-a['meanF_pp']:+7.2f}pp", flush=True)

    print("\n" + "=" * 96)
    print("PAIRED ANALYSIS OVER 18 CELLS")
    print("=" * 96)
    for key, label in (("d_meanF", "mean forgetting (pp)"),
                       ("d_acq", "acquisition"),
                       ("d_final", "final accuracy")):
        v = [c[key] for c in cells]
        m, s, se, (lo, hi) = t_ci(v)
        better = sum(1 for x in v if (x < 0 if key == "d_meanF" else x > 0))
        unit = "pp" if key == "d_meanF" else ""
        print(f"\n  {label}")
        print(f"    paired mean {m:+.4f}{unit}   sd {s:.4f}   se {se:.4f}")
        print(f"    95% CI      [{lo:+.4f}, {hi:+.4f}]{unit}")
        print(f"    cells improved: {better}/18")
        sig = (hi < 0) if key == "d_meanF" else (lo > 0)
        print(f"    excludes zero: {sig}")

    v = [c["d_meanF"] for c in cells]
    m, s, se, (lo, hi) = t_ci(v)
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  replay changes mean forgetting by {m:+.2f}pp "
          f"(95% CI [{lo:+.2f}, {hi:+.2f}])")
    if hi < 0:
        print("  => REPLAY EFFECT IS ESTABLISHED. The CI excludes zero over 18 paired")
        print("     cells spanning 3 seeds and all 6 task orders.")
        verdict = "established"
    else:
        print("  => REPLAY EFFECT NOT ESTABLISHED. The CI includes zero, so round 17's")
        print("     -24.22pp was a favourable draw.")
        verdict = "not_established"
    print(f"\n  noise context from round 19: within-seed order sd was 3-6pp. This test")
    print(f"  compares replay against no_replay WITHIN each seed-order cell, so that")
    print(f"  variance cancels and only the mechanism difference remains.")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay-paired-18.json").write_text(
        json.dumps({"cells": cells,
                    "meanF": {"mean": m, "sd": s, "se": se, "ci95": [lo, hi]},
                    "verdict": verdict}, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/replay-paired-18.json")


if __name__ == "__main__":
    main()
