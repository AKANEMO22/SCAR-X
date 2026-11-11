"""R7 round 19 — is the seed spread ORDER-driven or SEED-driven, and can a 3pp gate resolve?

ROUND 18 FOUND A PROTOCOL PROBLEM
---------------------------------
Same protocol, same mechanism, three seeds:
    140011  5.66pp   140017  5.86pp   140023  11.52pp
Spread = 5.86pp = TWICE the mission's 3pp mean-forgetting gate.

If that is seed-to-seed noise, then a 3pp gate cannot be resolved with three seeds and any
PASS/FAIL is a draw from a distribution wider than the threshold.

TWO COMPETING EXPLANATIONS
--------------------------
H-order   The 11.52pp belongs to the TASK SEQUENCE, not the initialisation. Seed 140023
          happens to sit at a position where the sequence hurts more. Predicts: permuting
          order moves the outlier, and within-seed order variance is large.

H-seed    The 11.52pp belongs to that initialisation's representation. Predicts: order
          permutation does not move it much, and within-seed variance is small relative to
          between-seed variance.

DESIGN
  For each seed, run all 6 task orders (3! = 6). 3 seeds x 6 orders = 18 cells.
  Same replay setting throughout (buf 128, wt 1, the round-17 best).
  Report the decomposition: variance across seeds vs variance across orders within seed.

If order variance dominates -> the "worst seed" framing in round 18 was wrong, and the
11.52pp was an order effect I attributed to an initialisation.

This also directly answers whether a 3pp gate is measurable: if the pooled spread exceeds
3pp under an identical mechanism, it is not.
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
BASE_TASKS = ("sim_xor", "parity3", "order_short")
BUF, WT = 128, 1
ORDERS = list(itertools.permutations(BASE_TASKS))


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


def run_order(seed, order, data):
    core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
    head = Head(S.cfg(seed).hidden)
    buf_seq, buf_lab = [], []
    n = len(order)
    R = np.full((n, n), np.nan)

    for i_t, tk in enumerate(order):
        trx, try_, tex, tey = data[tk]
        head.train(feats(core, trx), try_)
        idx = np.random.default_rng(seed + i_t).choice(N_TR, min(BUF, N_TR),
                                                     replace=False)
        buf_seq.extend(trx[j].copy() for j in idx)
        buf_lab.extend(int(try_[j]) for j in idx)
        if buf_seq:
            head.train(feats(core, np.stack(buf_seq)), np.array(buf_lab, np.int32),
                       epochs=800 * WT)
        for j_t, tk2 in enumerate(order):
            _, _, tex2, tey2 = data[tk2]
            R[i_t, j_t] = head.acc(feats(core, tex2), tey2)

    diag = np.diag(R)
    final = R[n - 1]
    fg = diag - final
    return {"acq": float(np.mean(diag)), "final": float(np.mean(final)),
            "meanF_pp": float(np.mean(fg[:n - 1]) * 100),
            "worstF_pp": float(np.max(fg) * 100)}


def main():
    print("=" * 96)
    print("R7 ROUND 19 — 3 seeds x 6 task orders = 18 cells, identical mechanism")
    print("=" * 96)

    cells = []
    for seed in DEV:
        data = {}
        for tk in BASE_TASKS:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            data[tk] = (trx, try_, tex, tey)
        for order in ORDERS:
            r = run_order(seed, order, data)
            cells.append({"seed": seed, "order": list(order), **r})
            print(f"  seed{seed} {'>'.join(o[:4] for o in order):22s} "
                  f"acq {r['acq']:.4f} final {r['final']:.4f} "
                  f"meanF {r['meanF_pp']:6.2f}pp worstF {r['worstF_pp']:6.2f}pp",
                  flush=True)

    print("\n" + "=" * 96)
    print("VARIANCE DECOMPOSITION")
    print("=" * 96)
    mfs = np.array([c["meanF_pp"] for c in cells])
    print(f"  pooled mean {mfs.mean():.2f}pp  sd {mfs.std(ddof=1):.2f}pp  "
          f"min {mfs.min():.2f}  max {mfs.max():.2f}")

    seed_means = {s: np.mean([c["meanF_pp"] for c in cells if c["seed"] == s])
                  for s in DEV}
    order_means = {" > ".join(o): np.mean([c["meanF_pp"] for c in cells
                                           if tuple(c["order"]) == o])
                   for o in ORDERS}
    between_seed = float(np.var([seed_means[s] for s in DEV], ddof=1))
    between_order = float(np.var(list(order_means.values()), ddof=1))
    within = float(np.mean([np.var([c["meanF_pp"] for c in cells if c["seed"] == s],
                                   ddof=1) for s in DEV]))

    print(f"\n  by seed:")
    for s in DEV:
        vals = [c["meanF_pp"] for c in cells if c["seed"] == s]
        print(f"    {s}  mean {seed_means[s]:6.2f}pp  "
              f"range [{min(vals):5.2f}, {max(vals):5.2f}]  sd {np.std(vals, ddof=1):5.2f}")
    print(f"\n  by order:")
    for k, v in sorted(order_means.items(), key=lambda kv: kv[1]):
        print(f"    {k:30s} {v:6.2f}pp")

    print(f"\n  variance across seeds (of seed means)  : {between_seed:8.2f}")
    print(f"  variance across orders (of order means): {between_order:8.2f}")
    print(f"  mean within-seed variance              : {within:8.2f}")

    gate = 3.0
    resolvable = (mfs.std(ddof=1) < gate / 2)
    print(f"\n  gate {gate}pp; pooled sd {mfs.std(ddof=1):.2f}pp; "
          f"resolvable at 2-sigma: {'NO' if not resolvable else 'YES'}")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    if between_order > between_seed:
        print("  ORDER DOMINATES. The 11.52pp outlier in round 18 was an order effect")
        print("  I attributed to an initialisation. The 'worst seed' framing was wrong.")
        verdict = "order_dominates"
    else:
        print("  SEED DOMINATES. The initialisation, not the order, drives the spread.")
        verdict = "seed_dominates"

    print(f"\n  Even so: with pooled sd {mfs.std(ddof=1):.2f}pp against a {gate}pp gate,")
    print(f"  a single-seed PASS cannot be distinguished from noise unless the")
    print(f"  observed meanF is below {gate - 2*mfs.std(ddof=1):.2f}pp.")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "order-variance.json").write_text(
        json.dumps({"cells": cells, "seed_means": seed_means,
                    "order_means": order_means,
                    "var_between_seed": between_seed,
                    "var_between_order": between_order,
                    "var_within_seed": within,
                    "pooled_sd": float(mfs.std(ddof=1)),
                    "gate_pp": gate, "resolvable": bool(resolvable),
                    "verdict": verdict}, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/order-variance.json")


if __name__ == "__main__":
    main()
