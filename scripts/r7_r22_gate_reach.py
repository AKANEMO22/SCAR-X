"""R7 round 22 — is the 3pp forgetting gate reachable at ALL? 10 seeds x 6 orders.

THE QUESTION THIS ANSWERS
-------------------------
Round 21 established interleaved replay: forgetting -15.23pp, acquisition +0.14pp, 18/18
cells. But the ABSOLUTE level lands around 15-16pp against a 3pp gate, and round 19
measured the absolute level's noise floor at sd 5.36pp with only 3 seeds.

So the mission's central retention requirement is unresolved in BOTH directions:
  - no configuration has been shown to reach 3pp
  - and 3 seeds cannot tell 3pp from 9pp

This round settles it by scale. 10 seeds x 6 task orders = 60 cells under the best known
mechanism, reporting the FULL DISTRIBUTION rather than a mean.

ANSWER FORMAT (registered before running)
  If ZERO of 60 cells reach meanF <= 3pp -> the gate is not reachable by readout-side
     replay on this suite, and that is a measured bound, not a preference.
  If SOME cells reach it -> the gate IS reachable and the problem is variance control,
     which is a different (and more tractable) finding.

EFFICIENCY: the hidden core is frozen, so per-task features depend only on (seed, task) and
not on arm or order. They are computed ONCE per (seed, task) and reused by every arm and
every order. That is what makes 120 runs feasible.

ARMS
  no_replay    head retrained per task, 1600 epochs current
  interleaved  1600 epochs, each epoch drawing half current / half replay (round 21)
Both at identical total gradient budget.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = tuple(140011 + 6 * k for k in range(10))
N_TR, N_TE = 384, 256
BASE = ("sim_xor", "parity3", "order_short")
BUF = 128
EPOCHS = 800          # per source; both arms total 2*EPOCHS
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

    def train(self, A, y, epochs=2 * EPOCHS, lr=.5):
        for _ in range(epochs):
            self._step(A, y, lr)

    def acc(self, A, y):
        return float(np.mean(np.argmax(A @ self.w.T + self.b, 1) == y))


def run(seed, order, Ftr, ytr, Fte, yte, arm):
    head = Head(Ftr[order[0]].shape[1])
    buf_idx, buf_lab, buf_tk = [], [], []
    n = len(order)
    R = np.full((n, n), np.nan)
    rng = np.random.default_rng(seed + 999)

    for i_t, tk in enumerate(order):
        A_c = head.z(Ftr[tk])
        y_c = ytr[tk]

        if arm == "no_replay":
            head.train(A_c, y_c, epochs=2 * EPOCHS)
        else:
            idx = np.random.default_rng(seed + i_t).choice(
                N_TR, min(BUF, N_TR), replace=False)
            buf_idx.extend(int(j) for j in idx)
            buf_lab.extend(int(ytr[tk][j]) for j in idx)
            buf_tk.extend([tk] * len(idx))
            A_r = np.stack([head.z(Ftr[buf_tk[k]])[buf_idx[k]]
                            for k in range(len(buf_idx))])
            y_r = np.array(buf_lab, np.int32)
            nc, nr = len(y_c), len(y_r)
            half_c = max(1, nc // 2); half_r = max(1, nr // 2)
            for _ in range(2 * EPOCHS):
                ic = rng.choice(nc, half_c, replace=False)
                ir = rng.choice(nr, half_r, replace=False)
                head._step(np.concatenate([A_c[ic], A_r[ir]]),
                           np.concatenate([y_c[ic], y_r[ir]]), .5)

        for j_t, tk2 in enumerate(order):
            R[i_t, j_t] = head.acc(head.z(Fte[tk2]), yte[tk2])

    diag = np.diag(R); final = R[n - 1]; fg = diag - final
    return float(np.mean(fg[:n - 1]) * 100), float(np.mean(diag)), float(np.mean(final)), \
        float(np.max(fg) * 100)


def main():
    t0 = time.time()
    print("=" * 100)
    print("R7 ROUND 22 — is the 3pp gate reachable? 10 seeds x 6 orders = 60 cells")
    print("=" * 100)

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
            r = {}
            for arm in ("no_replay", "interleaved"):
                mf, acq, fin, wf = run(seed, order, Ftr, ytr, Fte, yte, arm)
                r[arm] = {"meanF": mf, "acq": acq, "final": fin, "worstF": wf}
            r["seed"] = seed; r["order"] = list(order)
            r["d_meanF"] = r["interleaved"]["meanF"] - r["no_replay"]["meanF"]
            r["d_acq"] = r["interleaved"]["acq"] - r["no_replay"]["acq"]
            cells.append(r)
        print(f"  seed {s_i+1}/10 {seed} done "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 100)
    print("DISTRIBUTION OF ABSOLUTE MEAN FORGETTING (60 cells each arm)")
    print("=" * 100)
    for arm in ("no_replay", "interleaved"):
        v = np.array([c[arm]["meanF"] for c in cells])
        print(f"\n  {arm}")
        print(f"    mean {v.mean():7.2f}pp   sd {v.std(ddof=1):6.2f}   "
              f"min {v.min():7.2f}   median {np.median(v):7.2f}   max {v.max():7.2f}")
        for th in (3, 5, 10):
            n = int((v <= th).sum())
            print(f"    cells at or below {th:2d}pp: {n:2d}/60  ({n/60*100:.0f}%)")

    print("\n" + "=" * 100)
    print("THE GATE QUESTION")
    print("=" * 100)
    vi = np.array([c["interleaved"]["meanF"] for c in cells])
    vw = np.array([c["interleaved"]["worstF"] for c in cells])
    reach_mean = int((vi <= 3).sum())
    reach_worst = int((vw <= 5).sum())
    both = int(((vi <= 3) & (vw <= 5)).sum())
    print(f"  interleaved cells meeting meanF<=3pp : {reach_mean}/60")
    print(f"  interleaved cells meeting worstF<=5pp: {reach_worst}/60")
    print(f"  interleaved cells meeting BOTH       : {both}/60")

    if reach_mean == 0:
        print("\n  => ZERO cells reach 3pp. The gate is NOT reachable by readout-side")
        print("     replay on this suite. That is a measured bound over 10 seeds and all")
        print("     6 task orders, not a preference or a single unfavourable draw.")
        verdict = "gate_unreachable"
    else:
        best = min(cells, key=lambda c: c["interleaved"]["meanF"])
        print(f"\n  => {reach_mean}/60 cells reach 3pp. The gate IS reachable.")
        print(f"     Best cell: seed {best['seed']} order "
              f"{'>'.join(o[:4] for o in best['order'])} "
              f"at {best['interleaved']['meanF']:.2f}pp")
        print(f"     So the problem is VARIANCE CONTROL, not an unreachable threshold.")
        verdict = "gate_reachable"

    d = np.array([c["d_meanF"] for c in cells])
    da = np.array([c["d_acq"] for c in cells])
    print(f"\n  paired mechanism effect over 60 cells:")
    print(f"    forgetting  {d.mean():+7.2f}pp  sd {d.std(ddof=1):5.2f}  "
          f"{int((d < 0).sum())}/60 improved")
    print(f"    acquisition {da.mean():+7.4f}  sd {da.std(ddof=1):5.4f}  "
          f"se {da.std(ddof=1)/np.sqrt(60):.4f}  "
          f"95%CI [{da.mean()-1.96*da.std(ddof=1)/np.sqrt(60):+.4f}, "
          f"{da.mean()+1.96*da.std(ddof=1)/np.sqrt(60):+.4f}]")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate-reachability-60.json").write_text(
        json.dumps({"cells": cells, "seeds": list(SEEDS),
                    "orders": [list(o) for o in ORDERS],
                    "reach_mean3": reach_mean, "reach_worst5": reach_worst,
                    "reach_both": both, "verdict": verdict,
                    "paired_forgetting": {"mean": float(d.mean()),
                                          "sd": float(d.std(ddof=1))},
                    "paired_acquisition": {"mean": float(da.mean()),
                                           "sd": float(da.std(ddof=1))}},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/gate-reachability-60.json")
    print(f"  wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
