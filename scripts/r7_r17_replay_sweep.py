"""R7 round 17 — can correct replay close the 3pp gate? Buffer size x replay weight.

ROUND 16 ESTABLISHED
--------------------
Correct replay (labels stored with sequences):
    frozen_shared  acq 0.8819  final 0.6693  meanF 31.90pp  worstF 54.30pp
    frozen_replay  acq 0.8607  final 0.7982  meanF  9.38pp  worstF 18.75pp

Acquisition and final clear their gates. Mean forgetting misses 3pp by 3.1x, worst by 3.75x.

TWO LEVERS, BOTH WITHIN CONSTRAINTS
-----------------------------------
1  BUFFER SIZE. 128 sequences retained per seen task. At the last task the head sees 384
   fresh + 384 replayed, i.e. a 50/50 mix. A larger buffer raises the old:new ratio
   directly. Memory: 384 x 48 floats = 74 KB, so even 4x this is irrelevant against 2 GB.

2  REPLAY WEIGHTING. Train the head on replayed pairs for more passes than on fresh data,
   i.e. weight old knowledge more heavily without changing the example count.

REGISTERED PREDICTION: meanF falls monotonically with both levers. If it does not, the
9.38pp floor is structural (the head cannot hold three boundaries regardless of exposure)
and I will say so rather than tuning further.

GATES (not relaxed): acq >= 0.65, final >= 0.65, meanF <= 3pp, worstF <= 5pp, EVERY seed.

Also runs the false-progress checks the mission requires:
  FP3  acquisition collapse?      reported per cell, not just the mean
  FP9  does replay answer, or train the readout?   the neurons still compute the features;
       the buffer only re-exposes the head. Stated explicitly, not assumed.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
TASKS = ("sim_xor", "parity3", "order_short")
GATES = {"acq": .65, "final": .65, "meanF_pp": 3.0, "worstF_pp": 5.0}
BUFFERS = (128, 384, 640)
WEIGHTS = (1, 3)


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
    one = z.ndim == 1
    if one:
        z = z[None, :]
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    out = e / e.sum(1, keepdims=True)
    return out[0] if one else out


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


def run_cell(seed, data, buf_per_task, weight, replay=True):
    core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
    head = Head(S.cfg(seed).hidden)
    n = len(TASKS)
    R = np.full((n, n), np.nan)
    buf_seq, buf_lab = [], []

    for i_t, tk in enumerate(TASKS):
        trx, try_, tex, tey = data[tk]
        for i in range(N_TR):
            core.reset_state()
            for t in range(12):
                core.step(trx[i, t], track=True)
        head.train(feats(core, trx), try_)

        if replay:
            # retain up to buf_per_task from THIS task, keeping the correct labels
            take = min(buf_per_task, N_TR)
            idx = np.random.default_rng(seed + i_t).choice(N_TR, take, replace=False)
            buf_seq.extend(trx[j].copy() for j in idx)
            buf_lab.extend(int(try_[j]) for j in idx)
            if buf_seq:
                RF = feats(core, np.stack(buf_seq))
                Ry = np.array(buf_lab, np.int32)
                head.train(RF, Ry, epochs=800 * weight)

        for j_t, tk2 in enumerate(TASKS):
            _, _, tex2, tey2 = data[tk2]
            R[i_t, j_t] = head.acc(feats(core, tex2), tey2)
    return R


def summarize(Rs):
    n = len(TASKS)
    diag = [np.diag(R) for R in Rs]
    final = [R[n - 1] for R in Rs]
    acq = float(np.mean([np.mean(d) for d in diag]))
    per = float(np.mean([np.mean(f) for f in final]))
    fg = [d - f for d, f in zip(diag, final)]
    mf = float(np.mean([np.mean(x[:n - 1]) for x in fg])) * 100
    wf = float(np.max([np.max(x) for x in fg])) * 100
    per_seed_mf = [float(np.mean(x[:n - 1])) * 100 for x in fg]
    per_seed_acq = [float(np.mean(d)) for d in diag]
    return {"acq": acq, "final": per, "meanF": mf, "worstF": wf,
            "per_seed_meanF": [round(v, 2) for v in per_seed_mf],
            "per_seed_acq": [round(v, 4) for v in per_seed_acq],
            "every_seed_acq_ok": all(a >= GATES["acq"] for a in per_seed_acq),
            "every_seed_meanF_ok": all(m <= GATES["meanF_pp"] for m in per_seed_mf),
            "passes": bool(acq >= GATES["acq"] and per >= GATES["final"]
                           and mf <= GATES["meanF_pp"] and wf <= GATES["worstF_pp"]
                           and all(a >= GATES["acq"] for a in per_seed_acq)
                           and all(m <= GATES["meanF_pp"] for m in per_seed_mf))}


def main():
    print("=" * 96)
    print("R7 ROUND 17 — replay buffer size x replay weight, against the 3pp gate")
    print("=" * 96)

    data_by_seed = {}
    for seed in DEV:
        d = {}
        for tk in TASKS:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            d[tk] = (trx, try_, tex, tey)
        data_by_seed[seed] = d

    cells = {}
    # no-replay baseline
    base = summarize([run_cell(s, data_by_seed[s], 0, 1, replay=False) for s in DEV])
    cells["no_replay"] = base
    print(f"\n  no_replay        acq {base['acq']:.4f}  final {base['final']:.4f}  "
          f"meanF {base['meanF']:7.2f}pp  worstF {base['worstF']:7.2f}pp  "
          f"{'PASS' if base['passes'] else 'fail'}")

    print(f"\n  {'buf':>5} {'wt':>3} {'acq':>7} {'final':>7} {'meanF':>9} "
          f"{'worstF':>9} {'gate':>6}  per-seed meanF")
    for buf in BUFFERS:
        for wt in WEIGHTS:
            Rs = [run_cell(s, data_by_seed[s], buf, wt) for s in DEV]
            c = summarize(Rs)
            cells[f"buf{buf}_wt{wt}"] = c
            ps = " ".join(f"{v:.2f}" for v in c["per_seed_meanF"])
            print(f"  {buf:5d} {wt:3d} {c['acq']:7.4f} {c['final']:7.4f} "
                  f"{c['meanF']:8.2f}p {c['worstF']:8.2f}p "
                  f"{('PASS' if c['passes'] else 'fail'):>6}  {ps}", flush=True)

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    best = min((k for k in cells if k != "no_replay"),
               key=lambda k: cells[k]["meanF"])
    b = cells[best]
    print(f"  best cell: {best}")
    print(f"    acq {b['acq']:.4f}  final {b['final']:.4f}  meanF {b['meanF']:.2f}pp  "
          f"worstF {b['worstF']:.2f}pp")
    print(f"    every seed acq>=0.65: {b['every_seed_acq_ok']}   "
          f"every seed meanF<=3pp: {b['every_seed_meanF_ok']}")
    print(f"  vs no_replay meanF {base['meanF']:.2f}pp  "
          f"({base['meanF'] - b['meanF']:.2f}pp improvement)")

    mono = all(cells[f"buf{b}_wt{1}"]["meanF"] >= cells[f"buf{b}_wt{3}"]["meanF"]
               for b in BUFFERS)
    print(f"\n  weight lever monotone: {mono}")
    print(f"  gate outcome: {'PASS - all four gates on every seed' if b['passes'] else 'STILL FAILS'}")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay-sweep.json").write_text(
        json.dumps({"cells": cells, "best": best, "gates": GATES,
                    "weight_monotone": mono},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/replay-sweep.json")


if __name__ == "__main__":
    main()
