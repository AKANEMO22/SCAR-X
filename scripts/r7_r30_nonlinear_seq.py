"""R7 round 30 — the registered test: actual SEQUENTIAL forgetting with a nonlinear readout.

WHAT ROUND 29 DID AND DID NOT SHOW
----------------------------------
Round 29 measured `solo - joint` (the co-representation floor) for three head forms:
    linear  6.52pp    wide16  8.79pp    mlp32  0.36pp
So the FLOOR is head-expressiveness, and a nonlinear head has essentially none.

But that is not a retention result. Round 29 never ran the SEQUENTIAL protocol with a nonlinear
head. Round 24 showed the LINEAR head has, on top of its floor, a large sequential interference
term. The question round 30 answers:

    with a nonlinear readout, does sequential training + interleaved replay reach
    meanF <= 3pp AND worstF <= 5pp on EVERY seed, with acquisition >= 0.65?

REGISTERED IN ROUND 29, BEFORE THIS RUN
    Prediction: yes, because the nonlinear floor is 0.36pp, leaving ~2.6pp of headroom below
    the 3pp gate for whatever sequential interference remains.
    Falsifier: if the nonlinear head's sequential forgetting exceeds its own joint floor by
    more than ~3pp, capacity fixes co-representation but NOT sequential interference, and the
    mechanism question reopens at a higher floor.

ARMS
    linear          round 24's head (2-logit), with interleaved replay + tail  -> reference
    mlp32           tanh 32 units, same protocol, same budget
    mlp32_replay    tanh 32 units, NO replay (does the mechanism still matter?)

METRICS (both aggregations, as rounds 25/26 established)
    meanF_loss = mean over j of max(delta_j, 0)   <- the mission's reading
    worstF     = max delta_j
    floor_j    = max(0, solo_j - joint_j)          <- from the same run, same head
    excess_j   = max(0, delta_j - floor_j)         <- what sequential updating adds
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = tuple(240001 + 7 * k for k in range(15))
ORDER = ("order_short", "parity3", "sim_xor")
N_TR, N_TE = 384, 256
BUF = 128
EPOCHS = 800
TAIL = 300
SOLO = 1600
JOINT = 3 * SOLO
ARMS = ("linear", "mlp32", "mlp32_noreplay")


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
    o = e / e.sum(1, keepdims=True)
    return o[0] if one else o


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


class Lin:
    name = "linear"

    def __init__(self, d, seed=0):
        self.mu = None; self.sd = None
        self.w = np.zeros((2, d)); self.b = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 0.05)
        return (F - self.mu) / self.sd

    def step(self, A, y, lr=.5):
        p = softmax(A @ self.w.T + self.b)
        e = -p; e[np.arange(len(y)), y] += 1
        self.w += lr * (e.T @ A) / len(y)
        self.b += lr * e.mean(0)

    def acc(self, A, y):
        return float(np.mean(np.argmax(A @ self.w.T + self.b, 1) == y))


class MLP:
    name = "mlp32"

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


def mk(name, d, seed):
    return Lin(d, seed) if name == "linear" else MLP(d, seed, 32)


def sequential(seed, head_name, use_replay, Ftr, ytr, Fte, yte):
    head = mk(head_name, Ftr[ORDER[0]].shape[1], seed)
    seq_idx, seq_lab, seq_tk = [], [], []
    n = len(ORDER)
    R = np.full((n, n), np.nan)
    rng = np.random.default_rng(seed + 4242)

    for i_t, tk in enumerate(ORDER):
        A_c = head.z(Ftr[tk]); y_c = ytr[tk]
        if not use_replay:
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
        for j_t, tk2 in enumerate(ORDER):
            R[i_t, j_t] = head.acc(head.z(Fte[tk2]), yte[tk2])

    if use_replay:      # tail re-consolidation of the earliest task (round 26)
        A1 = head.z(Ftr[ORDER[0]])[seq_idx[:BUF]]
        y1 = np.array(seq_lab[:BUF], np.int32)
        Al = head.z(Ftr[ORDER[-1]]); yl = ytr[ORDER[-1]]
        h1 = max(1, len(y1) // 2); hl = max(1, len(yl) // 2)
        for _ in range(TAIL):
            i1 = rng.choice(len(y1), h1, replace=False)
            il = rng.choice(len(yl), hl, replace=False)
            head.step(np.concatenate([A1[i1], Al[il]]),
                      np.concatenate([y1[i1], yl[il]]))
        for j_t, tk2 in enumerate(ORDER):
            R[n - 1, j_t] = head.acc(head.z(Fte[tk2]), yte[tk2])

    diag = np.diag(R); final = R[n - 1]; delta = diag - final
    meas = delta[:n - 1]
    return {"R": R.tolist(),
            "meanF_loss": float(np.mean(np.maximum(meas, 0)) * 100),
            "worstF": float(np.max(delta) * 100),
            "acq": float(np.mean(diag)), "final": float(np.mean(final))}


def floor_of(head_name, seed, Ftr, ytr, Fte, yte):
    solo = {}
    for tk in ORDER:
        h = mk(head_name, Ftr[tk].shape[1], seed)
        A = h.z(Ftr[tk])
        for _ in range(SOLO):
            h.step(A, ytr[tk])
        solo[tk] = h.acc(h.z(Fte[tk]), yte[tk])
    Fj = np.concatenate([Ftr[tk] for tk in ORDER])
    yj = np.concatenate([ytr[tk] for tk in ORDER])
    hj = mk(head_name, Fj.shape[1], seed)
    Aj = hj.z(Fj)
    rng = np.random.default_rng(seed + 5)
    n = len(yj); half = max(1, n // 2)
    for _ in range(JOINT):
        idx = rng.choice(n, half, replace=False)
        hj.step(Aj[idx], yj[idx])
    joint = {tk: hj.acc(hj.z(Fte[tk]), yte[tk]) for tk in ORDER}
    per = {tk: max(0.0, (solo[tk] - joint[tk]) * 100) for tk in ORDER}
    return per, float(np.mean(list(per.values())))


def main():
    t0 = time.time()
    print("=" * 100)
    print("R7 ROUND 30 — registered test: SEQUENTIAL forgetting with a nonlinear readout")
    print("=" * 100)
    print(f"  order {' > '.join(ORDER)} | 15 fresh seeds | arms {', '.join(ARMS)}")
    print(f"  registered: mlp32 reaches meanF<=3 AND worstF<=5 on EVERY seed")
    print(f"  falsifier  : sequential forgetting exceeds its own floor by > ~3pp\n")

    cells = []
    for k, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in ORDER:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = feats(core, trx); ytr[tk] = try_
            Fte[tk] = feats(core, tex); yte[tk] = tey

        floors = {}
        for hn in ("linear", "mlp32"):
            per, mean = floor_of(hn, seed, Ftr, ytr, Fte, yte)
            floors[hn] = {"per_task": per, "mean": mean}

        rec = {"seed": seed, "floors": floors, "arms": {}}
        for arm in ARMS:
            hn = "linear" if arm == "linear" else "mlp32"
            use_rep = arm != "mlp32_noreplay"
            rec["arms"][arm] = sequential(seed, hn, use_rep, Ftr, ytr, Fte, yte)
        cells.append(rec)
        a = rec["arms"]
        print(f"  {k+1:2d}/15 {seed}  lin {a['linear']['meanF_loss']:5.2f}/"
              f"{a['linear']['worstF']:5.2f}  "
              f"mlp {a['mlp32']['meanF_loss']:5.2f}/{a['mlp32']['worstF']:5.2f}  "
              f"norep {a['mlp32_noreplay']['meanF_loss']:5.2f}/"
              f"{a['mlp32_noreplay']['worstF']:5.2f}  "
              f"floor {floors['mlp32']['mean']:4.2f}", flush=True)

    print("\n" + "=" * 100)
    print("RESULTS (loss-based; gate: meanF<=3, worstF<=5, acquisition>=0.65)")
    print("=" * 100)
    print(f"  {'arm':16s} {'meanF':>8} {'n<=3':>6} {'worstF':>8} {'n<=5':>6} "
          f"{'acq':>7} {'n>=.65':>7} {'BOTH':>6}")
    summary = {}
    for arm in ARMS:
        mf = np.array([c["arms"][arm]["meanF_loss"] for c in cells])
        wf = np.array([c["arms"][arm]["worstF"] for c in cells])
        aq = np.array([c["arms"][arm]["acq"] for c in cells])
        both = int(((mf <= 3) & (wf <= 5) & (aq >= .65)).sum())
        summary[arm] = {"meanF": float(mf.mean()), "n3": int((mf <= 3).sum()),
                        "worstF": float(wf.mean()), "n5": int((wf <= 5).sum()),
                        "acq": float(aq.mean()), "nAq": int((aq >= .65).sum()),
                        "both": both}
        s = summary[arm]
        print(f"  {arm:16s} {s['meanF']:8.2f} {s['n3']:6d} {s['worstF']:8.2f} "
              f"{s['n5']:6d} {s['acq']:7.4f} {s['nAq']:5d}/15 {both:5d}/15")

    print("\n" + "=" * 100)
    print("DOES SEQUENTIAL ADD ANYTHING ON TOP OF THE FLOOR?  (mlp32)")
    print("=" * 100)
    fl = np.array([c["floors"]["mlp32"]["mean"] for c in cells])
    tot = np.array([c["arms"]["mlp32"]["meanF_loss"] for c in cells])
    exc = tot - fl
    print(f"  floor (solo-joint)   mean {fl.mean():6.2f}pp  max {fl.max():6.2f}")
    print(f"  total measured       mean {tot.mean():6.2f}pp  max {tot.max():6.2f}")
    print(f"  excess over floor    mean {exc.mean():+6.2f}pp  max {exc.max():+6.2f}")
    print(f"\n  headroom left below the 3pp gate: {3.0 - fl.mean():.2f}pp")
    print(f"  excess consumes: {exc.mean():.2f}pp of it")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    s = summary["mlp32"]
    print(f"  registered prediction: mlp32 passes all three gates on EVERY seed")
    print(f"  measured: meanF {s['meanF']:.2f} ({s['n3']}/15), "
          f"worstF {s['worstF']:.2f} ({s['n5']}/15), "
          f"acq {s['acq']:.4f} ({s['nAq']}/15)")
    print(f"  seeds passing ALL THREE: {s['both']}/15")
    if s["both"] == 15:
        print(f"\n  => CONFIRMED. Every seed passes all three gates.")
        verdict = "registered_confirmed"
    elif s["both"] >= 10:
        print(f"\n  => MOSTLY CONFIRMED: {s['both']}/15 seeds pass all three gates.")
        verdict = "mostly_confirmed"
    elif s["n3"] == 0:
        print(f"\n  => REJECTED. No seed meets meanF<=3pp even with a nonlinear readout.")
        verdict = "rejected"
    else:
        print(f"\n  => PARTIAL: {s['both']}/15. Sequence interference still dominates.")
        verdict = "partial"

    fires = exc.mean() > 3.0
    ftxt = ("FIRES - capacity fixes co-representation but not sequential interference"
            if fires else "does not fire")
    print(f"\n  falsifier check: excess over floor = {exc.mean():.2f}pp ({ftxt})")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "nonlinear-sequential.json").write_text(json.dumps({
        "seeds": list(SEEDS), "order": list(ORDER), "arms": list(ARMS),
        "summary": summary, "verdict": verdict,
        "floor_mlp_mean": float(fl.mean()), "total_mlp_mean": float(tot.mean()),
        "excess_mlp_mean": float(exc.mean()),
        "registered": "mlp32 meanF<=3 AND worstF<=5 AND acq>=0.65 on every seed",
        "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: nonlinear-sequential.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
