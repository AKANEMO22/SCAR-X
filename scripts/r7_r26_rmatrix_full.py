"""R7 round 26 — full R matrices, both aggregations, and the task the metric ignores.

ROUND 25 FOUND A DEFECT IN MY OWN REPORTING
    meanF = mean over j of (R[j][j] - R[T-1][j])
is a mean of SIGNED deltas. Seed 240022 reported meanF 0.20pp while worstF was 12.89pp,
because one task gained 12.49pp and paid for the other's 12.89pp loss.

Round 26 fixes the measurement properly:
  1  store the FULL R matrix per cell (round 24's artifact kept only aggregates)
  2  report BOTH aggregations:
        meanF_signed   mean of signed deltas  (what rounds 20/22/24 used)
        meanF_loss     mean of max(delta, 0)  (what a loss-based gate means)
  3  report the LAST task separately. It is excluded from every forgetting average
     because its delta is zero by construction, but its FINAL accuracy is measurable
     and the mission gates final accuracy too.

Also runs a mechanism the decomposition points at and I have not tried: a RE-CONSOLIDATION
pass on the earliest task at the end. Round 25's algebra showed worstF tracks imbalance, and
the earliest task accumulates the most subsequent updates, so a short replay of task 1 after
the last task should raise the low term and cut the gap directly.

  arms
    no_replay        baseline
    interleaved      round 24's best (buf 128, 1600 epochs mixed)
    interleaved_tail interleaved + 300 extra epochs of HALF task-1 replay / half last task

  The tail arm uses the same buffer and the same data; only the final 300 epochs differ.

REGISTERED PREDICTION: the tail pass raises the minimum per-task delta (lowers worstF) while
costing at most 1pp of final accuracy, because it re-exposes the head to the earliest task at
the point where its representation has drifted furthest.

FALSIFIER: if worstF does not fall by >= 2pp versus interleaved on a paired comparison, the
mechanism is rejected and imbalance is not addressable from the readout side.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = tuple(240001 + 7 * k for k in range(15))     # same fresh set as round 24
ORDER = ("order_short", "parity3", "sim_xor")
N_TR, N_TE = 384, 256
BUF = 128
EPOCHS = 800
TAIL_EPOCHS = 300
ARMS = ("no_replay", "interleaved", "interleaved_tail")


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


def run(seed, Ftr, ytr, Fte, yte, arm, task1_idx=None, task1_lab=None):
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

    if arm == "interleaved_tail":
        # re-consolidate the EARLIEST task against the LAST one
        A1 = head.z(Ftr[ORDER[0]])[task1_idx]
        y1 = task1_lab
        Al = head.z(Ftr[ORDER[-1]])
        yl = ytr[ORDER[-1]]
        h1 = max(1, len(y1) // 2)
        hl = max(1, len(yl) // 2)
        for _ in range(TAIL_EPOCHS):
            i1 = rng.choice(len(y1), h1, replace=False)
            il = rng.choice(len(yl), hl, replace=False)
            head._step(np.concatenate([A1[i1], Al[il]]),
                       np.concatenate([y1[i1], yl[il]]), .5)
        for j_t, tk2 in enumerate(ORDER):
            R[n - 1, j_t] = head.acc(head.z(Fte[tk2]), yte[tk2])

    diag = np.diag(R); final = R[n - 1]
    delta = diag - final
    meas = delta[:n - 1]
    return {"R": R.tolist(), "delta": delta.tolist(),
            "meanF_signed": float(np.mean(meas) * 100),
            "meanF_loss": float(np.mean(np.maximum(meas, 0)) * 100),
            "worstF": float(np.max(delta) * 100),
            "acq": float(np.mean(diag)), "final": float(np.mean(final)),
            "last_task_final": float(final[-1])}


def main():
    t0 = time.time()
    print("=" * 100)
    print("R7 ROUND 26 — full R matrices, both aggregations, tail re-consolidation")
    print("=" * 100)
    print(f"  order {' > '.join(ORDER)} | 15 fresh seeds | arms {', '.join(ARMS)}\n")

    cells = []
    for k, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in ORDER:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = feats(core, trx); ytr[tk] = try_
            Fte[tk] = feats(core, tex); yte[tk] = tey
        t1idx = np.random.default_rng(seed + 1).choice(N_TR, BUF, replace=False)
        t1lab = ytr[ORDER[0]][t1idx]

        rec = {"seed": seed}
        for arm in ARMS:
            rec[arm] = run(seed, Ftr, ytr, Fte, yte, arm, t1idx, t1lab)
        cells.append(rec)
        s, i, t = rec["no_replay"], rec["interleaved"], rec["interleaved_tail"]
        print(f"  {k+1:2d}/15 {seed}  no_rep {s['meanF_signed']:7.2f} | "
              f"inter {i['meanF_signed']:7.2f} w{i['worstF']:6.2f} | "
              f"tail {t['meanF_signed']:7.2f} w{t['worstF']:6.2f}", flush=True)

    print("\n" + "=" * 100)
    print("BOTH AGGREGATIONS (interleaved)")
    print("=" * 100)
    ms = np.array([c["interleaved"]["meanF_signed"] for c in cells])
    ml = np.array([c["interleaved"]["meanF_loss"] for c in cells])
    wf = np.array([c["interleaved"]["worstF"] for c in cells])
    print(f"  meanF_signed  mean {ms.mean():6.2f}  "
          f"seeds<=3pp {int((ms <= 3).sum())}/15")
    print(f"  meanF_loss    mean {ml.mean():6.2f}  "
          f"seeds<=3pp {int((ml <= 3).sum())}/15")
    print(f"  meanF loss-signed gap: {(ml - ms).mean():+.2f}pp  "
          f"(this is how much a gain was paying for a loss)")
    print(f"  worstF        mean {wf.mean():6.2f}  "
          f"seeds<=5pp {int((wf <= 5).sum())}/15")

    print("\n" + "=" * 100)
    print("PAIRED: interleaved_tail vs interleaved  (the registered test)")
    print("=" * 100)
    for key, label, want in (("worstF", "worstF (pp)", "lower"),
                             ("meanF_loss", "meanF_loss (pp)", "lower"),
                             ("final", "final accuracy", "higher")):
        v = np.array([c["interleaved_tail"][key] - c["interleaved"][key]
                      for c in cells])
        se = v.std(ddof=1) / np.sqrt(len(v))
        lo, hi = v.mean() - 2.145 * se, v.mean() + 2.145 * se
        better = int((v < 0).sum()) if want == "lower" else int((v > 0).sum())
        unit = "pp" if key != "final" else ""
        print(f"  {label:16s} {v.mean():+7.3f}{unit}  CI [{lo:+.3f}, {hi:+.3f}]  "
              f"{better}/15 better")

    print("\n" + "=" * 100)
    print("GATE STATUS, BOTH AGGREGATIONS, ALL ARMS")
    print("=" * 100)
    print(f"  {'arm':18s} {'meanF_sig':>10} {'n<=3':>6} {'meanF_loss':>11} {'n<=3':>6} "
          f"{'worstF':>8} {'n<=5':>6}")
    summary = {}
    for arm in ARMS:
        a = np.array([c[arm]["meanF_signed"] for c in cells])
        b = np.array([c[arm]["meanF_loss"] for c in cells])
        c_ = np.array([c[arm]["worstF"] for c in cells])
        summary[arm] = {"meanF_signed": float(a.mean()),
                        "n_signed_3": int((a <= 3).sum()),
                        "meanF_loss": float(b.mean()),
                        "n_loss_3": int((b <= 3).sum()),
                        "worstF": float(c_.mean()),
                        "n_worst_5": int((c_ <= 5).sum())}
        s = summary[arm]
        print(f"  {arm:18s} {s['meanF_signed']:10.2f} {s['n_signed_3']:6d} "
              f"{s['meanF_loss']:11.2f} {s['n_loss_3']:6d} "
              f"{s['worstF']:8.2f} {s['n_worst_5']:6d}")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    d_worst = np.array([c["interleaved_tail"]["worstF"]
                        - c["interleaved"]["worstF"] for c in cells])
    se = d_worst.std(ddof=1) / np.sqrt(len(d_worst))
    lo, hi = d_worst.mean() - 2.145 * se, d_worst.mean() + 2.145 * se
    print(f"  registered: tail pass lowers worstF by >= 2pp")
    print(f"  measured  : {d_worst.mean():+.2f}pp  CI [{lo:+.2f}, {hi:+.2f}]")
    if hi < -2.0:
        print("  => CONFIRMED. Re-consolidating the earliest task at the end cuts")
        print("     worst-case forgetting, as the imbalance algebra predicted.")
        verdict = "tail_confirmed"
    elif hi < 0:
        print("  => Direction right but below the registered 2pp threshold.")
        verdict = "tail_weak"
    else:
        print("  => REJECTED. worst-case forgetting is not addressable by a readout-side")
        print("     tail pass on this suite.")
        verdict = "tail_rejected"
    print(f"\n  wall {time.time()-t0:.0f}s")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rmatrix-both-aggregations.json").write_text(json.dumps({
        "order": list(ORDER), "seeds": list(SEEDS), "cells": cells,
        "summary": summary, "verdict": verdict,
        "tail_paired_worstF": {"mean": float(d_worst.mean()),
                               "ci95": [float(lo), float(hi)]},
        "registered_prediction": "tail lowers worstF >= 2pp",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  written: rmatrix-both-aggregations.json")


if __name__ == "__main__":
    main()
