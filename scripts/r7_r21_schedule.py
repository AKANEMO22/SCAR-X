"""R7 round 21 — the exchange rate is set by HOW replay is scheduled, not by how much.

ROUND 20 LEFT THIS OPEN
-----------------------
    mean forgetting   -19.66pp  CI [-24.40, -14.92]  18/18 cells
    acquisition        -2.60pp  CI [ -4.81,  -0.38]   7/18 cells

Replay buys 19.66pp of forgetting for 2.60pp of acquisition. The mission forbids trading
acquisition for low forgetting, so the ratio is the thing to fix.

WHAT THE ROUND-20 CODE ACTUALLY DOES
    head.train(current_task_features, labels)        # 800 epochs
    if replay:
        head.train(replay_features, replay_labels)   # then 800 MORE epochs

So the head's FINAL gradient state is dominated by replayed old tasks. That is a plausible
and specific cause of the acquisition cost: not that old data is present, but that it is
trained LAST at equal weight.

TWO COMPETING EXPLANATIONS
--------------------------
H-schedule  The cost is a scheduling artefact. Mixing the two sources in one loop, so every
            epoch sees both, should cut the cost while preserving retention. Predicts:
            interleaved has much smaller acquisition delta than sequential, with most of
            the forgetting gain intact.

H-intrinsic The cost is intrinsic: spending gradient budget on old data necessarily costs
            current-task fit. Predicts: interleaved costs about the same as sequential,
            because the total exposure to old data is identical.

DESIGN
  18 cells (3 seeds x 6 orders), three arms, IDENTICAL total gradient budget (1600 epochs):
      no_replay    current only, 1600 epochs
      sequential   800 current, then 800 replay          (round 20)
      interleaved  1600 epochs on a mixed batch, each epoch drawing half current / half
                   replay, shuffling both

  The budget match is what makes this a scheduling comparison rather than a capacity one.

FALSIFIER: if interleaved acquisition delta is not at least 1.5pp better than sequential
while retaining >= 70% of the forgetting gain, H-intrinsic holds and the exchange rate is
not fixable by scheduling.
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
BUF = 128
EPOCHS = 800
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

    def _step(self, A, y, lr):
        p = softmax(A @ self.w.T + self.b)
        e = -p; e[np.arange(len(y)), y] += 1
        self.w += lr * (e.T @ A) / len(y)
        self.b += lr * e.mean(0)

    def train(self, F, y, epochs=EPOCHS, lr=.5):
        A = self.z(F)
        for _ in range(epochs):
            self._step(A, y, lr)

    def train_mixed(self, Fc, yc, Fr, yr, epochs=2 * EPOCHS, lr=.5, rng=None):
        """Every epoch sees BOTH sources, half each. Same total gradient steps."""
        if rng is None:
            rng = np.random.default_rng(0)
        Ac = self.z(Fc)
        Ar = (Fr - self.mu) / self.sd
        nc, nr = len(yc), len(yr)
        half_r = min(nr, max(1, nr // 2))
        half_c = min(nc, max(1, nc // 2))
        for _ in range(epochs):
            ic = rng.choice(nc, half_c, replace=False)
            ir = rng.choice(nr, half_r, replace=False)
            A = np.concatenate([Ac[ic], Ar[ir]])
            y = np.concatenate([yc[ic], yr[ir]])
            self._step(A, y, lr)

    def acc(self, F, y):
        return float(np.mean(np.argmax(self.z(F) @ self.w.T + self.b, 1) == y))


def run(seed, order, data, arm):
    core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
    head = Head(S.cfg(seed).hidden)
    buf_seq, buf_lab = [], []
    n = len(order)
    R = np.full((n, n), np.nan)
    rng = np.random.default_rng(seed + 999)

    for i_t, tk in enumerate(order):
        trx, try_, tex, tey = data[tk]
        F_c = feats(core, trx)

        if arm == "no_replay":
            head.train(F_c, try_, epochs=2 * EPOCHS)
        elif arm == "sequential":
            head.train(F_c, try_, epochs=EPOCHS)
        # retention of this task's examples, used by both replay arms
        if arm in ("sequential", "interleaved"):
            idx = np.random.default_rng(seed + i_t).choice(
                N_TR, min(BUF, N_TR), replace=False)
            buf_seq.extend(trx[j].copy() for j in idx)
            buf_lab.extend(int(try_[j]) for j in idx)

        if arm == "sequential":
            F_r = feats(core, np.stack(buf_seq))
            head.train(F_r, np.array(buf_lab, np.int32), epochs=EPOCHS)
        elif arm == "interleaved":
            F_r = feats(core, np.stack(buf_seq))
            head.train_mixed(F_c, try_, F_r, np.array(buf_lab, np.int32),
                             epochs=2 * EPOCHS, rng=rng)

        for j_t, tk2 in enumerate(order):
            _, _, tex2, tey2 = data[tk2]
            R[i_t, j_t] = head.acc(feats(core, tex2), tey2)

    diag = np.diag(R); final = R[n - 1]; fg = diag - final
    return {"acq": float(np.mean(diag)), "final": float(np.mean(final)),
            "meanF_pp": float(np.mean(fg[:n - 1]) * 100),
            "worstF_pp": float(np.max(fg) * 100)}


def t_ci(x, tcrit=2.110):
    x = np.asarray(x, float)
    m = float(x.mean()); s = float(x.std(ddof=1)); se = s / np.sqrt(len(x))
    return m, s, (m - tcrit * se, m + tcrit * se)


def main():
    print("=" * 100)
    print("R7 ROUND 21 — is the acquisition cost a SCHEDULING artefact?")
    print("=" * 100)
    print("  matched total budget: no_replay 1600 ep | sequential 800+800 | "
          "interleaved 1600 mixed\n")

    cells = []
    for seed in DEV:
        data = {}
        for tk in BASE:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            data[tk] = (trx, try_, tex, tey)
        for order in ORDERS:
            r = {arm: run(seed, order, data, arm)
                 for arm in ("no_replay", "sequential", "interleaved")}
            cells.append({"seed": seed, "order": list(order), **r})
            print(f"  seed{seed} {'>'.join(o[:4] for o in order):22s} "
                  f"no_rep F{r['no_replay']['meanF_pp']:6.2f} "
                  f"acq{r['no_replay']['acq']:.3f} | "
                  f"seq F{r['sequential']['meanF_pp']:6.2f} "
                  f"acq{r['sequential']['acq']:.3f} | "
                  f"int F{r['interleaved']['meanF_pp']:6.2f} "
                  f"acq{r['interleaved']['acq']:.3f}", flush=True)

    print("\n" + "=" * 100)
    print("PAIRED DELTAS vs no_replay (18 cells)")
    print("=" * 100)
    res = {}
    for arm in ("sequential", "interleaved"):
        for key, label in (("meanF_pp", "forgetting"), ("acq", "acquisition"),
                           ("final", "final")):
            v = [c[arm][key] - c["no_replay"][key] for c in cells]
            m, s, ci = t_ci(v)
            n_better = sum(1 for x in v
                           if (x < 0 if key == "meanF_pp" else x > 0))
            res[f"{arm}|{key}"] = {"mean": m, "sd": s, "ci95": list(ci),
                                   "improved": n_better}
            unit = "pp" if key != "acq" else ""
            print(f"  {arm:12s} {label:12s} {m:+8.4f}{unit}  "
                  f"CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  {n_better}/18 better")

    print("\n" + "=" * 100)
    print("THE EXCHANGE RATE")
    print("=" * 100)
    sF = res["sequential|meanF_pp"]["mean"]
    sA = res["sequential|acq"]["mean"]
    iF = res["interleaved|meanF_pp"]["mean"]
    iA = res["interleaved|acq"]["mean"]
    print(f"  sequential   forgetting {sF:+7.2f}pp   acquisition {sA:+.4f}")
    print(f"  interleaved  forgetting {iF:+7.2f}pp   acquisition {iA:+.4f}")
    print(f"\n  acquisition recovered : {(iA - sA)*100:+.2f}pp")
    gain_retained = (iF / sF * 100) if sF != 0 else float("nan")
    print(f"  forgetting gain kept  : {gain_retained:.1f}% of sequential's")
    print(f"  ratio seq  : {abs(sF)/abs(sA) if sA else float('inf'):.1f}pp forget per 1pp acq")
    print(f"  ratio int  : {abs(iF)/abs(iA) if iA else float('inf'):.1f}pp forget per 1pp acq")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    acq_recovered = (iA - sA)
    if acq_recovered >= 0.015 and gain_retained >= 70:
        print(f"  => H-SCHEDULE. Interleaving recovers {acq_recovered*100:.2f}pp of")
        print(f"     acquisition while keeping {gain_retained:.0f}% of the forgetting gain.")
        print(f"     The exchange rate is a scheduling property, not intrinsic.")
        verdict = "H_schedule"
    elif acq_recovered >= 0.015:
        print(f"  => Interleaving recovers acquisition but only keeps "
              f"{gain_retained:.0f}% of the gain. Partial.")
        verdict = "partial"
    else:
        print(f"  => H-INTRINSIC. Interleaving recovers only "
              f"{acq_recovered*100:.2f}pp; the cost comes from spending budget on old")
        print(f"     data, not from when it is spent.")
        verdict = "H_intrinsic"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay-schedule.json").write_text(
        json.dumps({"cells": cells, "paired": res,
                    "exchange_rate": {"sequential": [sF, sA],
                                      "interleaved": [iF, iA],
                                      "acq_recovered_pp": acq_recovered * 100,
                                      "gain_retained_pct": gain_retained},
                    "verdict": verdict, "budget_epochs": 2 * EPOCHS},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/replay-schedule.json")


if __name__ == "__main__":
    main()
