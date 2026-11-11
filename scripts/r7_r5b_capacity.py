"""R7 round 5b — is READOUT CAPACITY really the binding constraint? (diagnostic only)

Round 5 concluded that the 3pp forgetting gate is unreachable because a 2-output LINEAR
head has 98 parameters for three overlapping 48-dim task distributions, and
orthogonalisation preserves directions but cannot create capacity.

That was an INTERPRETATION, not a measurement. This tests it directly.

DIAGNOSTIC, NOT A CANDIDATE. Brief section II forbids expanding capacity to mask a
learning-rule failure. A wider or nonlinear head used as a CANDIDATE would be exactly
that. Used as a *learnability diagnostic* — the same role BPTT plays — it answers one
question: is capacity, rather than the local rule, what bounds retention?

  If a nonlinear head retains all three tasks within the gates, capacity IS the
  constraint and the mission's second gate requires a scope decision.
  If it does not, capacity is NOT the constraint and my round-5 interpretation is wrong.

Heads compared (identical features, identical data, identical R-matrix protocol):
  linear_2logit      the current configuration (98 params)
  linear_wide_32     a wider LINEAR head (2x32 outputs averaged, 3136 params)
  nonlinear_mlp      one hidden layer, 32 units, tanh (diagnostic ceiling proxy)

All heads are trained offline on frozen features. None is a candidate. None touches
src/scarx/neural/. No production weight is derived from any of them.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
TASKS = ("sim_xor", "parity3", "order_short")
GATES = {"acq": .65, "final": .65, "mean_f_pp": 3.0, "worst_f_pp": 5.0}


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


def readout_feats(core, x):
    out = []
    for i in range(len(x)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(x[i, t], track=False)
            tot += core.spikes
        out.append(tot)
    return np.stack(out)


class Lin:
    def __init__(self, d, seed=0):
        self.mu = None; self.sd = None
        g = np.random.default_rng(seed + 31)
        self.w = g.normal(0, .01, (2, d)); self.b = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 1e-6)
        return (F - self.mu) / self.sd

    def train(self, F, y, epochs=400, lr=.5):
        A = self.z(F); n = len(y)
        for _ in range(epochs):
            lg = A @ self.w.T + self.b; lg -= lg.max(1, keepdims=True)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            e = -p; e[np.arange(n), y] += 1
            self.w += lr * (e.T @ A) / n; self.b += lr * e.mean(0)

    def acc(self, F, y):
        return float(np.mean(np.argmax(self.z(F) @ self.w.T + self.b, 1) == y))


class Wide:
    """Wider LINEAR head: W is (2K, d), readout averages the K blocks per class."""

    def __init__(self, d, K=32, seed=0):
        self.mu = None; self.sd = None; self.K = K
        g = np.random.default_rng(seed + 37)
        self.w = g.normal(0, .01, (2 * K, d)); self.b = np.zeros(2 * K)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 1e-6)
        return (F - self.mu) / self.sd

    def logits(self, A):
        lg = A @ self.w.T + self.b
        return lg.reshape(len(A), 2, self.K).mean(2)

    def train(self, F, y, epochs=400, lr=.5):
        A = self.z(F); n = len(y)
        tgt = np.zeros((n, 2 * self.K))
        for k in range(self.K):
            tgt[np.arange(n), y * self.K + k] = 1.0
        for _ in range(epochs):
            lg = A @ self.w.T + self.b
            lg -= lg.max(1, keepdims=True)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            e = tgt - p
            self.w += lr * (e.T @ A) / n; self.b += lr * e.mean(0)

    def acc(self, F, y):
        return float(np.mean(np.argmax(self.logits(self.z(F)), 1) == y))


class MLP:
    """One tanh hidden layer. DIAGNOSTIC ceiling proxy, never a candidate."""

    def __init__(self, d, H=32, seed=0):
        self.mu = None; self.sd = None
        g = np.random.default_rng(seed + 41)
        self.W1 = g.normal(0, 1 / np.sqrt(d), (H, d))
        self.b1 = np.zeros(H)
        self.W2 = g.normal(0, 1 / np.sqrt(H), (2, H))
        self.b2 = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 1e-6)
        return (F - self.mu) / self.sd

    def fwd(self, A):
        Hd = np.tanh(A @ self.W1.T + self.b1)
        return Hd, Hd @ self.W2.T + self.b2

    def train(self, F, y, epochs=400, lr=.5):
        A = self.z(F); n = len(y)
        for _ in range(epochs):
            Hd, lg = self.fwd(A)
            lg = lg - lg.max(1, keepdims=True)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            e = -p; e[np.arange(n), y] += 1
            gW2 = (e.T @ Hd) / n; gb2 = e.mean(0)
            eH = (e @ self.W2) * (1 - Hd ** 2)
            gW1 = (eH.T @ A) / n; gb1 = eH.mean(0)
            self.W1 += lr * gW1; self.b1 += lr * gb1
            self.W2 += lr * gW2; self.b2 += lr * gb2

    def acc(self, F, y):
        _, lg = self.fwd(self.z(F))
        return float(np.mean(np.argmax(lg, 1) == y))


def run():
    root = Path(__file__).resolve().parents[1]
    run = new_run(root, "r7-r5b-capacity-diagnostic-v1", {
        "seeds": list(DEV), "tasks": list(TASKS),
        "role": "DIAGNOSTIC ONLY - no head is a candidate",
        "heads": ["linear_2logit", "linear_wide_32", "mlp32_tanh"],
        "gates": GATES})

    rows = []
    t0 = time.time()
    for seed in DEV:
        data = {}
        for tk in TASKS:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            data[tk] = (trx, try_, tex, tey)

        # frozen features per task, computed ONCE (hidden frozen, so reusable)
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        F_tr, F_te = {}, {}
        for tk in TASKS:
            trx, try_, tex, tey = data[tk]
            F_tr[tk] = readout_feats(core, trx)
            F_te[tk] = readout_feats(core, tex)

        for hname in ("linear_2logit", "linear_wide_32", "mlp32_tanh"):
            if hname == "linear_2logit":
                mk = lambda: Lin(core.config.hidden, seed)      # noqa: E731
            elif hname == "linear_wide_32":
                mk = lambda: Wide(core.config.hidden, 32, seed)  # noqa: E731
            else:
                mk = lambda: MLP(core.config.hidden, 32, seed)   # noqa: E731
            head = mk()
            R = np.full((len(TASKS), len(TASKS)), np.nan)
            for i_t, tk in enumerate(TASKS):
                head.train(F_tr[tk], data[tk][1])
                for j_t, tk2 in enumerate(TASKS):
                    R[i_t, j_t] = head.acc(F_te[tk2], data[tk2][3])
            rows.append({"seed": seed, "head": hname, "R": R.tolist()})
            print(f"  seed{seed} {hname:15s} R={np.round(R,3).tolist()}", flush=True)

    def summarize(h):
        Rs = [np.array(r["R"]) for r in rows if r["head"] == h]
        n = len(TASKS)
        acq = float(np.mean([np.mean(np.diag(R)) for R in Rs]))
        per = {tk: float(np.mean([R[n - 1, j] for R in Rs])) for j, tk in enumerate(TASKS)}
        final = float(np.mean(list(per.values())))
        fg = [np.diag(R) - R[n - 1] for R in Rs]
        meanF = float(np.mean([np.mean(f) for f in fg])) * 100
        worstF = float(np.max([np.max(f) for f in fg])) * 100
        return {"acq": acq, "final": final, "meanF": meanF, "worstF": worstF,
                "per_task": per,
                "passes": bool(acq >= GATES["acq"] and final >= GATES["final"]
                               and meanF <= GATES["mean_f_pp"]
                               and worstF <= GATES["worst_f_pp"])}

    summary = {h: summarize(h) for h in
               ("linear_2logit", "linear_wide_32", "mlp32_tanh")}
    write_json(run / "result.json",
               {"trials": rows, "summary": summary,
                "wall_seconds": time.time() - t0})

    print("\n" + "=" * 86)
    print("R7 ROUND 5b — is READOUT CAPACITY the constraint? (diagnostic)")
    print("=" * 86)
    print(f"  {'head':16s} {'acq':>7} {'final':>7} {'meanF':>8} {'worstF':>8} {'gate':>6}")
    for h, s in summary.items():
        print(f"  {h:16s} {s['acq']:7.4f} {s['final']:7.4f} "
              f"{s['meanF']:7.2f}p {s['worstF']:7.2f}p "
              f"{('PASS' if s['passes'] else 'fail'):>6}")
    print(f"\n  NONE of these heads is a candidate. Diagnostic only.")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
