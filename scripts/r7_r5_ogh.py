"""R7 round 5 — OGH: Orthogonal Gradient Head.

THE PROBLEM (round 4b, measured): readout interference costs 20.96pp of mean forgetting
while hidden plasticity contributes -1.17pp. One linear head cannot hold three task
boundaries at once: sim_xor falls 0.992 -> 0.469 as later tasks retrain the same head.

THE MECHANISM. Keep the head's update for task k in the null space of the gradients that
the previous tasks' decision functions depend on. This is a local, online-legal
projection: it uses only stored per-task gradient directions and the current gradient.
No task ID is supplied at inference, no replay of old DATA (only the stored directions),
no extra neurons, no extra parameters.

    G_k   = gradient of the current task's loss w.r.t. head params, at each minibatch
    B     = orthonormal basis of previously stored gradient directions (bounded, dim d)
    g_perp = G_k - B B^T G_k                     projection onto the null space
    W_k   += lr * g_perp                          the update
    after task k: append the task's mean gradient direction to B, re-orthonormalise

This is the standard "orthogonal weights" idea applied to the READOUT of a fixed
representation. It is NOT a continuous-learning trick that needs task boundaries at
inference: the head is a single weight matrix throughout, and B is only bookkeeping.

COST. B is at most d x d floats; d = 48 here, so 2304 floats = 9 KB. O(d^2) per update.

WHY THIS IS FALSIFIABLE. Prediction: mean forgetting drops from 20.96pp toward the 3pp
gate while acquisition stays >= 0.65. Falsifier: if acquisition collapses (the projection
removes the directions needed to learn) the mechanism fails even if forgetting is low —
that would be False Progress 3 again.

CONTROLS.
  ogh            the mechanism
  ogh_shuffled   same projection budget, but B is filled with RANDOM orthonormal
                 directions. If this matches OGH, the benefit is "fewer degrees of
                 freedom", not task-relevant orthogonalisation.
  frozen_shared  the 20.96pp baseline (no protection)
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
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

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


class OGHHead:
    """Linear head whose per-task update is projected off previous tasks' directions."""

    def __init__(self, dim, seed=0, mode="ogh", cap=8):
        self.mu = None; self.sd = None
        g = np.random.default_rng(seed + 31)
        self.w = g.normal(0, .01, (2, dim)); self.b = np.zeros(2)
        self.mode = mode
        self.cap = cap
        self.B = np.zeros((dim, 0))       # orthonormal stored directions
        self._task_mean = np.zeros(dim)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 1e-6)
        return (F - self.mu) / self.sd

    def _project(self, g):
        if self.B.shape[1] == 0 or self.mode == "plain":
            return g
        return g - self.B @ (self.B.T @ g)

    def train(self, F, y, epochs=400, lr=.5):
        A = self.z(F)
        n = len(y)
        d = A.shape[1]
        acc_grad = np.zeros(d)
        for _ in range(epochs):
            lg = A @ self.w.T + self.b
            lg -= lg.max(1, keepdims=True)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            e = -p; e[np.arange(n), y] += 1
            gw = (e.T @ A) / n
            gb = e.mean(0)
            acc_grad += gw[1] - gw[0]
            for r in range(2):
                self.w[r] += lr * self._project(gw[r])
            self.b += lr * gb
        if self.mode != "plain":
            self._task_mean = acc_grad / max(1, epochs)
            self._store(self._task_mean)

    def _store(self, v):
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            return
        v = v / nv
        if self.B.shape[1] == 0:
            self.B = v[:, None]
        else:
            new = v - self.B @ (self.B.T @ v)
            nn = np.linalg.norm(new)
            if nn < 1e-6:
                return
            self.B = np.concatenate([self.B, (new / nn)[:, None]], 1)
        if self.B.shape[1] > self.cap:
            self.B = self.B[:, -self.cap:]

    def fill_random(self, d, seed=0):
        rng = np.random.default_rng(seed + 999)
        M = rng.normal(0, 1, (d, self.cap))
        Q, _ = np.linalg.qr(M)
        self.B = Q

    def acc(self, F, y):
        return float(np.mean(np.argmax(self.z(F) @ self.w.T + self.b, 1) == y))


def run():
    root = Path(__file__).resolve().parents[1]
    run = new_run(root, "r7-r5-ogh-v1", {
        "seeds": list(DEV), "tasks": list(TASKS), "n_train": N_TR, "n_test": N_TE,
        "arms": ["frozen_shared", "ogh", "ogh_shuffled"], "gates": GATES,
        "metric": "R[i][j]; forgetting_j = R[j][j] - R[T-1][j]"})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            data = {}
            for tk in TASKS:
                trx, try_ = GENS[tk](seed + 200_000, N_TR)
                tex, tey = GENS[tk](seed + 300_000, N_TE)
                data[tk] = (trx, try_, tex, tey)

            for arm in ("frozen_shared", "ogh", "ogh_shuffled"):
                core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
                # frozen hidden everywhere: round 4b showed hidden learning does not
                # affect forgetting, so this isolates the readout mechanism
                head = OGHHead(core.config.hidden, seed,
                               mode="plain" if arm == "frozen_shared" else "ogh")
                if arm == "ogh_shuffled":
                    head.fill_random(core.config.hidden, seed)

                R = np.full((len(TASKS), len(TASKS)), np.nan)
                for i_t, tk in enumerate(TASKS):
                    trx, try_, tex, tey = data[tk]
                    for i in range(N_TR):
                        core.reset_state()
                        for t in range(12):
                            core.step(trx[i, t], track=True)
                        core.learn(int(try_[i]), mode="frozen")
                    F = readout_feats(core, trx)
                    head.train(F, try_)
                    for j_t, tk2 in enumerate(TASKS):
                        _, _, tex2, tey2 = data[tk2]
                        R[i_t, j_t] = head.acc(readout_feats(core, tex2), tey2)
                rows.append({"seed": seed, "arm": arm, "R": R.tolist(),
                             "basis_dim": int(head.B.shape[1])})
                mon.check()
                print(f"  seed{seed} {arm:15s} R="
                      f"{np.round(R, 3).tolist()}", flush=True)

    def summarize(arm):
        Rs = [np.array(r["R"]) for r in rows if r["arm"] == arm]
        n = len(TASKS)
        acq = float(np.mean([np.mean(np.diag(R)) for R in Rs]))
        final = float(np.mean([np.mean(R[n - 1]) for R in Rs]))
        fg = [np.diag(R) - R[n - 1] for R in Rs]
        meanF = float(np.mean([np.mean(f) for f in fg])) * 100
        worstF = float(np.max([np.max(f) for f in fg])) * 100
        return {"acq": acq, "final": final, "meanF": meanF, "worstF": worstF,
                "passes": bool(acq >= GATES["acq"] and final >= GATES["final"]
                               and meanF <= GATES["mean_f_pp"]
                               and worstF <= GATES["worst_f_pp"])}

    summary = {a: summarize(a) for a in ("frozen_shared", "ogh", "ogh_shuffled")}
    write_json(run / "result.json",
               {"trials": rows, "summary": summary,
                "resources": dict(mon.report()),
                "wall_seconds": time.time() - t0})

    print("\n" + "=" * 88)
    print("R7 ROUND 5 — orthogonal gradient head vs readout interference")
    print("=" * 88)
    print(f"  {'arm':16s} {'acq':>7} {'final':>7} {'meanF':>8} {'worstF':>8} {'gate':>6}")
    for a, s in summary.items():
        print(f"  {a:16s} {s['acq']:7.4f} {s['final']:7.4f} "
              f"{s['meanF']:7.2f}p {s['worstF']:7.2f}p "
              f"{('PASS' if s['passes'] else 'fail'):>6}")
    print(f"\n  gates: acq>=.65 final>=.65 meanF<=3pp worstF<=5pp")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
