"""R7 round 7 — CPS: decision-subspace protection, targeted by the joint control.

WHAT ROUND 6 CHANGED
--------------------
Round 5's OGH protected each task's MEAN gradient direction, capped at 8 of 48 dims. It
reduced forgetting by 3.43pp against its control and still failed at 17.01pp.

Round 6 established the joint control: a linear head trained on ALL three tasks at once
reaches 0.8620 (worst task 0.7734). So the head CAN hold the union. The failure is
sequential interference, and OGH's target — one mean direction per task — was too coarse.

CPS (decision-subspace protection): instead of one mean gradient, store the top-r
eigenvectors of each task's **gradient covariance**. That is the subspace the task's
decision depends on, not its average direction.

    For each minibatch, g_t = grad of the loss w.r.t. head params (48-dim after the
    class difference). Accumulate C = sum_t g_t g_t^T (running, O(d^2)).
    B_k = top-r eigenvectors of C_k, orthonormalised against previous tasks' B.
    Update:  W += lr * (g - B Bᵀ g)

Cost: C is 48x48 = 2304 floats per task (9 KB), plus the eigen-decomposition each task
(negligible at d=48). r = 4 per task, 3 tasks = 12 of 48 dims protected.

CONTROLS (mandatory, same budget)
    cps            top-r eigen-subspace protection
    cps_shuffled   same number of protected dims, random orthonormal basis
    ogh_mean       round-5 mechanism (one mean direction per task) for comparability
    frozen_shared  no protection

FALSIFIER. If CPS does not beat cps_shuffled on the honest mean forgetting while
preserving acquisition, it is rejected. Round 6 also established the target: honest mean
<= 3pp with worst <= 5pp AND acquisition >= 0.65. The joint ceiling is 0.8620, so an
acquisition collapse toward 0.48 (the replay failure mode) is not recoverable credit.
"""
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
R_PER_TASK = 4


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


class ProtHead:
    """Linear head whose updates are projected off a protected subspace."""

    def __init__(self, d, seed=0, mode="cps", r=R_PER_TASK, batch=32):
        self.mu = None; self.sd = None
        g = np.random.default_rng(seed + 31)
        self.w = g.normal(0, .01, (2, d)); self.b = np.zeros(2)
        self.mode, self.r, self.batch, self.d = mode, r, batch, d
        self.B = np.zeros((d, 0))
        self._cov = np.zeros((d, d))
        self._mean_dir = np.zeros(d)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 1e-6)
        return (F - self.mu) / self.sd

    def _proj(self, g):
        if self.B.shape[1] == 0 or self.mode == "none":
            return g
        return g - self.B @ (self.B.T @ g)

    def train(self, F, y, epochs=400, lr=.5):
        A = self.z(F); n = len(y)
        self._cov[:] = 0.0
        acc_mean = np.zeros(self.d)
        for _ in range(epochs):
            perm = np.random.default_rng(0).permutation(n)
            for st in range(0, n, self.batch):
                idx = perm[st:st + self.batch]
                Ab, yb = A[idx], y[idx]
                lg = Ab @ self.w.T + self.b
                lg -= lg.max(1, keepdims=True)
                p = np.exp(lg); p /= p.sum(1, keepdims=True)
                e = -p; e[np.arange(len(yb)), yb] += 1
                gw = (e.T @ Ab) / len(yb)
                gb = e.mean(0)
                gdiff = gw[1] - gw[0]                 # class-discriminating direction
                self._cov += np.outer(gdiff, gdiff)
                acc_mean += gdiff
                for rr in range(2):
                    self.w[rr] += lr * self._proj(gw[rr])
                self.b += lr * gb
        self._mean_dir = acc_mean / max(1, epochs)

    def commit_task(self, seed):
        if self.mode == "none":
            return
        if self.mode == "cps":
            if np.abs(self._cov).sum() < 1e-12:
                return
            vals, vecs = np.linalg.eigh(self._cov)
            order = np.argsort(vals)[::-1][:self.r]
            new = vecs[:, order]
        elif self.mode == "mean":
            v = self._mean_dir
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                return
            new = (v / nv)[:, None]
        elif self.mode == "shuffled":
            rng = np.random.default_rng(seed + 4242)
            M = rng.normal(0, 1, (self.d, self.r))
            new, _ = np.linalg.qr(M)
        else:
            raise ValueError(self.mode)
        if self.B.shape[1]:
            new = new - self.B @ (self.B.T @ new)
        Q, _ = np.linalg.qr(new)
        keep = [i for i in range(Q.shape[1]) if np.linalg.norm(Q[:, i]) > 1e-6]
        self.B = np.concatenate([self.B, Q[:, keep]], 1) if keep else self.B

    def acc(self, F, y):
        return float(np.mean(np.argmax(self.z(F) @ self.w.T + self.b, 1) == y))


def run():
    root = Path(__file__).resolve().parents[1]
    run = new_run(root, "r7-r7-cps-v1", {
        "seeds": list(DEV), "tasks": list(TASKS), "r_per_task": R_PER_TASK,
        "arms": ["frozen_shared", "ogh_mean", "cps", "cps_shuffled"],
        "metric": "honest meanF over T-1 measurable tasks",
        "joint_ceiling": 0.8620, "gates": GATES})

    rows = []
    t0 = time.time()
    for seed in DEV:
        data = {}
        for tk in TASKS:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            data[tk] = (trx, try_, tex, tey)
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        F_tr = {tk: readout_feats(core, data[tk][0]) for tk in TASKS}
        F_te = {tk: readout_feats(core, data[tk][2]) for tk in TASKS}

        for arm, mode in (("frozen_shared", "none"), ("ogh_mean", "mean"),
                          ("cps", "cps"), ("cps_shuffled", "shuffled")):
            head = ProtHead(core.config.hidden, seed, mode=mode)
            R = np.full((len(TASKS), len(TASKS)), np.nan)
            for i_t, tk in enumerate(TASKS):
                head.train(F_tr[tk], data[tk][1])
                head.commit_task(seed + i_t)
                for j_t, tk2 in enumerate(TASKS):
                    R[i_t, j_t] = head.acc(F_te[tk2], data[tk2][3])
            rows.append({"seed": seed, "arm": arm, "R": R.tolist(),
                         "basis_dim": int(head.B.shape[1])})
            print(f"  seed{seed} {arm:14s} R={np.round(R,3).tolist()}", flush=True)

    def summ(arm):
        Rs = [np.array(r["R"]) for r in rows if r["arm"] == arm]
        n = len(TASKS)
        acq = float(np.mean([np.mean(np.diag(R)) for R in Rs]))
        per = {tk: float(np.mean([R[n-1, j] for R in Rs])) for j, tk in enumerate(TASKS)}
        final = float(np.mean(list(per.values())))
        fg = [np.diag(R) - R[n-1] for R in Rs]
        meanF_hon = float(np.mean([np.mean(f[:n-1]) for f in fg])) * 100
        worstF = float(np.max([np.max(f) for f in fg])) * 100
        per_seed = [float(np.mean(f[:n-1])) * 100 for f in fg]
        return {"acq": acq, "final": final, "meanF_honest": meanF_hon,
                "worstF": worstF, "per_seed_honest": per_seed, "per_task": per,
                "passes": bool(acq >= GATES["acq"] and final >= GATES["final"]
                               and meanF_hon <= GATES["mean_f_pp"]
                               and worstF <= GATES["worst_f_pp"])}

    summary = {a: summ(a) for a in ("frozen_shared", "ogh_mean", "cps", "cps_shuffled")}
    write_json(run / "result.json",
               {"trials": rows, "summary": summary,
                "wall_seconds": time.time() - t0})

    print("\n" + "=" * 88)
    print("R7 ROUND 7 — decision-subspace protection (honest metric)")
    print("=" * 88)
    print(f"  {'arm':15s} {'acq':>7} {'final':>7} {'meanF_h':>9} {'worstF':>8} "
          f"{'gate':>6}   per-seed meanF_h")
    for a, s in summary.items():
        ps = " ".join(f"{v:.2f}" for v in s["per_seed_honest"])
        print(f"  {a:15s} {s['acq']:7.4f} {s['final']:7.4f} {s['meanF_honest']:8.2f}p "
              f"{s['worstF']:7.2f}p {('PASS' if s['passes'] else 'fail'):>6}   {ps}")
    print(f"\n  joint linear ceiling: 0.8620 (round 6)")
    print(f"  gates: acq>=.65 final>=.65 meanF_hon<=3pp worstF<=5pp")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
