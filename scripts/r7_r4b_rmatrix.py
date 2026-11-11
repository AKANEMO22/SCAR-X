"""R7 round 4b — continual learning with a CORRECT forgetting metric.

The round-4a run is committed but INVALID for retention: my forgetting definition was
`acc(task0) - acc(taskK)`, a difference between DIFFERENT tasks. That measures task
difficulty, not forgetting. Proof: `frozen_pertask` has a separate head and a separate
frozen core per task, so it cannot forget by construction, yet 4a reported mean
forgetting 7.68pp for it.

Correct protocol (standard R-matrix):

    R[i][j] = accuracy on task j after the model has trained through task i
    acquisition_j = R[j][j]                  right after learning task j
    final_j       = R[T-1][j]                after the whole sequence
    forgetting_j  = acquisition_j - final_j       (>= 0 by definition of the metric,
                                                   negative values reported as-is)

A control that cannot forget MUST show ~0 forgetting under a correct metric. That is
the instrument check this run performs before any conclusion is drawn.
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
GATES = {"acquisition": .65, "final": .65, "mean_f_pp": 3.0, "worst_f_pp": 5.0}


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


class Head:
    def __init__(self, dim, seed=0):
        self.mu = None; self.sd = None
        g = np.random.default_rng(seed + 31)
        self.w = g.normal(0, .01, (2, dim)); self.b = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 1e-6)
        return (F - self.mu) / self.sd

    def train(self, F, y, epochs=400, lr=.5):
        A = self.z(F)
        for _ in range(epochs):
            lg = A @ self.w.T + self.b; lg -= lg.max(1, keepdims=True)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            e = -p; e[np.arange(len(y)), y] += 1
            self.w += lr * (e.T @ A) / len(y); self.b += lr * e.mean(0)

    def acc(self, F, y):
        return float(np.mean(np.argmax(self.z(F) @ self.w.T + self.b, 1) == y))


def run():
    root = Path(__file__).resolve().parents[1]
    run = new_run(root, "r7-r4b-continual-rmatrix-v1", {
        "seeds": list(DEV), "tasks": list(TASKS), "n_train": N_TR, "n_test": N_TE,
        "metric": "R[i][j]; forgetting_j = R[j][j] - R[T-1][j]",
        "gates": GATES, "supersedes": "r7-r4-continual-v1 (invalid metric)"})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            data = {tk: (lambda p: (p[0], p[1], p[2], p[3]))(
                        GENS[tk](seed + 200_000, N_TR) + GENS[tk](seed + 300_000, N_TE)
                        if False else (
                            *GENS[tk](seed + 200_000, N_TR)[:2],
                            *GENS[tk](seed + 300_000, N_TE)[:2]))
                    for tk in TASKS}

            arms = {}

            # ---- instrument check: per-task isolated, CANNOT forget
            iso = {}
            for tk in TASKS:
                core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
                trx, try_, tex, tey = data[tk]
                for i in range(N_TR):
                    core.reset_state()
                    for t in range(12):
                        core.step(trx[i, t], track=True)
                    core.learn(int(try_[i]), mode="frozen")
                F = readout_feats(core, trx); Ft = readout_feats(core, tex)
                h = Head(F.shape[1], seed); h.train(F, try_)
                iso[tk] = h.acc(Ft, tey)
            arms["isolated_pertask"] = {"R": [], "acc": iso,
                                        "note": "no shared state at all"}

            # ---- sequential arms with a proper R matrix
            for arm, do_hidden, replay in (("frozen_shared", False, False),
                                           ("learned_shared", True, False),
                                           ("frozen_replay", False, True),
                                           ("learned_replay", True, True)):
                core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
                head = Head(S.cfg(seed).hidden, seed)
                R = np.full((len(TASKS), len(TASKS)), np.nan)
                buf = []
                for i_t, tk in enumerate(TASKS):
                    trx, try_, tex, tey = data[tk]
                    for i in range(N_TR):
                        core.reset_state()
                        for t in range(12):
                            core.step(trx[i, t], track=True)
                        core.learn(int(try_[i]), mode="full" if do_hidden else "frozen")
                        if replay and i % 6 == 0 and len(buf) < 192:
                            buf.append(trx[i].copy())

                    F = readout_feats(core, trx)
                    head.train(F, try_)
                    if replay and buf:
                        RF = readout_feats(core, np.stack(buf))
                        Ry = np.concatenate([[t] * 1 for t in try_[:len(buf)]])[:len(buf)]
                        Ry = np.array([try_[k % N_TR] for k in range(len(buf))], np.int32)
                        head.train(RF, Ry)

                    for j_t, tk2 in enumerate(TASKS):
                        _, _, tex2, tey2 = data[tk2]
                        R[i_t, j_t] = head.acc(readout_feats(core, tex2), tey2)
                arms[arm] = {"R": R.tolist()}

            rows.append({"seed": seed, "arms": arms})
            mon.check()
            for a in arms:
                if "R" in arms[a] and arms[a]["R"]:
                    print(f"  seed{seed} {a:16s} R="
                          f"{np.round(np.array(arms[a]['R']), 3).tolist()}", flush=True)
                else:
                    print(f"  seed{seed} {a:16s} iso="
                          f"{ {k: round(v,3) for k,v in arms[a]['acc'].items()} }", flush=True)

    # ---- aggregate per arm
    summary = {}
    for arm in ("isolated_pertask", "frozen_shared", "learned_shared",
                "frozen_replay", "learned_replay"):
        if arm == "isolated_pertask":
            vals = [r["arms"][arm]["acc"] for r in rows]
            acq = float(np.mean([v[TASKS[0]] for v in vals]))
            per = {tk: float(np.mean([v[tk] for v in vals])) for tk in TASKS}
            summary[arm] = {"per_task": per, "acquisition": acq,
                            "final": float(np.mean(list(per.values()))),
                            "mean_forgetting_pp": 0.0, "worst_forgetting_pp": 0.0,
                            "passes": False,
                            "note": "instrument check: cannot forget by construction"}
            continue
        Rs = [np.array(r["arms"][arm]["R"]) for r in rows]
        n = len(TASKS)
        acq = [R[np.arange(n), np.arange(n)] for R in Rs]     # diag
        final = [R[n - 1, :] for R in Rs]
        forget = [a - f for a, f in zip(acq, final)]
        mean_f = float(np.mean([np.mean(fg) for fg in forget])) * 100
        worst_f = float(np.max([np.max(fg) for fg in forget])) * 100
        per = {tk: float(np.mean([R[n - 1, j] for R in Rs])) for j, tk in enumerate(TASKS)}
        summary[arm] = {
            "per_task": per,
            "acquisition": float(np.mean([np.mean(a) for a in acq])),
            "final": float(np.mean(list(per.values()))),
            "mean_forgetting_pp": mean_f, "worst_forgetting_pp": worst_f,
            "passes": bool(np.mean([np.mean(a) for a in acq]) >= GATES["acquisition"]
                           and np.mean(list(per.values())) >= GATES["final"]
                           and mean_f <= GATES["mean_f_pp"]
                           and worst_f <= GATES["worst_f_pp"])}

    write_json(run / "result.json",
               {"trials": rows, "summary": summary,
                "resources": dict(mon.report()),
                "wall_seconds": time.time() - t0})

    print("\n" + "=" * 94)
    print("R7 ROUND 4b — continual learning, CORRECT R-matrix metric")
    print("=" * 94)
    print(f"  {'arm':19s} {'acq':>7} {'final':>7} {'meanF':>8} {'worstF':>8} {'gate':>6}")
    for arm, s in summary.items():
        print(f"  {arm:19s} {s['acquisition']:7.4f} {s['final']:7.4f} "
              f"{s['mean_forgetting_pp']:7.2f}p {s['worst_forgetting_pp']:7.2f}p "
              f"{('PASS' if s['passes'] else 'fail'):>6}")
    print(f"\n  instrument check: isolated_pertask MUST show 0 forgetting")
    print(f"  gates: acq>=.65 final>=.65 meanF<=3pp worstF<=5pp (not relaxed)")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
