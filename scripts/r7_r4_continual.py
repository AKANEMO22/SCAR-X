"""R7 round 4 — continual learning with a CAUSAL decomposition of forgetting.

The mission's second goal is untouched so far. Rounds 1-3 established that the
representation is limited by (1) a 3-tick memory horizon, (2) no cross-tick conjunction,
(3) a lossy readout channel, (4) a directionless credit trace.

Given that, a retention experiment could easily measure "forgetting" that is really
"the representation never held the skill". So this round decomposes instead of
measuring a single number.

THREE SOURCES OF FORGETTING (brief section X), each with its own arm:

  S1  hidden synapses change and destroy old features
      arm: `learned_shared`  (production hidden learning, one shared head)
      vs    `frozen_shared`  (hidden frozen, one shared head)
      The difference is the HIDDEN contribution.

  S2  the output head changes and can no longer read old features
      arm: `frozen_shared`   (one head, sequentially overwritten)
      vs    `frozen_pertask` (one head PER task, so no readout interference)
      The difference is the READOUT contribution.

  S3  no replay / no cue to reinstate the state
      arm: `frozen_replay` and `learned_replay` (a bounded replay buffer under the
      SAME per-task update budget)

TASKS respect the measured 3-tick horizon from round 1. Anything needing a longer
memory would confound forgetting with a decayed signal.

Metrics per the mission's gates: acquisition, final accuracy, mean forgetting, worst
forgetting. Thresholds are NOT relaxed: acquisition >= 65%, final >= 65%,
mean forgetting <= 3pp, worst <= 5pp.
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


# ---------------------------------------------------------------- tasks
def sim_xor(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        a = int(rng.integers(2)); b = int(rng.integers(2))
        y[i] = a ^ b
        t = int(rng.integers(1, 4))
        x[i, t, 0] += 1.5 * a
        x[i, t, 1] += 1.5 * b
        x[i, 10, 7] += 1.5
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
    """Which event came first, with a 2-tick gap (inside the measured horizon)."""
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


# ------------------------------------------------------------- channels
def readout_feats(core, x):
    """Round-1 channel finding applied: spike sum, not the lossy decay."""
    out = []
    for i in range(len(x)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(x[i, t], track=False)
            tot += core.spikes
        out.append(tot)
    return np.stack(out)


def learn_episode(core, seq, label, do_hidden):
    core.reset_state()
    for t in range(len(seq)):
        core.step(seq[t], track=True)
    core.learn(int(label), mode="full" if do_hidden else "frozen")


# ----------------------------------------------------------------- head
class Head:
    def __init__(self, dim, seed=0):
        self.mu = None; self.sd = None
        g = np.random.default_rng(seed + 31)
        self.w = g.normal(0, .01, (2, dim))
        self.b = np.zeros(2)

    def fit_scaler(self, F):
        self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 1e-6)

    def z(self, F):
        return (F - self.mu) / self.sd

    def train(self, F, y, epochs=400, lr=.5):
        if self.mu is None:
            self.fit_scaler(F)
        A = self.z(F)
        for _ in range(epochs):
            lg = A @ self.w.T + self.b; lg -= lg.max(1, keepdims=True)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            e = -p; e[np.arange(len(y)), y] += 1
            self.w += lr * (e.T @ A) / len(y); self.b += lr * e.mean(0)

    def acc(self, F, y):
        return float(np.mean(np.argmax(self.z(F) @ self.w.T + self.b, 1) == y))

    def clone(self):
        h = Head(self.w.shape[1])
        h.w = self.w.copy(); h.b = self.b.copy()
        h.mu = None if self.mu is None else self.mu.copy()
        h.sd = None if self.sd is None else self.sd.copy()
        return h


# ------------------------------------------------------------- protocol
def run():
    root = Path(__file__).resolve().parents[1]
    run = new_run(root, "r7-r4-continual-v1", {
        "seeds": list(DEV), "tasks": list(TASKS), "n_train": N_TR, "n_test": N_TE,
        "channel": "spike_sum", "note": "DEV only; causal decomposition of forgetting",
        "gates": {"acquisition": .65, "final": .65,
                  "mean_forgetting_pp": 3, "worst_forgetting_pp": 5}})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            data = {}
            for tk in TASKS:
                trx, try_ = GENS[tk](seed + 200_000, N_TR)
                tex, tey = GENS[tk](seed + 300_000, N_TE)
                data[tk] = (trx, try_, tex, tey)

            arms = {}

            # ---- S2 baseline: one head PER task (no readout interference at all)
            cores = {}
            for tk in TASKS:
                core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
                trx, try_, tex, tey = data[tk]
                for i in range(N_TR):
                    learn_episode(core, trx[i], try_[i], do_hidden=False)
                F = readout_feats(core, trx); Ft = readout_feats(core, tex)
                h = Head(F.shape[1], seed); h.train(F, try_)
                arms.setdefault("frozen_pertask", {})[tk] = h.acc(Ft, tey)

            # ---- sequential arms
            for arm, do_hidden, replay in (("frozen_shared", False, False),
                                           ("learned_shared", True, False),
                                           ("frozen_replay", False, True),
                                           ("learned_replay", True, True)):
                core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
                head = None
                acc = {}
                buf = []          # bounded replay buffer: 64 per seen task
                for tk in TASKS:
                    trx, try_, tex, tey = data[tk]
                    for i in range(N_TR):
                        learn_episode(core, trx[i], try_[i], do_hidden=do_hidden)
                        if replay and i % 6 == 0 and len(buf) < 192:
                            buf.append((trx[i].copy(), int(try_[i]), tk))

                    F = readout_feats(core, trx)
                    if head is None:
                        head = Head(F.shape[1], seed)
                    # same update budget: the arm sees task data, plus replay if enabled
                    head.train(F, try_)
                    if replay and buf:
                        RF = np.stack([readout_feats(core, np.stack([b[0]]))[0]
                                       for b in buf])
                        Ry = np.array([b[1] for b in buf], np.int32)
                        head.train(RF, Ry)

                    for tk2 in TASKS:
                        if tk2 not in data:
                            continue
                        _, _, tex2, tey2 = data[tk2]
                        Ft2 = readout_feats(core, tex2)
                        acc[(tk, tk2)] = head.acc(Ft2, tey2)
                arms[arm] = {}
                for (just, evaluated), v in acc.items():
                    arms[arm].setdefault(evaluated, []).append(v)
                arms[arm] = {tk2: float(arms[arm][tk2][-1]) for tk2 in arms[arm]}

            rows.append({"seed": seed, "arms": arms,
                         "hidden_digest": S.digest(core)[:16]})
            mon.check()
            for a in arms:
                print(f"  seed{seed} {a:16s} " +
                      " ".join(f"{tk}={arms[a].get(tk, float('nan')):.3f}"
                               for tk in TASKS), flush=True)

    # ---- aggregate
    def agg(arm, tk):
        vals = [r["arms"][arm].get(tk) for r in rows
                if arm in r["arms"] and r["arms"][arm].get(tk) is not None]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {}
    for arm in ("frozen_pertask", "frozen_shared", "learned_shared",
                "frozen_replay", "learned_replay"):
        per = {tk: agg(arm, tk) for tk in TASKS}
        acq = per[TASKS[0]]
        final = float(np.mean([per[tk] for tk in TASKS]))
        forget = {tk: per[TASKS[0]] - per[tk] for tk in TASKS[1:]}
        mean_f = float(np.mean(list(forget.values())))
        worst_f = float(np.max(list(forget.values())))
        summary[arm] = {
            "per_task": per, "acquisition": acq, "final": final,
            "mean_forgetting_pp": mean_f * 100, "worst_forgetting_pp": worst_f * 100,
            "passes": bool(acq >= .65 and final >= .65
                           and mean_f * 100 <= 3 and worst_f * 100 <= 5)}

    write_json(run / "result.json",
               {"trials": rows, "summary": summary,
                "resources": dict(mon.report()),
                "wall_seconds": time.time() - t0})

    print("\n" + "=" * 92)
    print("R7 ROUND 4 — continual learning, causal decomposition (3 seeds)")
    print("=" * 92)
    print(f"  {'arm':17s} {'acq':>7} {'final':>7} {'meanF':>8} {'worstF':>8} "
          f"{'gate':>6}  per-task")
    for arm, s in summary.items():
        print(f"  {arm:17s} {s['acquisition']:7.4f} {s['final']:7.4f} "
              f"{s['mean_forgetting_pp']:7.2f}p {s['worst_forgetting_pp']:7.2f}p "
              f"{('PASS' if s['passes'] else 'fail'):>6}  " +
              " ".join(f"{tk}={s['per_task'][tk]:.3f}" for tk in TASKS))
    print(f"\n  gates: acq>=0.65, final>=0.65, meanF<=3pp, worstF<=5pp (NOT relaxed)")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
