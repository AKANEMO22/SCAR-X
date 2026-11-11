"""R7 round 16 — replay was rejected on a LABEL BUG. Re-testing it correctly.

THE BUG (found by re-reading round 4's script)
----------------------------------------------
In `r7_r4b_rmatrix.py` the replay buffer stored only the SEQUENCE:

    buf.append(trx[i].copy())            # sequence, no label
    ...
    RF = readout_feats(core, np.stack(buf))
    Ry = np.array([try_[k % N_TR] for k in range(len(buf))], np.int32)   # <-- WRONG
    head.train(RF, Ry)

`try_` is the CURRENT task's label array. So the replay examples were trained against
labels drawn from the current task, applied positionally to stored sequences from
PREVIOUS tasks. The replay arm was learning essentially random labels.

That is why round 4 reported:

    frozen_replay  acq 0.4818  meanF -0.04pp
    learned_replay acq 0.4688  meanF  0.26pp

I concluded "replay buys retention by not learning" and declared FP3. **That conclusion
was an artefact of my own bug.** Replay was never fairly tested.

WHAT THIS RUNS — replay done correctly
    buf stores (sequence, LABEL, task_id) together
    replay trains the head on those stored (sequence, label) pairs
    identical per-task update budget for every arm

ARMS
    frozen_shared      no replay (the 31.4pp baseline)
    frozen_replay      correct replay, budget matched
    learned_replay     correct replay with production hidden learning
    frozen_pertask     instrument check: cannot forget, must read ~0

PREDICTION (registered before running): with correct labels, replay should reach
meanF near 0 WITHOUT collapsing acquisition. If acquisition still collapses, the
round-4 conclusion stands for a different reason and I will say so.
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
REPLAY_KEEP = 128          # sequences retained per seen task


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


def run_arm(seed, arm, data):
    core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
    n = len(TASKS)
    R = np.full((n, n), np.nan)

    if arm == "frozen_pertask":
        for tk in TASKS:
            core_t = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
            trx, try_, tex, tey = data[tk]
            F = feats(core_t, trx)
            h = Head(F.shape[1]); h.train(F, try_)
            for j, tk2 in enumerate(TASKS):
                _, _, tex2, tey2 = data[tk2]
                R[j if tk2 == tk else 0, j] = h.acc(feats(core_t, tex2), tey2)
        # structural: each task learned in isolation, its own row/col
        R = np.full((n, n), np.nan)
        for j, tk in enumerate(TASKS):
            core_t = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
            trx, try_, tex, tey = data[tk]
            F = feats(core_t, trx)
            h = Head(F.shape[1]); h.train(F, try_)
            R[:, j] = h.acc(feats(core_t, tex), tey)
        return {"R": R.tolist()}

    head = Head(S.cfg(seed).hidden)
    buf_seq, buf_lab = [], []          # <-- LABELS STORED WITH SEQUENCES
    do_hidden = arm == "learned_replay"
    replay = arm in ("frozen_replay", "learned_replay")

    for i_t, tk in enumerate(TASKS):
        trx, try_, tex, tey = data[tk]
        for i in range(N_TR):
            core.reset_state()
            for t in range(12):
                core.step(trx[i, t], track=True)
            if do_hidden:
                core.learn(int(try_[i]), mode="full")
            if replay and i % 3 == 0 and len(buf_seq) < REPLAY_KEEP * (i_t + 1):
                buf_seq.append(trx[i].copy())
                buf_lab.append(int(try_[i]))       # the CORRECT label

        F = feats(core, trx)
        head.train(F, try_)
        if replay and buf_seq:
            RF = feats(core, np.stack(buf_seq))
            Ry = np.array(buf_lab, np.int32)        # correct pairs
            head.train(RF, Ry)

        for j_t, tk2 in enumerate(TASKS):
            _, _, tex2, tey2 = data[tk2]
            R[i_t, j_t] = head.acc(feats(core, tex2), tey2)
    return {"R": R.tolist()}


def main():
    print("=" * 92)
    print("R7 ROUND 16 — replay re-tested with CORRECT labels")
    print("=" * 92)
    rows = []
    for seed in DEV:
        data = {}
        for tk in TASKS:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            data[tk] = (trx, try_, tex, tey)
        for arm in ("frozen_shared", "frozen_replay", "learned_replay",
                    "frozen_pertask"):
            if arm == "frozen_shared":
                R = np.full((len(TASKS), len(TASKS)), np.nan)
                core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
                head = Head(S.cfg(seed).hidden)
                for i_t, tk in enumerate(TASKS):
                    trx, try_, tex, tey = data[tk]
                    head.train(feats(core, trx), try_)
                    for j_t, tk2 in enumerate(TASKS):
                        _, _, tex2, tey2 = data[tk2]
                        R[i_t, j_t] = head.acc(feats(core, tex2), tey2)
                out = {"R": R.tolist()}
            else:
                out = run_arm(seed, arm, data)
            rows.append({"seed": seed, "arm": arm, **out})
            print(f"  seed{seed} {arm:16s} R="
                  f"{np.round(np.array(out['R']), 3).tolist()}", flush=True)

    def summ(arm):
        Rs = [np.array(r["R"]) for r in rows if r["arm"] == arm]
        n = len(TASKS)
        diag = [np.diag(R) for R in Rs]
        final = [R[n - 1] for R in Rs]
        acq = float(np.mean([np.mean(d) for d in diag]))
        per = float(np.mean([np.mean(f) for f in final]))
        fg = [d - f for d, f in zip(diag, final)]
        mf = float(np.mean([np.mean(x[:n - 1]) for x in fg])) * 100
        wf = float(np.max([np.max(x) for x in fg])) * 100
        return {"acq": acq, "final": per, "meanF": mf, "worstF": wf,
                "passes": bool(acq >= GATES["acq"] and per >= GATES["final"]
                               and mf <= GATES["meanF_pp"] and wf <= GATES["worstF_pp"])}

    print("\n" + "=" * 92)
    print("RESULTS")
    print("=" * 92)
    print(f"  {'arm':18s} {'acq':>7} {'final':>7} {'meanF':>9} {'worstF':>9} {'gate':>6}")
    summary = {}
    for arm in ("frozen_shared", "frozen_replay", "learned_replay", "frozen_pertask"):
        s = summ(arm)
        summary[arm] = s
        print(f"  {arm:18s} {s['acq']:7.4f} {s['final']:7.4f} {s['meanF']:8.2f}p "
              f"{s['worstF']:8.2f}p {('PASS' if s['passes'] else 'fail'):>6}")

    print(f"\n  round-4 (BUGGY labels): frozen_replay acq 0.4818 meanF -0.04pp")
    print(f"  round-16 (correct)    : frozen_replay acq "
          f"{summary['frozen_replay']['acq']:.4f} meanF "
          f"{summary['frozen_replay']['meanF']:.2f}pp")
    print(f"  instrument check: frozen_pertask meanF "
          f"{summary['frozen_pertask']['meanF']:.2f}pp (must be ~0)")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay-corrected.json").write_text(
        json.dumps({"per_seed": rows, "summary": summary, "gates": GATES,
                    "correction": "round 4 replay used current-task labels for stored "
                                  "sequences from previous tasks",
                    "round4_buggy": {"frozen_replay_acq": 0.4818,
                                     "frozen_replay_meanF": -0.04}},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/replay-corrected.json")


if __name__ == "__main__":
    main()
