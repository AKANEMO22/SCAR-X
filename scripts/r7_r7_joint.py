"""R7 round 7 — is the 3-task benchmark solvable by a linear head AT ALL?

Round 6 corrected my own round-5b claim: the honest mean forgetting is 4.30pp, not the
2.86pp I reported, because I averaged over a task whose forgetting is zero by
construction.

Before proposing any more continual-learning mechanism, one diagnostic decides whether the
whole line of work is well-posed:

    If a linear head trained JOINTLY on all three tasks (all data available at once, no
    forgetting possible) cannot reach the gates, then no sequential method can either,
    and the benchmark asks for more than the readout can represent.

    If joint training DOES reach the gates, the capacity exists and the problem is
    genuinely about interference — which is solvable in principle.

This is the single most informative control in the retention line, and I had not run it.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
TASKS = ("sim_xor", "parity3", "order_short")
GATES = {"acq": .65, "final": .65}


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


def train_linear(F, y, epochs=1500, lr=.5):
    mu = F.mean(0); sd = np.maximum(F.std(0), 1e-6)
    A = (F - mu) / sd
    w = np.zeros((2, F.shape[1])); b = np.zeros(2)
    for _ in range(epochs):
        lg = A @ w.T + b; lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        e = -p; e[np.arange(len(y)), y] += 1
        w += lr * (e.T @ A) / len(y); b += lr * e.mean(0)
    return mu, sd, w, b


def main():
    print("=" * 86)
    print("R7 ROUND 7 — can a LINEAR head solve all three tasks JOINTLY?")
    print("=" * 86)
    print("  (joint = all data available at once, so forgetting is impossible)")
    print(f"\n  {'seed':>8} {'sim_xor':>9} {'parity3':>9} {'order_short':>12} "
          f"{'mean':>8} {'gate':>6}")
    results = []
    for seed in DEV:
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Fs, ys, Fts, yts = [], [], [], []
        for tk in TASKS:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            Fs.append(readout_feats(core, trx)); ys.append(try_)
            Fts.append(readout_feats(core, tex)); yts.append(tey)
        F = np.concatenate(Fs); y = np.concatenate(ys)
        mu, sd, w, b = train_linear(F, y)

        accs = []
        for k, tk in enumerate(TASKS):
            A = (Fts[k] - mu) / sd
            acc = float(np.mean(np.argmax(A @ w.T + b, 1) == yts[k]))
            accs.append(acc)
        mean = float(np.mean(accs))
        ok = mean >= GATES["final"] and min(accs) >= GATES["acq"]
        results.append({"seed": seed, "accs": accs, "mean": mean, "passes": ok})
        print(f"  {seed:>8} {accs[0]:9.4f} {accs[1]:9.4f} {accs[2]:12.4f} "
              f"{mean:8.4f} {('PASS' if ok else 'fail'):>6}")

    mean_all = float(np.mean([r["mean"] for r in results]))
    worst_task = float(np.min([min(r["accs"]) for r in results]))
    print(f"\n  joint mean over seeds : {mean_all:.4f}")
    print(f"  joint worst task/seed : {worst_task:.4f}")
    print(f"  gates: every task >= {GATES['acq']}, mean >= {GATES['final']}")

    if worst_task >= GATES["acq"]:
        print("\n  => JOINT SOLVABLE. A linear head CAN represent all three.")
        print("     The retention problem is genuine interference, not capacity")
        print("     with respect to the gate specifically.")
    else:
        print("\n  => JOINT NOT SOLVABLE within the gates. A linear head trained on")
        print("     ALL data at once cannot hit the threshold on every task, so no")
        print("     sequential method can either. The benchmark asks more of a")
        print("     2-logit linear head than it can represent.")

    (ROOT / "artifacts" / "deepseek-r7").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "deepseek-r7" / "joint-solvability.json").write_text(
        json.dumps({"tasks": TASKS, "gates": GATES, "per_seed": results,
                    "joint_mean": mean_all, "joint_worst": worst_task,
                    "joint_solvable": bool(worst_task >= GATES["acq"])},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/joint-solvability.json")


if __name__ == "__main__":
    main()
