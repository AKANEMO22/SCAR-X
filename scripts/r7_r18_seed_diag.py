"""R7 round 18 — why is seed 140023 stationary while the other two improve?

ROUND 17 LEFT THIS OPEN
-----------------------
    cell           acq     final    meanF    per-seed meanF
    no_replay    0.8819   0.6693   31.90pp   (aggregate)
    buf128_wt1   0.8728   0.8216    7.68pp   5.66 / 5.86 / 11.52

Seeds 140011 and 140017 fall to ~5.7pp; seed 140023 sits at 11.52pp across every buffer
size and every replay weight. Tuning further would fit the mean while one third of the
seeds does not move.

THREE COMPETING EXPLANATIONS
-----------------------------
H1  REPRESENTATION. Seed 140023's features genuinely cannot hold all three tasks, so no
    head-side mechanism can help. Predicts: its JOINT ceiling (all data at once) is much
    lower than the other seeds'.

H2  INTERFERENCE. Its joint ceiling is normal, but sequential training destroys it more.
    Predicts: joint high, sequential low, gap larger than other seeds.

H3  OPTIMISATION. The head is saturated / under-converged for that seed's feature scale.
    Predicts: TRAIN accuracy on the final task is also low, i.e. the head cannot even fit.

DISCRIMINATING MEASUREMENTS (each seed separately)
    joint_ceiling      head trained on the union of all three tasks, then evaluated per task
    sequential_R       the R matrix under the replay setting that was best overall
    head_train_acc     train accuracy at the last task — separates H3 from H1/H2
    feature_overlap    mean |corr| between per-task feature means, a direct overlap proxy

H3 is ruled out if head_train_acc is high while test is low.
H1 is supported if joint_ceiling is materially lower for this seed.
H2 is supported if joint is high for all seeds but the sequential gap is larger here.
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
BUF, WT = 128, 1          # the best cell from round 17


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

    def predict(self, F):
        return np.argmax(self.z(F) @ self.w.T + self.b, 1)

    def acc(self, F, y):
        return float(np.mean(self.predict(F) == y))


def main():
    print("=" * 96)
    print("R7 ROUND 18 — per-seed diagnosis: representation (H1) vs interference (H2) vs optimisation (H3)")
    print("=" * 96)

    out_rows = []
    for seed in DEV:
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        data = {}
        for tk in TASKS:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            data[tk] = (trx, try_, tex, tey)

        F_tr = {tk: feats(core, data[tk][0]) for tk in TASKS}
        F_te = {tk: feats(core, data[tk][2]) for tk in TASKS}

        # ---------- H1: joint ceiling
        Fj = np.concatenate([F_tr[tk] for tk in TASKS])
        yj = np.concatenate([data[tk][1] for tk in TASKS])
        h = Head(Fj.shape[1]); h.train(Fj, yj)
        joint = {tk: h.acc(F_te[tk], data[tk][3]) for tk in TASKS}
        joint_mean = float(np.mean(list(joint.values())))
        joint_tr = h.acc(Fj, yj)

        # ---------- H2: sequential R with the best replay cell
        h2 = Head(Fj.shape[1])
        buf_seq, buf_lab = [], []
        n = len(TASKS)
        R = np.full((n, n), np.nan)
        last_train_acc = None
        for i_t, tk in enumerate(TASKS):
            trx, try_, tex, tey = data[tk]
            h2.train(F_tr[tk], try_)
            idx = np.random.default_rng(seed + i_t).choice(N_TR, min(BUF, N_TR),
                                                          replace=False)
            buf_seq.extend(trx[j].copy() for j in idx)
            buf_lab.extend(int(try_[j]) for j in idx)
            if buf_seq:
                h2.train(feats(core, np.stack(buf_seq)), np.array(buf_lab, np.int32),
                         epochs=800 * WT)
            if i_t == n - 1:
                A = h2.z(F_tr[tk])
                last_train_acc = float(np.mean(np.argmax(A @ h2.w.T + h2.b, 1) == try_))
            for j_t, tk2 in enumerate(TASKS):
                R[i_t, j_t] = h2.acc(F_te[tk2], data[tk2][3])

        diag = np.diag(R)
        final = R[n - 1]
        fg = diag - final
        seq_meanF = float(np.mean(fg[:n - 1]) * 100)

        # ---------- overlap proxy: correlation between per-task feature means
        means = np.stack([F_tr[tk].mean(0) for tk in TASKS])
        C = np.corrcoef(means)
        off = C[~np.eye(n, dtype=bool)]

        row = {"seed": seed, "joint_mean": joint_mean, "joint_train": joint_tr,
               "joint_per_task": {k: round(v, 4) for k, v in joint.items()},
               "seq_meanF": seq_meanF, "seq_acq": float(np.mean(diag)),
               "seq_final": float(np.mean(final)),
               "head_train_acc_last": last_train_acc,
               "task_mean_corr": float(np.abs(off).mean()),
               "R": [[round(v, 4) for v in r] for r in R]}
        out_rows.append(row)
        print(f"\n  seed {seed}")
        print(f"    joint ceiling      {joint_mean:.4f}  (train {joint_tr:.4f})  "
              f"per-task " + " ".join(f"{k}={joint[k]:.3f}" for k in TASKS))
        print(f"    sequential R       " + str([[round(v, 3) for v in r] for r in R]))
        print(f"    seq acq {float(np.mean(diag)):.4f}  final {float(np.mean(final)):.4f}  "
              f"meanF {seq_meanF:.2f}pp")
        print(f"    head TRAIN acc at last task {last_train_acc:.4f}")
        print(f"    between-task feature-mean |corr| {float(np.abs(off).mean()):.4f}")

    print("\n" + "=" * 96)
    print("DIAGNOSIS")
    print("=" * 96)
    print(f"  {'seed':>8} {'joint':>8} {'joint_tr':>9} {'seq acq':>8} {'seq meanF':>10} "
          f"{'gap':>8} {'train_last':>11} {'|corr|':>8}")
    for r in out_rows:
        gap = r["joint_mean"] - r["seq_final"]
        print(f"  {r['seed']:>8} {r['joint_mean']:8.4f} {r['joint_train']:9.4f} "
              f"{r['seq_acq']:8.4f} {r['seq_meanF']:9.2f}p {gap:8.4f} "
              f"{r['head_train_acc_last']:11.4f} {r['task_mean_corr']:8.4f}")

    worst = max(out_rows, key=lambda r: r["seq_meanF"])
    best = min(out_rows, key=lambda r: r["seq_meanF"])
    print(f"\n  worst seed {worst['seed']}: meanF {worst['seq_meanF']:.2f}pp, "
          f"joint {worst['joint_mean']:.4f}")
    print(f"  best  seed {best['seed']}: meanF {best['seq_meanF']:.2f}pp, "
          f"joint {best['joint_mean']:.4f}")
    print(f"  joint ceiling spread across seeds: "
          f"{max(r['joint_mean'] for r in out_rows) - min(r['joint_mean'] for r in out_rows):.4f}")

    h1 = worst["joint_mean"] < best["joint_mean"] - 0.05
    h3 = worst["head_train_acc_last"] > 0.90
    print("\n  H1 representation (joint ceiling materially lower for the worst seed):",
          "SUPPORTED" if h1 else "NOT SUPPORTED")
    print("  H3 optimisation (head cannot even fit its own task):",
          "RULED OUT" if h3 else "POSSIBLE")
    if not h1 and h3:
        print("  => INTERFERENCE (H2). The worst seed can represent all three jointly")
        print("     and its head fits, so the loss is created by sequential updating.")
        verdict = "H2_interference"
    elif h1:
        print("  => REPRESENTATION (H1). The worst seed's features cannot hold the union.")
        verdict = "H1_representation"
    else:
        print("  => needs another look")
        verdict = "unclear"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed-diagnosis.json").write_text(
        json.dumps({"per_seed": out_rows, "verdict": verdict,
                    "best_cell": {"buf": BUF, "wt": WT}},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/seed-diagnosis.json")


if __name__ == "__main__":
    main()
