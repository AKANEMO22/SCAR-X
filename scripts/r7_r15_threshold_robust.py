"""R7 round 15 — is the threshold gain REAL, and does it reach the second mission goal?

THE FINDING I MUST TEST ON MYSELF
---------------------------------
Round 14's threshold sweep: parity3 accuracy 0.8320 at production threshold 0.50 versus
0.9271 at threshold 1.00 — +9.5pp. Round 13's `hta_scale` saw +9.24pp from the same
mechanism (raising thresholds). That is the largest single effect I have measured in
fifteen rounds, and it is a HYPERPARAMETER change, not a learning rule.

Two ways it could be fake:

  FP4  it helps parity3 and hurts other tasks -> benchmark artefact, not a better code
  FP1  it helps because the task has few logical states and sparse codes happen to
       memorise them -> accuracy without representation

And one way it could be genuinely important: the mission's SECOND goal (retention) is
unsolved at 31.45pp forgetting. If a sparser code is more separable, interference should
drop. That would connect my best finding to my worst problem.

WHAT THIS RUNS
  for threshold in {0.50, 0.70, 1.00, 1.30}:
    for task in {sim_xor, parity3, order_short, delayed_short}:
      frozen accuracy          (does it generalise across tasks?)
      effective rank           (correctly summed features)
      true firing rate         (full spike train, not the last tick)
  and the retention R-matrix at each threshold on a 3-task sequence.

GATES (registered before running):
  a threshold change is only "real" if it helps on >= 3 of 4 tasks and hurts none by
  more than 3pp. If it helps 1-2 tasks, it is a benchmark artefact and I will say so.

Threshold swept as a DIAGNOSTIC. Production threshold unchanged.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.neural.core import Config  # noqa: E402
from scarx.evaluation import r7_secs as S  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
THRESHOLDS = (0.50, 0.70, 1.00, 1.30)
TASKS = ("sim_xor", "parity3", "order_short", "delayed_short")


def cfg(seed, threshold):
    base = S.cfg(seed)
    f = {k: getattr(base, k) for k in base.__dataclass_fields__}
    f["threshold"] = threshold
    return Config(**f)


# ------------------------------------------------------------------ tasks
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


def delayed_short(seed, n):
    """Two bits separated by 2 ticks (inside the measured 3-tick horizon)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        a = int(rng.integers(2)); b = int(rng.integers(2)); y[i] = a ^ b
        x[i, 2, 0] += 1.5 * a
        x[i, 4, 1] += 1.5 * b
        x[i, 10, 7] += 1.5
    return x, y


GENS = {"sim_xor": sim_xor, "parity3": parity3,
        "order_short": order_short, "delayed_short": delayed_short}


# --------------------------------------------------------------- helpers
def softmax(z):
    z = np.asarray(z, float)
    one = z.ndim == 1
    if one:
        z = z[None, :]
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    out = e / e.sum(1, keepdims=True)
    return out[0] if one else out


def feats_and_train(core, X):
    """Summed features AND the full spike train (round-14 lesson)."""
    out, train = [], []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        fired = np.zeros((12, core.config.hidden), np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
            fired[t] = core.spikes
        out.append(tot)
        train.append(fired)
    return np.stack(out), np.stack(train)


def eff_rank(F):
    C = np.cov(F - F.mean(0), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def fit_head(F, y, epochs=1200, lr=.5):
    """z-scored on TRAIN stats, floor 0.05 (relative, not absolute)."""
    mu = F.mean(0)
    sd = np.maximum(F.std(0), 0.05)
    A = (F - mu) / sd
    w = np.zeros((2, A.shape[1])); b = np.zeros(2)
    for _ in range(epochs):
        p = softmax(A @ w.T + b)
        e = -p; e[np.arange(len(y)), y] += 1
        w += lr * (e.T @ A) / len(y); b += lr * e.mean(0)
    return mu, sd, w, b


def head_acc(F, y, mu, sd, w, b):
    return float(np.mean(np.argmax(((F - mu) / sd) @ w.T + b, 1) == y))


# ------------------------------------------------------------- retention
def retention(seed, threshold, trx_all, try_all, tex_all, tey_all):
    """Sequential R-matrix with the CORRECT metric (round 4/6 lessons)."""
    core = S.SECSCore(cfg(seed, threshold), mode="scalar", seed=seed)
    head = {"mu": None, "sd": None, "w": None, "b": None}
    SEQ = TASKS[:3]
    n = len(SEQ)
    R = np.full((n, n), np.nan)

    for i_t, tk in enumerate(SEQ):
        F, _ = feats_and_train(core, trx_all[tk])
        y = try_all[tk]
        if head["w"] is None:
            mu = F.mean(0); sd = np.maximum(F.std(0), 0.05)
            head.update(mu=mu, sd=sd, w=np.zeros((2, F.shape[1])), b=np.zeros(2))
        A = (F - head["mu"]) / head["sd"]
        for _ in range(1200):
            p = softmax(A @ head["w"].T + head["b"])
            e = -p; e[np.arange(len(y)), y] += 1
            head["w"] += 0.5 * (e.T @ A) / len(y)
            head["b"] += 0.5 * e.mean(0)
        for j_t, tk2 in enumerate(SEQ):
            Ft, _ = feats_and_train(core, tex_all[tk2])
            R[i_t, j_t] = head_acc(Ft, tey_all[tk2], head["mu"], head["sd"],
                                   head["w"], head["b"])
    diag = np.diag(R)
    final = R[n - 1]
    forget = diag - final
    return {"R": R.tolist(),
            "acq": float(np.mean(diag)),
            "final": float(np.mean(final)),
            "meanF_honest_pp": float(np.mean(forget[:n - 1]) * 100),
            "worstF_pp": float(np.max(forget) * 100)}


def main():
    print("=" * 96)
    print("R7 ROUND 15 — is the threshold gain real across tasks, and does it reach retention?")
    print("=" * 96)

    rows = []
    for th in THRESHOLDS:
        for tk in TASKS:
            accs, ranks, rates = [], [], []
            for seed in DEV:
                core = S.SECSCore(cfg(seed, th), mode="scalar", seed=seed)
                trx, try_ = GENS[tk](seed + 200_000, N_TR)
                tex, tey = GENS[tk](seed + 300_000, N_TE)
                F, train = feats_and_train(core, trx)
                Ft, _ = feats_and_train(core, tex)
                mu, sd, w, b = fit_head(F, try_)
                accs.append(head_acc(Ft, tey, mu, sd, w, b))
                ranks.append(eff_rank(F))
                rates.append(float(train.mean()))
            rows.append({"th": th, "task": tk,
                         "acc": float(np.mean(accs)),
                         "rank": float(np.mean(ranks)),
                         "rate": float(np.mean(rates)),
                         "per_seed": [round(a, 4) for a in accs]})
            print(f"  th {th:.2f} {tk:14s} acc {rows[-1]['acc']:.4f} "
                  f"rank {rows[-1]['rank']:5.2f} rate {rows[-1]['rate']:.4f}", flush=True)

    # ---- gate: does any threshold beat production on >= 3 of 4 tasks?
    print("\n" + "=" * 96)
    print("IS THE THRESHOLD GAIN REAL? (gate: help >=3 of 4 tasks, hurt none > 3pp)")
    print("=" * 96)
    base = {r["task"]: r["acc"] for r in rows if r["th"] == 0.50}
    verdict = {}
    print(f"  {'th':>6} " + " ".join(f"{t:>14s}" for t in TASKS) + f" {'helps':>7} {'worst':>8}")
    for th in THRESHOLDS[1:]:
        deltas = {r["task"]: r["acc"] - base[r["task"]]
                  for r in rows if r["th"] == th}
        helps = sum(1 for d in deltas.values() if d > 0.01)
        worst = min(deltas.values())
        ok = bool(helps >= 3 and worst >= -0.03)
        verdict[th] = {"deltas_pp": {k: round(v * 100, 2) for k, v in deltas.items()},
                       "helps": helps, "worst_pp": round(worst * 100, 2), "passes": ok}
        print(f"  {th:6.2f} " + " ".join(f"{deltas[t]*100:+13.2f}p" for t in TASKS)
              + f" {helps}/4{'':>2} {worst*100:+7.2f}p  {'PASS' if ok else 'fail'}")

    real = [th for th, v in verdict.items() if v["passes"]]
    print(f"\n  => {'REAL' if real else 'BENCHMARK ARTEFACT'}: "
          f"{real if real else 'no threshold helps 3 of 4 tasks'}")

    # ---- retention at each threshold
    print("\n" + "=" * 96)
    print("DOES THE BETTER CODE HELP RETENTION? (the unsolved second goal)")
    print("=" * 96)
    ret = {}
    for th in THRESHOLDS:
        per = []
        for seed in DEV:
            trx_all, try_all, tex_all, tey_all = {}, {}, {}, {}
            for tk in TASKS[:3]:
                trx_all[tk], try_all[tk] = GENS[tk](seed + 200_000, N_TR)
                tex_all[tk], tey_all[tk] = GENS[tk](seed + 300_000, N_TE)
            per.append(retention(seed, th, trx_all, try_all, tex_all, tey_all))
        ret[th] = {"acq": float(np.mean([p["acq"] for p in per])),
                   "final": float(np.mean([p["final"] for p in per])),
                   "meanF_honest_pp": float(np.mean([p["meanF_honest_pp"] for p in per])),
                   "worstF_pp": float(np.mean([p["worstF_pp"] for p in per])),
                   "R_seed0": per[0]["R"]}
        r = ret[th]
        print(f"  th {th:.2f}  acq {r['acq']:.4f}  final {r['final']:.4f}  "
              f"meanF_hon {r['meanF_honest_pp']:6.2f}pp  worstF {r['worstF_pp']:6.2f}pp")

    print("\n  gates: acq>=.65 final>=.65 meanF<=3pp worstF<=5pp")
    best = min(ret, key=lambda t: ret[t]["meanF_honest_pp"])
    passes = (ret[best]["acq"] >= .65 and ret[best]["final"] >= .65
              and ret[best]["meanF_honest_pp"] <= 3 and ret[best]["worstF_pp"] <= 5)
    print(f"  best threshold for retention: {best:.2f} "
          f"(meanF {ret[best]['meanF_honest_pp']:.2f}pp)  "
          f"{'PASS' if passes else 'STILL FAILS'}")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "threshold-robustness.json").write_text(
        json.dumps({"per_task": rows, "threshold_verdict": verdict,
                    "real_thresholds": real, "retention": ret,
                    "caveat": "threshold swept as DIAGNOSTIC only; production unchanged"},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/threshold-robustness.json")


if __name__ == "__main__":
    main()
