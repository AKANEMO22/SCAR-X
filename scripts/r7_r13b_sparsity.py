"""R7 round 13b — SPARSITY, not rank: the round-12 interpretation was wrong.

Round 13 produced a result that overturns round 12's conclusion:

    arm           rank   dead   rate     acc
    frozen        4.71   26.3   0.096   0.8320
    hta           6.67   21.3   0.046   0.7539   rank +1.96 but accuracy -7.81pp
    hta_shuffled  5.50   23.3   0.113   0.6888
    hta_scale     5.48   41.0   0.013   0.9245   rank +0.77 but accuracy +9.24pp

`hta_scale` is NOT homeostasis: it applies |drive| to the threshold, so every neuron's
threshold only ever rises. Firing collapses to rate 0.013, dead units rise to 41 of 48,
and accuracy jumps to 0.9245.

So the variable that tracks accuracy is SPARSITY, not effective rank or dead-unit count.
Round 12's corr(rank, accuracy) = +0.795 was measured across a leak sweep where rank and
rate move together; it could not separate them. Round 13 separates them: rank up,
accuracy DOWN (hta); rank barely up, accuracy UP (hta_scale).

DISCRIMINATING TEST. Hold everything fixed and sweep ONLY the global threshold, which
changes sparsity without touching weights, topology, or the learning rule. If accuracy
rises monotonically as firing falls, sparsity is the lever and rank was a confound.

This is a DIAGNOSTIC. The production threshold is not changed; a separate Config is built.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.neural.core import Config, SpikingCore  # noqa: E402
from scarx.evaluation import r7_secs as S  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
THRESHOLDS = (0.50, 0.70, 1.00, 1.50, 2.50)


def cfg(seed, threshold):
    base = S.cfg(seed)
    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    fields["threshold"] = threshold
    return Config(**fields)


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


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def spike_sum(core, X):
    out, fire = [], []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
        out.append(tot)
        fire.append(core.spikes.copy())
    return np.stack(out), np.stack(fire)


def eff_rank(F):
    C = np.cov(F - F.mean(0), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def train_head(F, y, epochs=1200, lr=.5):
    mu = F.mean(0)
    sd = np.maximum(F.std(0), 0.05)
    A = (F - mu) / sd
    w = np.zeros((2, A.shape[1])); b = np.zeros(2)
    for _ in range(epochs):
        p = softmax(A @ w.T + b)
        e = -p; e[np.arange(len(y)), y] += 1
        w += lr * (e.T @ A) / len(y); b += lr * e.mean(0)
    return mu, sd, w, b


def acc(F, y, mu, sd, w, b):
    return float(np.mean(np.argmax(((F - mu) / sd) @ w.T + b, 1) == y))


def main():
    print("=" * 92)
    print("R7 ROUND 13b — sweep the threshold: is SPARSITY the lever, not rank?")
    print("=" * 92)
    print("  threshold swept as a DIAGNOSTIC only; production threshold remains 0.5\n")

    rows = []
    for th in THRESHOLDS:
        for seed in DEV:
            core = SpikingCore(cfg(seed, th))
            trx, try_ = parity3(seed + 200_000, N_TR)
            tex, tey = parity3(seed + 300_000, N_TE)
            F, fire = spike_sum(core, trx)
            Ft, _ = spike_sum(core, tex)
            ever = fire.reshape(-1, fire.shape[-1]).max(0)
            r = eff_rank(F)
            mu, sd, w, b = train_head(F, try_)
            a = acc(Ft, tey, mu, sd, w, b)
            rows.append({"threshold": th, "seed": seed, "rank": r,
                         "dead": int((ever == 0).sum()),
                         "rate": float(fire.mean()), "acc": a})
            print(f"  th {th:.2f} seed{seed}  rate {fire.mean():.4f}  "
                  f"rank {r:6.2f}  dead {int((ever == 0).sum()):2d}  acc {a:.4f}",
                  flush=True)

    print("\n" + "=" * 92)
    print("AGGREGATED BY THRESHOLD")
    print("=" * 92)
    print(f"  {'th':>6} {'rate':>8} {'rank':>8} {'dead':>6} {'acc':>8}")
    agg = []
    for th in THRESHOLDS:
        sel = [r for r in rows if r["threshold"] == th]
        row = {"th": th,
               "rate": float(np.mean([r["rate"] for r in sel])),
               "rank": float(np.mean([r["rank"] for r in sel])),
               "dead": float(np.mean([r["dead"] for r in sel])),
               "acc": float(np.mean([r["acc"] for r in sel]))}
        agg.append(row)
        print(f"  {th:6.2f} {row['rate']:8.4f} {row['rank']:8.2f} "
              f"{row['dead']:6.1f} {row['acc']:8.4f}")

    def corr(xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if xs.std() < 1e-12 or ys.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])

    rates = [r["rate"] for r in rows]
    ranks = [r["rank"] for r in rows]
    accs = [r["acc"] for r in rows]
    print("\n  correlation across all 15 (threshold, seed) cells:")
    print(f"    corr(firing rate,    accuracy) = {corr(rates, accs):+.4f}")
    print(f"    corr(effective rank, accuracy) = {corr(ranks, accs):+.4f}")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    c_rate, c_rank = corr(rates, accs), corr(ranks, accs)
    print(f"  rate vs acc {c_rate:+.3f}   |   rank vs acc {c_rank:+.3f}")
    if abs(c_rate) > abs(c_rank) + 0.1 and c_rate < -0.5:
        print("  => SPARSITY is the lever and RANK was a confound. Round 12 measured")
        print("     rank and rate between the SAME two conditions, so it could not")
        print("     separate them; this sweep can, and sparsity wins.")
        verdict = "SPARSITY_NOT_RANK"
    elif abs(c_rank) > abs(c_rate) + 0.1:
        print("  => RANK is the better predictor; round 12 stands.")
        verdict = "RANK"
    else:
        print("  => Neither separates cleanly; both are proxies for something else.")
        verdict = "AMBIGUOUS"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sparsity-vs-rank.json").write_text(
        json.dumps({"task": "parity3", "per_cell": rows, "by_threshold": agg,
                    "corr_rate_acc": c_rate, "corr_rank_acc": c_rank,
                    "verdict": verdict,
                    "caveat": "threshold swept as DIAGNOSTIC only"},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/sparsity-vs-rank.json")


if __name__ == "__main__":
    main()
