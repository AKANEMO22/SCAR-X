"""R7 round 12b — the geometry prediction: does rank BOUND what any readout can do?

Round 12 measured the frozen representation at effective rank 4.60 of 48 (9.6%), with
25.7 of 48 neurons never firing and mean |correlation| 0.226.

That suggests a specific, falsifiable claim that explains eleven rounds of failure:

    A representation with effective rank r can support at most r independent linear
    features. If r is ~5, then perturbing hidden weights - however correctly - moves the
    population inside a 5-dimensional subspace, and the credit direction mostly trades one
    axis for another rather than adding capacity. That is why `exact` (aligned) and
    `exact_shuffled` (random) both land at the same place: both change which 5 axes exist,
    neither increases how many.

DISCRIMINATING TEST. If rank is the binding variable, then rank should PREDICT decodability
across a range of seed/parameter conditions far better than anything else measured. If
rank does not predict accuracy while some other quantity does, the claim is wrong.

I sweep the leak constant (diagnostic only - production leak is NOT changed) which is known
to control how much temporal structure each neuron integrates, and therefore how many
distinct states the population visits. For each leak I measure:
    effective rank, dead units, mean |corr|, and linear decodability of parity3

If rank and accuracy move together across the sweep, geometry is causal.
If accuracy moves while rank stays flat (or vice versa), it is not.

Production parameters are NOT modified. `r7_secs.cfg_with_leak` builds a separate Config
for the diagnostic only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.neural.core import Config  # noqa: E402
from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
LEAKS = (0.50, 0.65, 0.80, 0.90)


def cfg_leak(seed, leak):
    base = S.cfg(seed)
    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    fields["leak"] = leak
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
    mu = F.mean(0); sd = np.maximum(F.std(0), 1e-6)
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
    print("R7 ROUND 12b — does effective rank PREDICT decodability?")
    print("=" * 92)
    print("  leak swept as a DIAGNOSTIC only; production leak remains 0.65\n")

    rows = []
    for leak in LEAKS:
        for seed in SEEDS:
            c = cfg_leak(seed, leak)
            core = S.SECSCore(c, mode="scalar", seed=seed)
            trx, try_ = parity3(seed + 200_000, N_TR)
            tex, tey = parity3(seed + 300_000, N_TE)
            F, fire = spike_sum(core, trx)
            Ft, _ = spike_sum(core, tex)

            r = eff_rank(F)
            ever = fire.reshape(-1, fire.shape[-1]).max(0)
            dead = int((ever == 0).sum())
            Fc = F - F.mean(0)
            C = np.cov(Fc, rowvar=False)
            d = np.sqrt(np.diag(C))
            R = C / np.outer(np.maximum(d, 1e-12), np.maximum(d, 1e-12))
            mcorr = float(np.abs(R[~np.eye(R.shape[0], dtype=bool)]).mean())

            mu, sd, w, b = train_head(F, try_)
            a = acc(Ft, tey, mu, sd, w, b)
            rows.append({"leak": leak, "seed": seed, "rank": r, "dead": dead,
                         "mcorr": mcorr, "acc": a,
                         "rate": float(fire.mean())})
            print(f"  leak {leak:.2f} seed{seed}  rank {r:6.2f}  dead {dead:2d}  "
                  f"|corr| {mcorr:.3f}  rate {fire.mean():.3f}  acc {a:.4f}",
                  flush=True)

    print("\n" + "=" * 92)
    print("AGGREGATED BY LEAK")
    print("=" * 92)
    print(f"  {'leak':>6} {'rank':>8} {'dead':>6} {'|corr|':>8} {'rate':>7} {'acc':>8}")
    agg = []
    for leak in LEAKS:
        sel = [r for r in rows if r["leak"] == leak]
        row = {"leak": leak,
               "rank": float(np.mean([r["rank"] for r in sel])),
               "dead": float(np.mean([r["dead"] for r in sel])),
               "mcorr": float(np.mean([r["mcorr"] for r in sel])),
               "rate": float(np.mean([r["rate"] for r in sel])),
               "acc": float(np.mean([r["acc"] for r in sel]))}
        agg.append(row)
        print(f"  {row['leak']:6.2f} {row['rank']:8.2f} {row['dead']:6.1f} "
              f"{row['mcorr']:8.3f} {row['rate']:7.3f} {row['acc']:8.4f}")

    def corr(xs, ys):
        xs, ys = np.asarray(xs), np.asarray(ys)
        if xs.std() < 1e-12 or ys.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])

    ranks = [r["rank"] for r in rows]
    accs = [r["acc"] for r in rows]
    accs_rank = [r["acc"] for r in rows]
    rates = [r["rate"] for r in rows]
    print("\n  correlation across all 12 (leak, seed) cells:")
    print(f"    corr(effective rank, accuracy) = {corr(ranks, accs):+.4f}")
    print(f"    corr(firing rate,    accuracy) = {corr(rates, accs_rank):+.4f}")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    c_rank = corr(ranks, accs)
    if c_rank > 0.6:
        print(f"  Effective rank PREDICTS accuracy (corr {c_rank:+.3f}).")
        print(f"  Geometry is causal: the representation has too few independent")
        print(f"  axes, and no credit rule can manufacture axes the population")
        print(f"  does not visit. This is the bottleneck rounds 1-11 kept hitting")
        print(f"  from other directions.")
    elif c_rank < 0.3:
        print(f"  Effective rank does NOT predict accuracy (corr {c_rank:+.3f}).")
        print(f"  The round-12 geometry observation is descriptive, not causal, and")
        print(f"  I will not claim it as the bottleneck.")
    else:
        print(f"  Partial association (corr {c_rank:+.3f}). Not decisive on its own.")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "geometry-vs-accuracy.json").write_text(
        json.dumps({"task": "parity3", "per_cell": rows, "by_leak": agg,
                    "corr_rank_acc": c_rank,
                    "corr_rate_acc": corr(rates, accs_rank),
                    "caveat": "leak swept as DIAGNOSTIC only; production leak unchanged"},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/geometry-vs-accuracy.json")


if __name__ == "__main__":
    main()
