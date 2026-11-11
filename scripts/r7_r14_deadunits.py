"""R7 round 14 — correction: my "dead units" metric was recording ONE TICK, not the train.

THE BUG I JUST FOUND
--------------------
Every script from round 12 onward computed "dead units" as:

    fire.append(core.spikes.copy())      # <-- LAST TICK ONLY
    ever = fire.reshape(-1, H).max(0)
    dead = (ever == 0).sum()

`fire` is therefore the spike vector at the FINAL tick (t=11, the readout gate), not the
spike train. The diagnostic in `local/verify_th.py` proves the contradiction directly:

    th=1.50:  fire.sum() = 0        ("no spikes at all")
              F.sum(1) > 0 for 384/384 rows   ("every sequence produced features")

Both read the same forward pass. The only consistent explanation is that `fire` captured
one tick while `F` accumulated all twelve.

CONSEQUENCE: round 12's headline "25.7 of 48 dead units" and round 13's "dead 48/48" are
NOT statements about neurons that never fire. They are statements about neurons that do
not fire **at the final tick**. Round 12's effective-rank measurement is unaffected (it
used the summed features F, which are correct), but its dead-unit claim must be retracted
and re-measured.

This script measures dead units correctly and re-runs the rank/sparsity analysis.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.neural.core import Config  # noqa: E402
from scarx.evaluation import r7_secs as S  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
THRESHOLDS = (0.50, 0.70, 1.00, 1.50)


def cfg(seed, threshold):
    base = S.cfg(seed)
    f = {k: getattr(base, k) for k in base.__dataclass_fields__}
    f["threshold"] = threshold
    return Config(**f)


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


def spike_sum(core, X):
    """Return (summed features, CORRECT full spike train of shape (n, T, H))."""
    out, train = [], []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        fired = np.zeros((12, core.config.hidden), np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
            fired[t] = core.spikes          # ALL ticks, not just the last
        out.append(tot)
        train.append(fired)
    return np.stack(out), np.stack(train)


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def eff_rank(F):
    C = np.cov(F - F.mean(0), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def fit(F, y, epochs=1200, lr=.5):
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
    print("R7 ROUND 14 — corrected dead-unit measurement")
    print("=" * 92)
    print(f"  {'th':>5} {'rate':>8} {'rank':>7} {'dead_true':>10} "
          f"{'dead_oldtick':>13} {'acc':>8}")

    rows = []
    for th in THRESHOLDS:
        for seed in DEV:
            core = S.SECSCore(cfg(seed, th), mode="scalar", seed=seed)
            trx, try_ = parity3(seed + 200_000, N_TR)
            tex, tey = parity3(seed + 300_000, N_TE)
            F, train = spike_sum(core, trx)
            Ft, _ = spike_sum(core, tex)

            ever = train.reshape(-1, train.shape[-1]).max(0)   # all 12 ticks
            dead_true = int((ever == 0).sum())
            dead_old = int((train[:, -1, :].max(0) == 0).sum())  # last tick only
            r = eff_rank(F)
            mu, sd, w, b = fit(F, try_)
            a = acc(Ft, tey, mu, sd, w, b)
            rows.append({"th": th, "seed": seed, "rank": r, "dead_true": dead_true,
                         "dead_old": dead_old, "rate": float(train.mean()), "acc": a})
            print(f"  {th:5.2f} {seed} rate {train.mean():7.4f} rank {r:6.2f} "
                  f"dead_true {dead_true:3d}  dead_lasttick {dead_old:3d}  acc {a:.4f}",
                  flush=True)

    print("\n" + "=" * 92)
    print("AGGREGATED — true dead units vs the erroneous last-tick count")
    print("=" * 92)
    print(f"  {'th':>5} {'rate':>8} {'rank':>7} {'dead_true':>10} "
          f"{'dead_lasttick':>13} {'acc':>8}")
    agg = []
    for th in THRESHOLDS:
        sel = [r for r in rows if r["th"] == th]
        row = {"th": th,
               "rate": float(np.mean([r["rate"] for r in sel])),
               "rank": float(np.mean([r["rank"] for r in sel])),
               "dead_true": float(np.mean([r["dead_true"] for r in sel])),
               "dead_old": float(np.mean([r["dead_old"] for r in sel])),
               "acc": float(np.mean([r["acc"] for r in sel]))}
        agg.append(row)
        print(f"  {th:5.2f} {row['rate']:8.4f} {row['rank']:7.2f} "
              f"{row['dead_true']:10.1f} {row['dead_old']:13.1f} {row['acc']:8.4f}")

    def corr(xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if xs.std() < 1e-12 or ys.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])

    ranks = [r["rank"] for r in rows]
    accs = [r["acc"] for r in rows]
    rates = [r["rate"] for r in rows]
    deads = [r["dead_true"] for r in rows]
    print("\n  correlations across 12 cells:")
    print(f"    corr(rank,      acc) = {corr(ranks, accs):+.4f}")
    print(f"    corr(rate,      acc) = {corr(rates, accs):+.4f}")
    print(f"    corr(dead_true, acc) = {corr(deads, accs):+.4f}")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    dt = np.mean([r["dead_true"] for r in rows if r["th"] == 0.50])
    do = np.mean([r["dead_old"] for r in rows if r["th"] == 0.50])
    print(f"  At the production threshold 0.50:")
    print(f"    TRUE dead units (never fire, any tick) : {dt:.1f} of 48")
    print(f"    what I reported in round 12           : {do:.1f} of 48")
    print(f"    => round 12's dead-unit figure was WRONG by "
          f"{abs(do-dt):.1f} units.")
    print(f"\n  That figure is retracted. Effective rank and the accuracy sweep are")
    print(f"  unaffected: both used the summed feature F, which was always correct.")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dead-units-corrected.json").write_text(
        json.dumps({"task": "parity3", "per_cell": rows, "by_threshold": agg,
                    "corr_rank_acc": corr(ranks, accs),
                    "corr_rate_acc": corr(rates, accs),
                    "corr_deadtrue_acc": corr(deads, accs),
                    "correction": "round 12 dead-unit count recorded only the last tick"},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/dead-units-corrected.json")


if __name__ == "__main__":
    main()
