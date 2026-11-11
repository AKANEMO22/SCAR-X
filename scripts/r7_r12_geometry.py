"""R7 round 12 — FEATURE GEOMETRY: the one hypothesis from the brief never measured.

Round 11 produced a result that needs explaining: with the EXACT trace, shuffled credit
(no direction at all) scored BETTER than aligned credit at every eta:

    eta 0.0004   exact -2.60pp    shuffled +0.78pp

If the credit direction mattered, permuting it would destroy the effect. It does not. The
natural reading is that the features lack the independent directions for any credit
signal to act on - which is exactly the brief's "sai hinh hoc cua feature" (wrong feature
geometry) hypothesis, and I have never measured it.

This round measures the geometry of the frozen representation directly:

  effective rank      participation ratio of the covariance eigenvalues,
                      (sum lambda)^2 / sum(lambda^2). 48 dims but how many real ones?
  condition number    lambda_max / lambda_min
  mean |correlation|  how redundant the 48 neurons are
  dead units          fraction of neurons that never fire
  feature scale       spread across the population

Then it asks whether geometry predicts decodability: if effective rank is far below what
the task needs, no credit rule can help because there are not enough independent axes.

DIAGNOSTIC ONLY. No training, no candidate.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

DEV = (140011, 140017, 140023)
N = 512


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


def features(core, X, kind="spike"):
    out, fire = [], []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
        out.append(tot)
        fire.append(core.spikes.copy())
    F = np.stack(out)
    return F, np.stack(fire)


def geometry(F, name):
    Fc = F - F.mean(0)
    C = np.cov(Fc, rowvar=False)
    ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 1e-12, None)
    eff_rank = float(ev.sum() ** 2 / (ev ** 2).sum())
    cond = float(ev.max() / ev.min())
    # mean absolute off-diagonal correlation
    d = np.sqrt(np.diag(C))
    R = C / np.outer(np.maximum(d, 1e-12), np.maximum(d, 1e-12))
    off = R[~np.eye(R.shape[0], dtype=bool)]
    return {"name": name, "dims": F.shape[1],
            "effective_rank": round(eff_rank, 2),
            "rank_fraction": round(eff_rank / F.shape[1], 3),
            "condition_number": float(f"{cond:.3e}"),
            "mean_abs_corr": round(float(np.abs(off).mean()), 4),
            "scale_mean": round(float(F.mean()), 4),
            "scale_std": round(float(F.std()), 4)}


def main():
    print("=" * 88)
    print("R7 ROUND 12 — the geometry of the frozen representation")
    print("=" * 88)

    rows = []
    for seed in DEV:
        c = S.cfg(seed)
        core = S.SECSCore(c, mode="scalar", seed=seed)
        trx, try_ = parity3(seed + 200_000, N)
        F, fire = features(core, trx)

        g = geometry(F, "spike_sum")
        # dead units: neurons that never fire in any episode
        ever = fire.reshape(-1, fire.shape[-1]).max(0)
        g["dead_units"] = int((ever == 0).sum())
        g["firing_rate"] = round(float(fire.mean()), 4)
        g["seed"] = seed
        rows.append(g)
        print(f"  seed{seed}  eff_rank {g['effective_rank']:5.2f}/{g['dims']} "
              f"({g['rank_fraction']*100:.1f}%)  |corr| {g['mean_abs_corr']:.4f}  "
              f"cond {g['condition_number']:.2e}  dead {g['dead_units']}/{g['dims']}  "
              f"rate {g['firing_rate']:.4f}")

    print("\n  what the task needs:")
    print("    parity3 is a 3-bit XOR. A linear readout over k independent")
    print("    directions can express up to 2^k patterns; it needs enough")
    print("    effective rank to place three interacting bits and their")
    print("    conditional structure. 3 independent axes is the absolute floor,")
    print("    and in practice many more are required to hold the interactions.")

    eff = np.mean([r["effective_rank"] for r in rows])
    corr = np.mean([r["mean_abs_corr"] for r in rows])
    dead = np.mean([r["dead_units"] for r in rows])
    print(f"\n  measured across {len(DEV)} seeds:")
    print(f"    effective rank   {eff:.2f} of 48  ({eff/48*100:.1f}%)")
    print(f"    mean |corr|      {corr:.4f}")
    print(f"    dead units       {dead:.1f} of 48")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if eff < 12:
        print(f"  The representation is DEGENERATE: {eff:.1f} effective dimensions")
        print(f"  out of 48 available. Mean pairwise |correlation| {corr:.3f}.")
        print(f"  A credit signal routes INTO these axes, so if only {eff:.0f} exist,")
        print(f"  no routing scheme can create independent features the network does")
        print(f"  not have. This is consistent with round 11's finding that shuffled")
        print(f"  credit matched aligned credit: with few real axes, direction is")
        print(f"  irrelevant.")
    else:
        print(f"  Effective rank {eff:.1f}/48 is reasonable. Geometry is NOT the")
        print(f"  bottleneck and round 11's shuffled result needs another explanation.")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature-geometry.json").write_text(
        json.dumps({"task": "parity3", "per_seed": rows,
                    "mean_effective_rank": eff, "mean_abs_corr": corr,
                    "caveat": "spike_sum channel, frozen population, diagnostic only"},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: artifacts/deepseek-r7/feature-geometry.json")


if __name__ == "__main__":
    main()
