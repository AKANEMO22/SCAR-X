"""Why did normalisation not close the gap?

Hypothesis: the head sees a 48-dim vector that is effectively BINARY and very
sparse (feature_decay 0.25 keeps almost nothing, so f is near 0 or near 1). A
delta rule on a sparse binary code updates only the units that fired, and with
~2-3 spikes per example over 48 units, most examples barely move the head. Ridge
does not care: it inverts the covariance and can weight rare-but-informative dims.

If true, the fix is not normalisation but a richer readout basis: more ticks of
information (spike sum over the segment) or a longer decay.

Measure: feature sparsity and how many units move per update.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation.conditioned_head import ConditionedCore, sim_xor
from scarx.evaluation.tasks21 import ridge_probe


def main():
    seed = 71001
    fx, fy = sim_xor(seed + 100_000, 128)
    tx, ty = sim_xor(seed + 300_000, 300)

    m = ConditionedCore(seed, head="zscore", signal="frozen")
    f = m.frozen_features(tx)
    print("=" * 72)
    print("FEATURE GEOMETRY of the readout substrate (decay 0.25)")
    print("=" * 72)
    print(f"  shape                 : {f.shape}")
    print(f"  mean value            : {f.mean():.4f}")
    print(f"  fraction == 0         : {(f == 0).mean():.4f}")
    print(f"  fraction > 0.5        : {(f > .5).mean():.4f}")
    print(f"  fraction > 0.9        : {(f > .9).mean():.4f}")
    nz = (f > .05).sum(1)
    print(f"  active units per example: mean {nz.mean():.1f} / 48 "
          f"(min {nz.min()}, max {nz.max()})")
    print(f"  per-dim std: mean {f.std(0).mean():.4f}  max {f.std(0).max():.4f}")

    print()
    print("=" * 72)
    print("WHICH BASIS CAN A LINEAR READER ACTUALLY USE?")
    print("=" * 72)
    ftx = m.frozen_features(fx)
    cands = {}
    # production substrate
    cands["f, decay 0.25 (production)"] = (ftx, f)
    # spike sum over all ticks: needs the spike train, not f
    def train_sum(seq):
        c = m.c
        x = np.asarray(seq, np.float32)
        b, t, _ = x.shape
        v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
        tot = np.zeros((b, c.hidden), np.float32)
        for s in range(t):
            v = c.leak * v + x[:, s] @ m.w_in.T + np.sum(m.w_rec[None, :, :] * z[:, m.indices], axis=2)
            lim = c.threshold + c.adaptation_strength * a
            z = (v >= lim).astype(np.float32)
            v = v - z * lim
            a = c.adaptation_decay * a + z
            tot += z
        return tot
    cands["spike SUM all ticks"] = (train_sum(fx), train_sum(tx))
    # multi-decay bank
    def bank(seq, decays=(.1, .25, .5, .75, .9)):
        c = m.c
        x = np.asarray(seq, np.float32)
        b, t, _ = x.shape
        v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
        fs = [np.zeros((b, c.hidden), np.float32) for _ in decays]
        for s in range(t):
            v = c.leak * v + x[:, s] @ m.w_in.T + np.sum(m.w_rec[None, :, :] * z[:, m.indices], axis=2)
            lim = c.threshold + c.adaptation_strength * a
            z = (v >= lim).astype(np.float32)
            v = v - z * lim
            a = c.adaptation_decay * a + z
            for i, d in enumerate(decays):
                fs[i] = d * fs[i] + (1 - d) * z
        return np.concatenate(fs, 1)
    cands["5-decay bank"] = (bank(fx), bank(tx))

    for name, (a, b) in cands.items():
        acc = ridge_probe(a, fy, b, ty)
        sparsity = float((b == 0).mean())
        print(f"  {name:30s} ridge={acc:.4f}  frac_zero={sparsity:.3f}")

    print()
    print("=" * 72)
    print("CONCLUSION")
    print("  If 'spike SUM' > 'decay 0.25' for ridge, the substrate itself is the")
    print("  limiter, and head normalisation cannot fix it. That would move the")
    print("  blocker from 'head scale' to 'feature decay', i.e. back to the")
    print("  temporal-memory finding - but now with a capable reader to show it.")
    print("=" * 72)


if __name__ == "__main__":
    main()
