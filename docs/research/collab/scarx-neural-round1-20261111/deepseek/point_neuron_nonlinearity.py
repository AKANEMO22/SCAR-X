"""Does the point ALIF neuron already provide the nonlinearity a dendrite is meant to add?

sim_xor puts two bits on the SAME tick and labels their XOR. Order-1 on the raw
tape is 0.49 (chance), so no linear readout can do it. The question is whether a
POINT neuron (no dendritic compartment) can, through threshold-and-reset.

Result preview: under HARD threshold it reaches 0.9625; under Bernoulli it stays
at 0.52. That asymmetry is the point of this script.

Then: is there anything the threshold nonlinearity genuinely cannot do? Test
parity-of-3, which needs a higher-order interaction than a single threshold.

Frozen weights only.
"""
import itertools

import numpy as np

from scarx.evaluation.tasks21 import ridge_probe
from scarx.neural.core import Config
from scarx.neural.plasticity import PlasticCore

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
DECAYS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
S = 1.5


def cfg(seed=41001):
    return Config(seed=seed, **CFG)


def sim_xor(seed, count):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, count).astype(np.int32)
    for i in range(count):
        a = int(rng.integers(2)); b = a ^ int(y[i])
        t = int(rng.integers(1, 4))
        x[i, t, 0] += S * a
        x[i, t, 1] += S * b
        x[i, 10, 7] += S
    return x, y


def parity3(seed, count):
    """label = b0 XOR b1 XOR b2, all three on the same tick.

    A single threshold on a weighted sum cannot compute parity-3 for any weights:
    parity is not linearly separable and needs a 3rd-order term. If the point
    neuron fails here while succeeding on XOR, that locates the true limit.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = np.zeros(count, np.int32)
    for i in range(count):
        bits = [int(rng.integers(2)) for _ in range(3)]
        t = int(rng.integers(1, 4))
        for c, b in enumerate(bits):
            x[i, t, c] += S * b
        x[i, 10, 7] += S
        y[i] = bits[0] ^ bits[1] ^ bits[2]
    return x, y


def bern(core, seq, seed=901):
    c = core.config; x = np.asarray(seq, np.float32); b, t, _ = x.shape
    rng = np.random.default_rng(seed)
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy(); out = []
    for s in range(t):
        v = c.leak * v + x[:, s] @ core.w_in.T + np.sum(core.w_rec[None, :, :] * z[:, core.indices], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        p = 1 / (1 + np.exp(-np.clip((v - lim) / .4, -20, 20)))
        z = (rng.random((b, c.hidden)) < p).astype(np.float32)
        v = v - z * lim; a = c.adaptation_decay * a + z; out.append(z.copy())
    return np.stack(out, 1)


def hard(core, seq):
    c = core.config; x = np.asarray(seq, np.float32); b, t, _ = x.shape
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy(); out = []
    for s in range(t):
        v = c.leak * v + x[:, s] @ core.w_in.T + np.sum(core.w_rec[None, :, :] * z[:, core.indices], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        z = (v >= lim).astype(np.float32)
        v = v - z * lim; a = c.adaptation_decay * a + z; out.append(z.copy())
    return np.stack(out, 1)


def integ(tr, d):
    n, t, h = tr.shape
    f = np.zeros((n, h), np.float32)
    for s in range(t):
        f = d * f + (1 - d) * tr[:, s]
    return f


def best1(tr, tt, fy, ty):
    c = {"uniform": (tr.sum(1), tt.sum(1))}
    for d in DECAYS:
        c[f"d{d}"] = (integ(tr, d), integ(tt, d))
    c["bank5"] = (np.concatenate([integ(tr, d) for d in (.1, .25, .5, .75, .9)], 1),
                  np.concatenate([integ(tt, d) for d in (.1, .25, .5, .75, .9)], 1))
    sc = {k: ridge_probe(a, fy, b, ty) for k, (a, b) in c.items()}
    return max(sc.values()), max(sc, key=sc.get)


def order2(tr, tt, fy, ty):
    f = np.einsum('bth,btk->bhk', tr, tr).reshape(len(tr), -1)
    g = np.einsum('bth,btk->bhk', tt, tt).reshape(len(tt), -1)
    return ridge_probe(f, fy, g, ty)


def order3(tr, tt, fy, ty):
    """Only the triples the task actually needs, to keep the basis small."""
    def feat(s):
        cols = [s.reshape(len(s), -1)]
        for t in range(s.shape[1]):
            for a, b, c in itertools.combinations(range(8), 3):
                cols.append((s[:, t, a] * s[:, t, b] * s[:, t, c])[:, None])
        return np.concatenate(cols, 1)
    return ridge_probe(feat(tr), fy, feat(tt), ty)


def main():
    print("=" * 78)
    print("1. XOR on one tick: does a POINT neuron compute it?")
    print("   raw-tape order-1 = 0.49 (chance) -> no linear readout can")
    print("=" * 78)
    for gs in (31001, 31009):
        fx, fy = sim_xor(gs, 400); tx, ty = sim_xor(gs + 900, 400)
        for sname, fn in (("bernoulli", bern), ("hard", hard)):
            core = PlasticCore(cfg(), "frozen")
            tr, tt = fn(core, fx), fn(core, tx)
            b1, who = best1(tr, tt, fy, ty)
            o2 = order2(tr, tt, fy, ty)
            verdict = "SOLVES" if b1 >= .65 else "cannot"
            print(f"  seed{gs} {sname:10s} best-1st={b1:.4f} ({who})  order-2={o2:.4f}  -> {verdict}")

    print()
    print("=" * 78)
    print("2. Parity-of-3 on one tick: what a single threshold CANNOT do")
    print("=" * 78)
    for gs in (31001, 31009):
        fx, fy = parity3(gs, 400); tx, ty = parity3(gs + 900, 400)
        r1 = ridge_probe(fx.reshape(len(fx), -1), fy, tx.reshape(len(tx), -1), ty)
        print(f"  seed{gs} raw tape order-1 = {r1:.4f}")
        for sname, fn in (("bernoulli", bern), ("hard", hard)):
            core = PlasticCore(cfg(), "frozen")
            tr, tt = fn(core, fx), fn(core, tx)
            b1, who = best1(tr, tt, fy, ty)
            o2 = order2(tr, tt, fy, ty)
            o3 = order3(tr, tt, fy, ty)
            print(f"    {sname:10s} best-1st={b1:.4f} ({who:7s}) order-2={o2:.4f} order-3={o3:.4f}")

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("  XOR solvable by hard-threshold point neuron -> threshold IS a nonlinearity")
    print("  Parity-3 not solvable -> that is the real limit; needs 3rd-order or depth")
    print("=" * 78)


if __name__ == "__main__":
    main()
