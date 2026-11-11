"""Search for a task that can actually carry a learning claim.

The campaign is blocked on task design, not on mechanism. The requirement, taken
from grok's own conclusion and from my gate-feasibility measurement, is:

    every first-order readout on a FROZEN hidden stays below 0.65,
    while an order-2 (product) basis is well above it.

The subtlety that makes this worth measuring rather than assuming: a spiking
neuron is NOT a linear device. Threshold + reset is itself a nonlinearity, so a
"point neuron + linear readout" may already compute an XOR through its spike
nonlinearity. If it does, a dendritic compartment is unnecessary and the whole
multicompartment proposal loses its motivation at this scale.

Candidate designs are all XOR-like in the input (order-1 blind) but differ in
whether they put the two features on the same tick or on separate ticks, because
that is what decides whether the neuron's threshold can see the conjunction.

Frozen weights only. No synapse is trained.
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


# ------------------------------------------------------------------ candidates
def sim_xor(seed, count):
    """Two channels written on the SAME tick; label = XOR of the two bits.

    Order-1 sees only the sum: (0,0)->0, (1,1)->2, (0,1)->1, (1,0)->1, so both
    classes have mean sum 1.0 and a linear readout is exactly at chance.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, count).astype(np.int32)
    for i in range(count):
        a = int(rng.integers(2))
        b = a ^ int(y[i])
        t = int(rng.integers(1, 4))          # same tick for both features
        x[i, t, 0] += S * a
        x[i, t, 1] += S * b
        x[i, 10, 7] += S                      # response gate, label-free
    return x, y


def seq_xor(seed, count):
    """Same XOR but the two features sit on DIFFERENT ticks, within one segment.

    This is the control for the neuron's temporal integration: a leaky sum still
    cannot tell which of two ticks held which bit.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, count).astype(np.int32)
    for i in range(count):
        a = int(rng.integers(2))
        b = a ^ int(y[i])
        p = int(rng.integers(1, 3))
        q = int(rng.integers(4, 7))
        x[i, p, 0] += S * a
        x[i, q, 1] += S * b
        x[i, 10, 7] += S
    return x, y


def conj_query(seed, count):
    """AND-of-two with an explicit query: label = (bit0 AND bit1) XOR query.

    Order-2 in the bit pair, order-3 only if a product-with-query is required.
    Included to see how the difficulty ladder behaves.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, count).astype(np.int32)
    for i in range(count):
        q = int(rng.integers(2))
        b0 = int(rng.integers(2))
        b1 = int(rng.integers(2))
        x[i, 2, 0] += S * b0
        x[i, 2, 1] += S * b1
        x[i, 10, 4 + q] += S
        y[i] = (b0 & b1) ^ q
    return x, y


GENERATORS = {"sim_xor": sim_xor, "seq_xor": seq_xor, "conj_query": conj_query}


# ------------------------------------------------------------------- sampling
def bern(core, seq, seed=901):
    c = core.config
    x = np.asarray(seq, np.float32)
    b, t, _ = x.shape
    rng = np.random.default_rng(seed)
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
    out = []
    for s in range(t):
        v = c.leak * v + x[:, s] @ core.w_in.T + np.sum(core.w_rec[None, :, :] * z[:, core.indices], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        p = 1 / (1 + np.exp(-np.clip((v - lim) / .4, -20, 20)))
        z = (rng.random((b, c.hidden)) < p).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z
        out.append(z.copy())
    return np.stack(out, 1)


def hard(core, seq):
    c = core.config
    x = np.asarray(seq, np.float32)
    b, t, _ = x.shape
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
    out = []
    for s in range(t):
        v = c.leak * v + x[:, s] @ core.w_in.T + np.sum(core.w_rec[None, :, :] * z[:, core.indices], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        z = (v >= lim).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z
        out.append(z.copy())
    return np.stack(out, 1)


def integ(tr, d):
    n, t, h = tr.shape
    f = np.zeros((n, h), np.float32)
    for s in range(t):
        f = d * f + (1 - d) * tr[:, s]
    return f


def best_first_order(tr, tt, fy, ty):
    cands = {"uniform": (tr.sum(1), tt.sum(1))}
    for d in DECAYS:
        cands[f"d{d}"] = (integ(tr, d), integ(tt, d))
    cands["bank5"] = (np.concatenate([integ(tr, d) for d in (.1, .25, .5, .75, .9)], 1),
                      np.concatenate([integ(tt, d) for d in (.1, .25, .5, .75, .9)], 1))
    sc = {k: ridge_probe(a, fy, b, ty) for k, (a, b) in cands.items()}
    return max(sc.values()), max(sc, key=sc.get)


def order2(tr, tt, fy, ty):
    f = np.einsum('bth,btk->bhk', tr, tr).reshape(len(tr), -1)
    g = np.einsum('bth,btk->bhk', tt, tt).reshape(len(tt), -1)
    return ridge_probe(f, fy, g, ty)


def main():
    print("=" * 80)
    print("TASK SEARCH: first-order frozen < 0.65 while order-2 is well above")
    print("=" * 80)

    for name, gen in GENERATORS.items():
        fx, fy = gen(31001, 400)
        tx, ty = gen(31901, 400)
        # raw tape sanity
        r1 = ridge_probe(fx.reshape(len(fx), -1), fy, tx.reshape(len(tx), -1), ty)
        print(f"\n### {name}")
        print(f"  raw tape order-1 : {r1:.4f}   (want ~0.50: no linear shortcut)")

        for sname, fn in (("bernoulli", bern), ("hard", hard)):
            core = PlasticCore(cfg(), "frozen")
            tr, tt = fn(core, fx), fn(core, tx)
            best, who = best_first_order(tr, tt, fy, ty)
            o2 = order2(tr, tt, fy, ty)
            gate = "QUALIFIES" if (best < .65 and o2 >= .80) else "no"
            print(f"  {sname:10s} best-1st={best:.4f} ({who:8s})  order-2={o2:.4f}  {gate}")

    print("\n" + "=" * 80)
    print("A task QUALIFIES only if best-1st < 0.65 AND order-2 >= 0.80 in BOTH samplers.")
    print("=" * 80)


if __name__ == "__main__":
    main()
