"""Round 3A substrates and task suite.

Five fixed feature maps, all computed from the SAME frozen ALIF spike train with
bounded streaming state. Hidden stays 48; extra dimensions are feature channels,
not neurons.

  A decay025            48   production f[t] = 0.25 f + 0.75 z
  B spikesum            48   sum of spikes over the window
  C bank5              240   five decays {0.10,0.25,0.50,0.75,0.90}
  D count_plus_fast     96   48 counts + 48 fast traces (decay 0.50)
  E temporal_bins_3    144   spike counts in three windows (by fixed boundaries)

Bounded state: each substrate needs at most 5*48 floats of running state and does
not retain the spike history. `StreamSubstrate` demonstrates that a streaming
update reproduces the batch computation exactly.
"""
import numpy as np

from scarx.neural.core import Config

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
S = 1.5
BANK = (.10, .25, .50, .75, .90)
FAST_D = .50                      # candidate D's fast trace decay, fixed in advance
BIN_EDGES = (4, 8)                # candidate E: [0,4), [4,8), [8,12)
SUBSTRATES = ("decay025", "spikesum", "bank5", "count_plus_fast", "temporal_bins_3")


def cfg(seed):
    return Config(seed=seed, **CFG)


def weights(seed):
    c = cfg(seed)
    g = np.random.default_rng(c.seed)
    idx = np.stack([g.choice(c.hidden, c.fan_in, replace=False) for _ in range(c.hidden)]).astype(np.int32)
    w_in = g.normal(.15, .45, (c.hidden, c.inputs)).astype(np.float32)
    w_rec = g.normal(0, .12 / np.sqrt(c.fan_in), (c.hidden, c.fan_in)).astype(np.float32)
    return c, idx, w_in, w_rec


# ------------------------------------------------------------------ tasks
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


def xor_separated(seed, count):
    """Two bits on DIFFERENT ticks within the window: order-sensitive XOR."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, count).astype(np.int32)
    for i in range(count):
        a = int(rng.integers(2)); b = a ^ int(y[i])
        p = int(rng.integers(1, 3))
        q = int(rng.integers(4, 7))
        x[i, p, 0] += S * a
        x[i, q, 1] += S * b
        x[i, 10, 7] += S
    return x, y


def order_task(seed, count):
    """Label = which of two events came first. Needs temporal order, not content."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = np.arange(count, dtype=np.int32) % 2
    for i in range(count):
        first, second = (0, 1) if y[i] == 0 else (1, 0)
        p = int(rng.integers(1, 3))
        q = int(rng.integers(5, 8))
        x[i, p, first] += S
        x[i, q, second] += S
        x[i, 10, 7] += S
    perm = rng.permutation(count)
    return x[perm], y[perm]


def pair_binding(seed, count):
    """Key->value pairing; the bag of writes cannot answer it."""
    rng = np.random.default_rng(seed)
    y = np.arange(count, dtype=np.int32) % 2
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        v0 = int(y[i])
        values = {0: v0, 1: 1 - v0}
        order = rng.permutation(2)
        positions = (int(rng.integers(1, 3)), int(rng.integers(4, 7)))
        for position, key in zip(positions, order):
            key = int(key)
            x[i, position, key] += S
            x[i, position, 2 + values[key]] += S
    perm = rng.permutation(count)
    return x[perm], y[perm]


def strict_binding(seed, count):
    """Query at t=10 asks about data written at t in {1,2} and {4,6}."""
    rng = np.random.default_rng(seed)
    y = np.arange(count, dtype=np.int32) % 2
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        query = int(rng.integers(2))
        a = int(y[i]) ^ query
        values = (a, 1 - a)
        order = rng.permutation(2)
        positions = (int(rng.integers(1, 3)), int(rng.integers(4, 7)))
        for position, key in zip(positions, order):
            key = int(key)
            x[i, position, key] += S
            x[i, position, 2 + values[key]] += S
        x[i, 10, 4 + query] += S
        x[i, 11, 7] += S
    perm = rng.permutation(count)
    return x[perm], y[perm]


TASKS = {
    "sim_xor": sim_xor,
    "xor": xor_separated,
    "order": order_task,
    "pair_binding": pair_binding,
    "strict_binding": strict_binding,
}
TIME_DEPENDENT = ("xor", "order", "strict_binding")


# --------------------------------------------------------------- substrates
def spike_train(seed, seqs):
    c, idx, w_in, w_rec = weights(seed)
    x = np.asarray(seqs, np.float32)
    b, t, _ = x.shape
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
    out = np.zeros((b, t, c.hidden), np.float32)
    for s in range(t):
        v = c.leak * v + x[:, s] @ w_in.T + np.sum(w_rec[None, :, :] * z[:, idx], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        z = (v >= lim).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z
        out[:, s] = z
    return out


def substrate_from_train(spk, name):
    """Build a substrate from an explicit (B,T,H) spike train. Pure function."""
    b, t, h = spk.shape
    if name == "decay025":
        f = np.zeros((b, h), np.float32)
        for s in range(t):
            f = .25 * f + .75 * spk[:, s]
        return f
    if name == "spikesum":
        return spk.sum(1)
    if name == "bank5":
        outs = []
        for d in BANK:
            f = np.zeros((b, h), np.float32)
            for s in range(t):
                f = d * f + (1 - d) * spk[:, s]
            outs.append(f)
        return np.concatenate(outs, 1)
    if name == "count_plus_fast":
        counts = spk.sum(1)
        f = np.zeros((b, h), np.float32)
        for s in range(t):
            f = FAST_D * f + (1 - FAST_D) * spk[:, s]
        return np.concatenate([counts, f], 1)
    if name == "temporal_bins_3":
        e0, e1 = BIN_EDGES
        return np.concatenate([spk[:, :e0].sum(1),
                               spk[:, e0:e1].sum(1),
                               spk[:, e1:].sum(1)], 1)
    raise ValueError(name)


def substrate(seed, seqs, name):
    return substrate_from_train(spike_train(seed, seqs), name)


class StreamSubstrate:
    """Streaming form: bounded state, one tick at a time.

    Reproduces `substrate_from_train` exactly (see tests). State size is
    constant in the number of ticks, so the history is never retained.
    """

    def __init__(self, name, hidden=48):
        self.name = name
        self.h = hidden
        if name in ("decay025", "count_plus_fast"):
            self.f = np.zeros(hidden, np.float32)
        if name == "count_plus_fast":
            self.fast = np.zeros(hidden, np.float32)
            self.counts = np.zeros((1, hidden), np.float32)
        if name == "bank5":
            self.f = np.zeros((len(BANK), hidden), np.float32)
        if name in ("spikesum", "temporal_bins_3"):
            self.counts = np.zeros((3 if name == "temporal_bins_3" else 1, hidden), np.float32)
        if name == "temporal_bins_3":
            self.t = 0

    def update(self, z):
        if self.name == "decay025":
            self.f = .25 * self.f + .75 * z
        elif self.name == "count_plus_fast":
            self.counts[0] += z
            self.fast = FAST_D * self.fast + (1 - FAST_D) * z
        elif self.name == "bank5":
            for i, d in enumerate(BANK):
                self.f[i] = d * self.f[i] + (1 - d) * z
        elif self.name == "spikesum":
            self.counts[0] += z
        elif self.name == "temporal_bins_3":
            b = 0 if self.t < BIN_EDGES[0] else (1 if self.t < BIN_EDGES[1] else 2)
            self.counts[b] += z
            self.t += 1

    def value(self):
        if self.name == "decay025":
            return self.f.copy()
        if self.name == "count_plus_fast":
            # batch order is [counts ; fast], and counts is the running sum,
            # not the decayed trace -- keep the two separate
            return np.concatenate([self.counts[0], self.fast])
        if self.name == "bank5":
            return self.f.reshape(-1).copy()
        if self.name == "spikesum":
            return self.counts[0].copy()
        if self.name == "temporal_bins_3":
            return self.counts.reshape(-1).copy()
        raise ValueError(self.name)

    def nstate(self):
        return int(sum(getattr(self, a).size for a in ("f", "fast", "counts")
                       if hasattr(self, a)))


def dims(name):
    return {"decay025": 48, "spikesum": 48, "bank5": 240,
            "count_plus_fast": 96, "temporal_bins_3": 144}[name]
