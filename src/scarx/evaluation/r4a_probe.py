"""Round 4A state capture and diagnostic probes.

Captures every representation the protocol names, all from the SAME frozen ALIF
population, so that a difference in decodability is attributable to the
representation and not to a different forward pass.

  R1  pre-query state       (spikes, voltage, adaptation, features at t=9)
  R2  full spike trajectory (information-accessibility diagnostic only)
  R3  post-query state      (state after t=10)
  R4  established final features (decay025 / spikesum / bank5)
  R5  bounded query-conditioned interaction (bilinear, fixed)

No state is retained for longer than one episode, and R2 is explicitly a
diagnostic rather than a proposed unbounded-history solution.
"""
import numpy as np

from scarx.neural.core import Config

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
BANK = (.10, .25, .50, .75, .90)
QUERY_TICK = 10


def cfg(seed):
    return Config(seed=seed, **CFG)


def weights(seed):
    c = cfg(seed)
    g = np.random.default_rng(c.seed)
    idx = np.stack([g.choice(c.hidden, c.fan_in, replace=False) for _ in range(c.hidden)]).astype(np.int32)
    w_in = g.normal(.15, .45, (c.hidden, c.inputs)).astype(np.float32)
    w_rec = g.normal(0, .12 / np.sqrt(c.fan_in), (c.hidden, c.fan_in)).astype(np.float32)
    return c, idx, w_in, w_rec


def capture(seed, seqs, query_tick=QUERY_TICK):
    """Run the frozen population and return every R1-R4 representation.

    query_tick must match the generator actually used: under a temporal shift the
    query moves, and a hardcoded tick would capture the state BEFORE the query.
    """
    c, idx, w_in, w_rec = weights(seed)
    x = np.asarray(seqs, np.float32)
    b, t, _ = x.shape
    v = np.zeros((b, c.hidden), np.float32)
    a = v.copy()
    z = v.copy()
    f = np.zeros((b, c.hidden), np.float32)
    banks = [np.zeros((b, c.hidden), np.float32) for _ in BANK]
    spk = np.zeros((b, t, c.hidden), np.float32)

    pre = {}
    post = {}
    for s in range(t):
        v = c.leak * v + x[:, s] @ w_in.T + np.sum(w_rec[None, :, :] * z[:, idx], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        z = (v >= lim).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z
        f = c.feature_decay * f + (1 - c.feature_decay) * z
        for i, d in enumerate(BANK):
            banks[i] = d * banks[i] + (1 - d) * z
        spk[:, s] = z
        if s == query_tick - 1:
            pre = {"spikes": z.copy(), "voltage": v.copy(),
                   "adaptation": a.copy(), "features": f.copy()}
        if s == query_tick:
            post = {"spikes": z.copy(), "voltage": v.copy(),
                    "adaptation": a.copy(), "features": f.copy()}

    final = {
        "decay025": f,
        "spikesum": spk.sum(1),
        "bank5": np.concatenate(banks, 1),
        "spike_traj": spk.reshape(b, -1),
    }
    return {"pre": pre, "post": post, "final": final, "spike_train": spk}


def state_vector(cap, which, where):
    """Concatenate the named sub-states into one feature matrix."""
    src = cap[where]
    return np.concatenate([src[k] for k in which], 1)


def query_cue(seqs):
    """The legal query cue taken from the INPUT, one-hot over {key0, key1}."""
    x = np.asarray(seqs, np.float32)
    q = (x[:, QUERY_TICK, 5] > .5).astype(np.float32)     # 0 -> key0, 1 -> key1
    return np.stack([1 - q, q], 1)


def bilinear_query_map(state, cue, n_proj=32, seed=1234):
    """Bounded R5: fixed random projections interacted with the query cue.

    phi = [ s ; cue ; (P s) * cue_0 ; (P s) * cue_1 ]
    with P a FIXED random projection (no training). This creates the interaction
    between query and stored state that a purely additive readout cannot form,
    at the cost of n_proj extra dimensions per query slot.
    """
    rng = np.random.default_rng(seed)
    d = state.shape[1]
    P = rng.normal(0, 1 / np.sqrt(d), (d, n_proj)).astype(np.float32)
    proj = state @ P
    return np.concatenate([state, cue, proj * cue[:, :1], proj * cue[:, 1:]], 1)


# ------------------------------------------------------------------- readers
def ridge(train_f, train_y, test_f, test_y, alpha=.01):
    """Linear diagnostic. Explicitly NOT an information ceiling."""
    mu = train_f.mean(0)
    sd = np.maximum(train_f.std(0), 1e-6)
    A = np.column_stack([(train_f - mu) / sd, np.ones(len(train_f))])
    B = np.column_stack([(test_f - mu) / sd, np.ones(len(test_f))])
    T = np.eye(2)[train_y]
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ T)
    return float(np.mean(np.argmax(B @ W, 1) == test_y))


class Head:
    """Registered R2A online delta rule, unchanged."""

    def __init__(self, dim, seed, lr=.08, clip=.025, zscore=True):
        g = np.random.default_rng(seed + 991)
        self.w = g.normal(0, .03, (2, dim)).astype(np.float32)
        self.b = np.zeros(2, np.float32)
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)
        self.n = 0
        self.lr, self.clip, self.zscore = lr, clip, zscore

    def observe(self, f):
        self.n += 1
        d = f - self.mean
        self.mean += d / self.n
        self.m2 += d * (f - self.mean)

    def basis(self, f):
        if not self.zscore:
            return np.asarray(f, np.float32)
        s = np.maximum(np.sqrt(self.m2 / max(1, self.n - 1)), .05).astype(np.float32)
        return np.clip((f - self.mean.astype(np.float32)) / s, -5, 5)

    def probs(self, f):
        lg = self.basis(f) @ self.w.T + self.b
        lg = lg - lg.max()
        e = np.exp(lg)
        return e / e.sum()

    def learn(self, f, y):
        p = self.probs(f)
        err = -p.copy()
        err[int(y)] += 1
        self.w += np.clip(self.lr * err[:, None] * self.basis(f)[None, :], -self.clip, self.clip)
        self.b += np.clip(self.lr * err, -self.clip, self.clip)
        np.clip(self.w, -3, 3, out=self.w)
        np.clip(self.b, -3, 3, out=self.b)

    def predict(self, feats):
        return np.argmax(np.stack([self.basis(f) for f in feats]) @ self.w.T + self.b, 1)

    def acc(self, feats, labels):
        return float(np.mean(self.predict(feats) == labels))


def train_head(trf, try_, epochs=1, seed=70001, **kw):
    h = Head(trf.shape[1], seed, **kw)
    for row in trf:
        h.observe(row)
    for _ in range(epochs):
        for row, y in zip(trf, try_):
            h.learn(row, int(y))
    return h
