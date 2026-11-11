"""Round 5B state capture and readers.

Carries over the R4A lessons explicitly:

  * `capture()` takes `query_tick` as a parameter. R4A's Stage-4 shift run failed
    because the harness pinned tick 10 while the generator moved the query to tick 11.
    That defect must not recur.
  * `Q A` here decodes the STORED VALUES (vA and vB separately and jointly), not the
    query label. R4A initially confused the two and had to retract the result.
  * Ridge is a linear diagnostic, never called an information ceiling, and never
    used to claim that a head can learn the mapping.
"""
import numpy as np

from scarx.neural.core import Config

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
BANK = (.10, .25, .50, .75, .90)
DEFAULT_QUERY_TICK = 10
N_TICKS = 12


def cfg(seed):
    return Config(seed=seed, **CFG)


def weights(seed):
    c = cfg(seed)
    g = np.random.default_rng(c.seed)
    idx = np.stack([g.choice(c.hidden, c.fan_in, replace=False)
                    for _ in range(c.hidden)]).astype(np.int32)
    w_in = g.normal(.15, .45, (c.hidden, c.inputs)).astype(np.float32)
    w_rec = g.normal(0, .12 / np.sqrt(c.fan_in), (c.hidden, c.fan_in)).astype(np.float32)
    return c, idx, w_in, w_rec


def capture(seed, seqs, query_tick=DEFAULT_QUERY_TICK):
    """Run the frozen ALIF population and return every representation.

    `query_tick` MUST match the generator actually used. Under a temporal shift the
    query moves, and a hardcoded tick would capture the state before the query.
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

    pre, post, at_query = {}, {}, {}
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
            at_query["spikes"] = z.copy()

    final = {"decay025": f, "spikesum": spk.sum(1),
             "bank5": np.concatenate(banks, 1), "spike_traj": spk.reshape(b, -1)}
    return {"pre": pre, "post": post, "final": final, "spike_train": spk}


def state_vector(cap, where, keys=("spikes", "voltage", "adaptation", "features")):
    return np.concatenate([cap[where][k] for k in keys], 1)


def ridge(train_f, train_y, test_f, test_y, alpha=.01):
    """Linear diagnostic. Explicitly NOT an information ceiling."""
    mu = train_f.mean(0)
    sd = np.maximum(train_f.std(0), 1e-6)
    A = np.column_stack([(train_f - mu) / sd, np.ones(len(train_f))])
    B = np.column_stack([(test_f - mu) / sd, np.ones(len(test_f))])
    T = np.eye(2)[train_y]
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ T)
    return float(np.mean(np.argmax(B @ W, 1) == test_y))


def ridge_predict(train_f, train_y, test_f, alpha=.01):
    mu = train_f.mean(0)
    sd = np.maximum(train_f.std(0), 1e-6)
    A = np.column_stack([(train_f - mu) / sd, np.ones(len(train_f))])
    B = np.column_stack([(test_f - mu) / sd, np.ones(len(test_f))])
    T = np.eye(2)[train_y]
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ T)
    return np.argmax(B @ W, 1)


class Head:
    """Registered local delta rule, unchanged from R2A/R4A."""

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
        self.w += np.clip(self.lr * err[:, None] * self.basis(f)[None, :],
                          -self.clip, self.clip)
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


def bilinear_query_map(state, cue, n_proj=32, seed=1234):
    """R5-style bounded interaction: fixed projection gated by the query cue."""
    rng = np.random.default_rng(seed)
    d = state.shape[1]
    P = rng.normal(0, 1 / np.sqrt(d), (d, n_proj)).astype(np.float32)
    proj = state @ P
    return np.concatenate([state, cue, proj * cue[:, :1], proj * cue[:, 1:]], 1)
