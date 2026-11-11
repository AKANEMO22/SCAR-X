"""Stage A/B core: an ALIF population with a conditionable delta-rule head.

RESEARCH ONLY. Not imported by serving code. No autograd, BPTT, RTRL.

Why the head needs conditioning
------------------------------
`tasks21.ridge_probe` z-scores each feature dimension before fitting. The campaign's
delta-rule head multiplies *raw amplitude*. On 48 spike features dominated by a few
high-variance units, the discriminative low-variance dimensions are ignored, so the
head sits at chance while an offline ridge on the SAME features reads 0.70.

Two candidate fixes, both readout-only (the neuron is untouched):
  zscore : running per-feature mean/var, normalise before the update
  rms    : running per-feature RMS, scale-invariant update

Signal arms
-----------
  scalar : clip(-loss - baseline) * elig            (the campaign's rule)
  prop30 : (W_out.T @ error)_j * elig_j  per neuron (proposal 30 boundary)
  shuffle: prop30 with the per-neuron factor permuted across neurons
  zero   : no hidden update
  frozen : hidden never updated
"""
import numpy as np

from scarx.neural.core import Config

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
S = 1.5


def make_config(seed, lr=0.05, hidden_lr=0.002):
    return Config(seed=seed, readout_lr=lr, hidden_lr=hidden_lr,
                  anchor_strength=0., **CFG)


def sim_xor(seed, count):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, count).astype(np.int32)
    for i in range(count):
        a = int(rng.integers(2))
        b = a ^ int(y[i])
        t = int(rng.integers(1, 4))
        x[i, t, 0] += S * a
        x[i, t, 1] += S * b
        x[i, 10, 7] += S
    return x, y


class ConditionedCore:
    """ALIF population + delta-rule head with optional feature conditioning."""

    def __init__(self, seed, head="raw", signal="scalar", lr=0.05, mode="hard"):
        if head not in ("raw", "zscore", "rms"):
            raise ValueError("head")
        if signal not in ("scalar", "prop30", "shuffle", "zero", "frozen"):
            raise ValueError("signal")
        if mode not in ("hard", "bernoulli"):
            raise ValueError("mode")
        self.c = make_config(seed, lr)
        self.head, self.signal, self.mode = head, signal, mode

        g = np.random.default_rng(self.c.seed)
        h, d, k, o = self.c.hidden, self.c.inputs, self.c.fan_in, self.c.outputs
        self.indices = np.stack([g.choice(h, k, replace=False) for _ in range(h)]).astype(np.int32)
        self.w_in = g.normal(.15, .45, (h, d)).astype(np.float32)
        self.w_rec = g.normal(0, .12 / np.sqrt(k), (h, k)).astype(np.float32)
        self.w_out = g.normal(0, .03, (o, h)).astype(np.float32)
        self.bias = np.zeros(o, np.float32)
        self.stable_in, self.stable_rec = self.w_in.copy(), self.w_rec.copy()

        self.rng = np.random.default_rng(self.c.seed + 7_100_000)
        self.sig_rng = np.random.default_rng(self.c.seed + 7_200_000)
        # running feature statistics, training data only
        self.feat_mean = np.zeros(h, np.float32)
        self.feat_m2 = np.ones(h, np.float32)
        self.feat_rms = np.ones(h, np.float32)
        self.stats_n = 0

        self.elig_in = np.zeros((self.c.plastic_rows, self.c.inputs), np.float32)
        self.elig_rec = np.zeros((self.c.plastic_rows, self.c.fan_in), np.float32)
        self.rows = np.arange(self.c.plastic_rows, dtype=np.int32)
        self.reward_mean = 0.0
        self.updates = 0
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self):
        c = self.c
        self.v = np.zeros(c.hidden, np.float32)
        self.a = np.zeros(c.hidden, np.float32)
        self.z = np.zeros(c.hidden, np.float32)
        self.f = np.zeros(c.hidden, np.float32)
        self.vpre = np.zeros(c.inputs + c.hidden, np.float32)
        self.elig_in[:] = 0
        self.elig_rec[:] = 0

    # ------------------------------------------------------- conditioning
    def normalised(self, f):
        if self.head == "raw":
            return f
        if self.head == "zscore":
            scale = np.sqrt(self.feat_m2 / max(1, self.stats_n - 1)) if self.stats_n > 1 else np.ones_like(f)
            scale = np.maximum(scale, .05)
            return (f - self.feat_mean) / scale
        # rms
        return f / np.maximum(self.feat_rms, .05)

    def observe(self, f):
        """Update running statistics from a TRAINING feature only."""
        self.stats_n += 1
        n = self.stats_n
        delta = f - self.feat_mean
        self.feat_mean += delta / n
        self.feat_m2 += delta * (f - self.feat_mean)
        self.feat_rms = np.sqrt((self.feat_rms ** 2) * ((n - 1) / n) + (f ** 2) / n)

    # ------------------------------------------------------------- forward
    def step(self, x, track=False):
        c = self.c
        old = self.z.copy()
        self.v = c.leak * self.v + self.w_in @ x + np.sum(self.w_rec * old[self.indices], axis=1)
        lim = c.threshold + c.adaptation_strength * self.a
        p = 1 / (1 + np.exp(-np.clip((self.v - lim) / .4, -20, 20)))
        if self.mode == "hard":
            self.z = (self.v >= lim).astype(np.float32)
        else:
            self.z = (self.rng.random(c.hidden) < p).astype(np.float32)
        self.v -= self.z * lim
        self.a = c.adaptation_decay * self.a + self.z
        self.f = c.feature_decay * self.f + (1 - c.feature_decay) * self.z
        if track:
            r = self.rows
            self.vpre *= c.leak
            self.vpre[:c.inputs] += x
            self.vpre[c.inputs:] += old
            score = (self.z[r] - p[r]) / .4
            self.elig_in += score[:, None] * self.vpre[None, :c.inputs]
            self.elig_rec += score[:, None] * self.vpre[c.inputs:][self.indices[r]]
            np.clip(self.elig_in, -10, 10, out=self.elig_in)
            np.clip(self.elig_rec, -10, 10, out=self.elig_rec)
        return self.probs()

    def probs(self):
        lg = self.w_out @ self.normalised(self.f) + self.bias
        lg = lg - lg.max()
        e = np.exp(lg)
        return e / e.sum()

    # ------------------------------------------------------------ learning
    def learn(self, target):
        c = self.c
        basis = self.normalised(self.f)
        p = np.asarray(self.probs(), np.float32)
        loss = -float(np.log(max(float(p[target]), 1e-12)))
        error = -p.copy()
        error[target] += 1

        self.observe(self.f)

        # readout: local delta rule on the (possibly normalised) basis
        self.w_out += np.clip(c.readout_lr * error[:, None] * basis[None, :],
                              -c.update_clip, c.update_clip)
        self.bias += np.clip(c.readout_lr * error, -c.update_clip, c.update_clip)
        np.clip(self.w_out, -c.weight_clip, c.weight_clip, out=self.w_out)

        adv = float(np.clip(-loss - self.reward_mean, -2, 2))
        self.reward_mean = .95 * self.reward_mean + .05 * (-loss)

        if self.signal == "frozen" or self.signal == "zero":
            self.updates += 1
            return loss, adv

        if self.signal == "scalar":
            per_neuron = np.full(c.hidden, adv, np.float32)
        else:
            # proposal 30: per-neuron readout-error factor, W_out.T @ error
            per_neuron = (self.w_out.T @ error)[: c.hidden].astype(np.float32)
            if self.signal == "shuffle":
                per_neuron = per_neuron[self.sig_rng.permutation(c.hidden)]

        r = self.rows
        for key, tr in (("in", self.elig_in), ("rec", self.elig_rec)):
            w = getattr(self, "w_" + key)
            scale = per_neuron[r][:, None]
            w[r] += np.clip(c.hidden_lr * scale * tr, -c.update_clip, c.update_clip)
            np.clip(w, -c.weight_clip, c.weight_clip, out=w)
        self.updates += 1
        return loss, adv

    # ---------------------------------------------------------- evaluation
    def run(self, seq, track):
        self.reset()
        for x in seq:
            self.step(x, track)

    def frozen_features(self, seqs, seed=901):
        """Stateless features with common random numbers. No state mutation."""
        c = self.c
        x = np.asarray(seqs, np.float32)
        b, t, _ = x.shape
        v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy(); f = v.copy()
        rng = np.random.default_rng(seed)
        for s in range(t):
            v = c.leak * v + x[:, s] @ self.w_in.T + np.sum(self.w_rec[None, :, :] * z[:, self.indices], axis=2)
            lim = c.threshold + c.adaptation_strength * a
            p = 1 / (1 + np.exp(-np.clip((v - lim) / .4, -20, 20)))
            if self.mode == "hard":
                z = (v >= lim).astype(np.float32)
            else:
                z = (rng.random((b, c.hidden)) < p).astype(np.float32)
            v = v - z * lim
            a = c.adaptation_decay * a + z
            f = c.feature_decay * f + (1 - c.feature_decay) * z
        return f

    def head_accuracy(self, seqs, labels):
        f = self.frozen_features(seqs)
        # must apply the same conditioning the head was trained under
        basis = np.stack([self.normalised(row) for row in f])
        return float(np.mean(np.argmax(basis @ self.w_out.T + self.bias, 1) == labels))

    def arrays(self):
        return {"w_in": self.w_in, "w_rec": self.w_rec, "w_out": self.w_out,
                "bias": self.bias, "v": self.v, "a": self.a, "f": self.f,
                "elig_in": self.elig_in, "elig_rec": self.elig_rec,
                "feat_mean": self.feat_mean, "feat_m2": self.feat_m2,
                "feat_rms": self.feat_rms,
                "reward_mean": np.array([self.reward_mean], np.float32)}
