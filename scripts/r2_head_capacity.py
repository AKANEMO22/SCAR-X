"""Can the online delta-rule head fit a substrate that demonstrably carries the label?

Stage 0 + the fit diagnostic showed the head cannot fit even its TRAINING set
(0.53) on decay-0.25 features, while ridge on the same features reaches 0.70 test.
That is an underfitting/step-size failure, not overfitting.

The brief's screen target requires: online head >= 0.65 on a task where the
oracle (ridge) is >= 0.80. The spike-sum substrate has ridge ~0.97, so it provides
the headroom. This script asks the decisive question:

    on the spike-sum substrate, can an online delta-rule head reach 0.65?

If yes, the brief's experiment is viable and Stage 2 is a real test.
If no, the limitation is in the rule family and must be reported as such.

Sweeps only what is needed to answer yes/no: epochs and lr. This is pre-protocol
exploration, not the registered screen.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation.tasks21 import ridge_probe  # noqa: E402
from scarx.neural.core import Config  # noqa: E402

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
S = 1.5
SEEDS = (92011, 92017, 92023)


def cfg(seed):
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


def weights(seed):
    c = cfg(seed)
    g = np.random.default_rng(c.seed)
    idx = np.stack([g.choice(c.hidden, c.fan_in, replace=False) for _ in range(c.hidden)]).astype(np.int32)
    w_in = g.normal(.15, .45, (c.hidden, c.inputs)).astype(np.float32)
    w_rec = g.normal(0, .12 / np.sqrt(c.fan_in), (c.hidden, c.fan_in)).astype(np.float32)
    return c, idx, w_in, w_rec


def features(seqs):
    """Both substrates: decay-0.25 feature and uniform spike sum."""
    c, idx, w_in, w_rec = weights(92011)   # weights seeded once per call site below
    raise RuntimeError("use build() instead")


def build(seed, seqs, substrate):
    c, idx, w_in, w_rec = weights(seed)
    x = np.asarray(seqs, np.float32)
    b, t, _ = x.shape
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy(); f = v.copy()
    spk = np.zeros((b, t, c.hidden), np.float32)
    for s in range(t):
        v = c.leak * v + x[:, s] @ w_in.T + np.sum(w_rec[None, :, :] * z[:, idx], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        z = (v >= lim).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z
        f = c.feature_decay * f + (1 - c.feature_decay) * z
        spk[:, s] = z
    return f if substrate == "decay025" else spk.sum(1)


class Head:
    def __init__(self, dim, seed, lr, clip):
        g = np.random.default_rng(seed + 991)
        self.w = g.normal(0, .03, (2, dim)).astype(np.float32)
        self.b = np.zeros(2, np.float32)
        self.mean = np.zeros(dim, np.float64); self.m2 = np.zeros(dim, np.float64); self.n = 0
        self.lr, self.clip = lr, clip

    def observe(self, f):
        self.n += 1; d = f - self.mean; self.mean += d / self.n; self.m2 += d * (f - self.mean)

    def norm(self, f):
        s = np.maximum(np.sqrt(self.m2 / max(1, self.n - 1)), .05).astype(np.float32)
        return np.clip((f - self.mean.astype(np.float32)) / s, -5, 5)

    def probs(self, f):
        lg = self.norm(f) @ self.w.T + self.b
        lg = lg - lg.max(); e = np.exp(lg); return e / e.sum()

    def learn(self, f, y):
        p = self.probs(f); err = -p.copy(); err[int(y)] += 1
        self.w += np.clip(self.lr * err[:, None] * self.norm(f)[None, :], -self.clip, self.clip)
        self.b += np.clip(self.lr * err, -self.clip, self.clip)
        np.clip(self.w, -3, 3, out=self.w); np.clip(self.b, -3, 3, out=self.b)

    def acc(self, feats, labels):
        return float(np.mean(np.argmax(np.stack([self.norm(f) for f in feats]) @ self.w.T
                                       + self.b, 1) == labels))


def trial(seed, substrate, lr, clip, epochs, n_train=1024):
    trx, try_ = sim_xor(seed + 200_000, n_train)
    tx, ty = sim_xor(seed + 300_000, 512)
    fx, fy = sim_xor(seed + 100_000, 128)
    trf = build(seed, trx, substrate)
    tef = build(seed, tx, substrate)
    ff = build(seed, fx, substrate)
    oracle = ridge_probe(trf, try_, tef, ty)
    h = Head(trf.shape[1], seed, lr, clip)
    for f in trf:
        h.observe(f)
    for _ in range(epochs):
        for f, y in zip(trf, try_):
            h.learn(f, int(y))
    return {"oracle": oracle, "test": h.acc(tef, ty), "train": h.acc(trf, try_)}


def main():
    print("=" * 82)
    print("CAN THE ONLINE HEAD REACH 0.65 ON A SUBSTRATE THE ORACLE SOLVES?")
    print(f"3 seeds, n_train=1024, hard-threshold sampler")
    print("=" * 82)
    for substrate in ("decay025", "spikesum"):
        for lr, clip, ep in ((.08, .025, 4), (.08, .5, 4), (.5, 2.0, 20), (1.0, 5.0, 40)):
            rs = [trial(s, substrate, lr, clip, ep) for s in SEEDS]
            o = np.mean([r["oracle"] for r in rs])
            te = np.mean([r["test"] for r in rs])
            tr = np.mean([r["train"] for r in rs])
            flag = "  <-- REACHES 0.65" if te >= .65 else ""
            print(f"  {substrate:10s} lr={lr:<4} clip={clip:<4} ep={ep:<3} "
                  f"oracle={o:.4f} train={tr:.4f} test={te:.4f}{flag}")


if __name__ == "__main__":
    main()
