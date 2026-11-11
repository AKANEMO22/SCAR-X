"""Pre-protocol diagnosis: is the online head underfitting or overfitting?

Stage 0 showed clip=.025 saturates on 78% of updates. Raising the clip cut
training loss from ~0.60 to ~0.37 (and to ~0.23 on one seed) while test accuracy
stayed at ~0.51. If the head were underfitting, loss and accuracy would improve
together. Loss down + accuracy flat means it is fitting the TRAINING set and not
generalising.

This script measures that directly: train accuracy vs test accuracy, and the
effect of more training data. It decides which two head conditions Stage 2 should
test, and whether the brief's screen target is reachable at all.
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


def traces(c, idx, w_in, w_rec, seqs):
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
    return f, spk


class Head:
    def __init__(self, dim, seed, lr=.08, clip=.025):
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


def trial(seed, n_train, clip, epochs=4):
    fx, fy = sim_xor(seed + 100_000, 128)
    trx, try_ = sim_xor(seed + 200_000, n_train)
    tx, ty = sim_xor(seed + 300_000, 512)
    c, idx, w_in, w_rec = weights(seed)
    trf, _ = traces(c, idx, w_in, w_rec, trx)
    tef, _ = traces(c, idx, w_in, w_rec, tx)
    h = Head(trf.shape[1], seed, clip=clip)
    for f in trf:
        h.observe(f)
    for _ in range(epochs):
        for f, y in zip(trf, try_):
            h.learn(f, int(y))
    return {"train_acc": h.acc(trf, try_), "test_acc": h.acc(tef, ty),
            "ridge_train": ridge_probe(trf, try_, trf, try_),
            "ridge_test": ridge_probe(trf, try_, tef, ty)}


def main():
    print("=" * 80)
    print("UNDERFIT or OVERFIT?  train vs test accuracy of the online head")
    print("=" * 80)
    for clip in (.025, .5):
        print(f"\n--- head clip = {clip} ---")
        print(f"  {'n_train':>8} {'train_acc':>10} {'test_acc':>9} {'gap':>8} {'ridge_test':>11}")
        for n in (256, 1024, 2048):
            rs = [trial(s, n, clip) for s in SEEDS]
            ta = np.mean([r["train_acc"] for r in rs])
            te = np.mean([r["test_acc"] for r in rs])
            rg = np.mean([r["ridge_test"] for r in rs])
            print(f"  {n:>8} {ta:10.4f} {te:9.4f} {ta-te:+8.4f} {rg:11.4f}")

    print("\n" + "=" * 80)
    print("READING")
    print("  train_acc >> test_acc  -> overfitting: the head fits noise, features")
    print("                            do not generalise, more data may help")
    print("  train_acc ~= test_acc  -> underfitting: capacity/step problem")
    print("=" * 80)


if __name__ == "__main__":
    main()
