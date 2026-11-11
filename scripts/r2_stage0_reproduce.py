"""Stage 0 — reproduce the offline-vs-online contrast, and test the clip hypothesis.

Targets from the brief (grok's runs, hard-threshold sampler, sim_xor):
    frozen spike-sum ridge   ~0.9701
    ridge on decay 0.25      ~0.6953
    online frozen head       ~0.5039
    online head + hidden lr  ~0.5143

This script does NOT reuse grok's code. It reimplements the same pieces from the
repository primitives so that agreement is independent:
  - ALIF population: SpikingCore.step equations, hard threshold
  - feature decay 0.25 (production)
  - head: delta rule with z-scored features, clipped update

Then it isolates one variable: the ratio clip/lr in the head update.

Suspicion under test: in `ConditionedReadout.learn` the update uses lr=0.08 with
np.clip(..., -0.025, 0.025). Since 0.08 * |error| * |f| is almost always larger
than 0.025 for z-scored f, the clip saturates on nearly every update, so the head
moves by a fixed step regardless of gradient magnitude. That would make the head
incapable of learning a large-margin offset even when the information is present.

Everything is frozen weights except the head. No synapse is trained.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation.tasks21 import ridge_probe  # noqa: E402
from scarx.neural.core import Config  # noqa: E402

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
S = 1.5
GROK_SEEDS = (92011, 92017, 92023)


def cfg(seed):
    return Config(seed=seed, **CFG)


def sim_xor(seed, count):
    """Same construction as the round-1 deepseek generator."""
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


def weights(seed):
    c = cfg(seed)
    g = np.random.default_rng(c.seed)
    idx = np.stack([g.choice(c.hidden, c.fan_in, replace=False) for _ in range(c.hidden)]).astype(np.int32)
    w_in = g.normal(.15, .45, (c.hidden, c.inputs)).astype(np.float32)
    w_rec = g.normal(0, .12 / np.sqrt(c.fan_in), (c.hidden, c.fan_in)).astype(np.float32)
    return c, idx, w_in, w_rec


def traces(c, idx, w_in, w_rec, seqs):
    """Hard-threshold spike train and production decay-0.25 feature. No state carried."""
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
    """Delta rule with z-scored features. Hyperparameters are explicit."""

    def __init__(self, dim, seed, lr=.08, clip=.025, wmax=3.0, mode="normalized"):
        g = np.random.default_rng(seed + 991)
        self.w = g.normal(0, .03, (2, dim)).astype(np.float32)
        self.b = np.zeros(2, np.float32)
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)
        self.n = 0
        self.lr, self.clip, self.wmax, self.mode = lr, clip, wmax, mode
        self.saturated = 0          # count of updates where the clip bound was hit
        self.total = 0

    def observe(self, f):
        self.n += 1
        d = f - self.mean
        self.mean += d / self.n
        self.m2 += d * (f - self.mean)

    def norm(self, f):
        if self.mode == "raw":
            return f.astype(np.float32)
        s = np.maximum(np.sqrt(self.m2 / max(1, self.n - 1)), .05).astype(np.float32)
        return np.clip((f - self.mean.astype(np.float32)) / s, -5, 5)

    def probs(self, f):
        lg = self.norm(f) @ self.w.T + self.b
        lg = lg - lg.max()
        e = np.exp(lg)
        return e / e.sum()

    def learn(self, f, y):
        p = self.probs(f)
        err = -p.copy(); err[int(y)] += 1
        basis = self.norm(f)
        delta = self.lr * err[:, None] * basis[None, :]
        self.saturated += int(np.any(np.abs(delta) >= self.clip))
        self.total += 1
        self.w += np.clip(delta, -self.clip, self.clip)
        self.b += np.clip(self.lr * err, -self.clip, self.clip)
        np.clip(self.w, -self.wmax, self.wmax, out=self.w)
        np.clip(self.b, -self.wmax, self.wmax, out=self.b)

    def acc(self, feats, labels):
        return float(np.mean(np.argmax(np.stack([self.norm(f) for f in feats]) @ self.w.T
                                       + self.b, 1) == labels))


def run(seed, train, train_y, test, test_y, lr, clip, mode="normalized", epochs=4):
    c, idx, w_in, w_rec = weights(seed)
    trf, _ = traces(c, idx, w_in, w_rec, train)
    tef, _ = traces(c, idx, w_in, w_rec, test)
    h = Head(trf.shape[1], seed, lr=lr, clip=clip, mode=mode)
    for f in trf:
        h.observe(f)                     # calibration from TRAINING features only
    losses = []
    for _ in range(epochs):
        for f, y in zip(trf, train_y):
            h.learn(f, int(y))
            p = h.probs(f)[int(y)]
            losses.append(-float(np.log(max(p, 1e-12))))
    return {"acc": h.acc(tef, test_y), "loss": float(np.mean(losses)),
            "sat_frac": h.saturated / max(1, h.total), "n_updates": h.total}


def main():
    print("=" * 78)
    print("STAGE 0 — independent reproduction of the offline-vs-online contrast")
    print("hard-threshold sampler, feature decay 0.25, frozen hidden")
    print("=" * 78)

    rows = []
    for seed in GROK_SEEDS:
        fx, fy = sim_xor(seed + 100_000, 128)
        trx, try_ = sim_xor(seed + 200_000, 256)
        tx, ty = sim_xor(seed + 300_000, 512)

        c, idx, w_in, w_rec = weights(seed)
        trf, trs = traces(c, idx, w_in, w_rec, trx)
        tef, tes = traces(c, idx, w_in, w_rec, tx)
        ff, _ = traces(c, idx, w_in, w_rec, fx)

        r_decay = ridge_probe(trf, try_, tef, ty)
        r_sum = ridge_probe(trs.sum(1), try_, tes.sum(1), ty)

        # grok's head configuration as published: lr .08, clip .025
        g = run(seed, trx, try_, tx, ty, lr=.08, clip=.025)
        # same head, clip no longer below lr
        fixed = run(seed, trx, try_, tx, ty, lr=.08, clip=.5)

        rows.append({"seed": seed, "ridge_decay025": r_decay, "ridge_spikesum": r_sum,
                     "online_published_clip": g["acc"], "online_clip_raised": fixed["acc"],
                     "sat_frac_published": g["sat_frac"],
                     "loss_published": g["loss"], "loss_raised": fixed["loss"]})
        print(f"\n seed {seed}")
        print(f"   ridge decay 0.25        {r_decay:.4f}")
        print(f"   ridge spike sum         {r_sum:.4f}")
        print(f"   online head clip=.025   {g['acc']:.4f}   "
              f"saturated updates {g['sat_frac']*100:.1f}%  loss {g['loss']:.4f}")
        print(f"   online head clip=0.50   {fixed['acc']:.4f}   "
              f"loss {fixed['loss']:.4f}")

    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], float)}
    print("\n" + "=" * 78)
    print("MEANS over the three grok seeds")
    print("=" * 78)
    print(f"  ridge decay 0.25          {m['ridge_decay025']:.4f}   (brief: ~0.6953)")
    print(f"  ridge spike sum           {m['ridge_spikesum']:.4f}   (brief: ~0.9701)")
    print(f"  online head clip=.025     {m['online_published_clip']:.4f}   (brief: ~0.5039)")
    print(f"  online head clip=0.50     {m['online_clip_raised']:.4f}")
    print(f"  clip saturation rate      {m['sat_frac_published']*100:.1f}% of updates")

    out = {"stage": 0, "task": "sim_xor", "sampler": "hard-threshold",
           "feature_decay": 0.25, "seeds": list(GROK_SEEDS),
           "per_seed": rows, "means": m,
           "brief_targets": {"ridge_decay025": 0.6953, "ridge_spikesum": 0.9701,
                             "online_frozen": 0.5039}}
    p = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r2"
    p.mkdir(parents=True, exist_ok=True)
    (p / "stage0-reproduction.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nwritten: {p / 'stage0-reproduction.json'}")


if __name__ == "__main__":
    main()
