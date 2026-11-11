"""Does the learning failure survive on the shipped (hard-threshold) sampler?

Every negative learning result in this campaign was obtained with the Bernoulli
neuron (`PlasticCore.features_many`). The product ships the hard-threshold neuron
(`SpikingCore.step` / `unified.py`). Section 4.2 of point-neuron-nonlinearity.md
showed the two differ by a 6.5x firing-rate inflation on identical weights.

This runs the same decisive contrast on BOTH paths, same generator, same seeds,
same budget:

    learned hidden  vs  frozen hidden

If learning is null on both paths, the campaign's negatives are robust.
If it is null only on Bernoulli, the negatives were measured on the wrong neuron.

Local scalar-score learning, matching the rule the campaign actually used.
No autograd, no BPTT, no RTRL.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scarx.evaluation.phase21 import interval  # noqa: E402
from scarx.evaluation.tasks21 import ridge_probe  # noqa: E402
from scarx.neural.core import Config  # noqa: E402

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
S = 1.5
SEEDS = (61001, 61007, 61013)


def cfg(seed, lr=.05):
    return Config(seed=seed, readout_lr=lr, hidden_lr=.002, anchor_strength=0., **CFG)


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


class Net:
    """One ALIF core with a linear readout, both samplers available."""

    def __init__(self, seed, lr, mode, hidden_learn=True):
        c = cfg(seed, lr)
        self.c = c
        self.mode = mode
        self.hidden_learn = hidden_learn
        g = np.random.default_rng(c.seed)
        self.indices = np.stack([g.choice(c.hidden, c.fan_in, replace=False)
                                 for _ in range(c.hidden)]).astype(np.int32)
        self.w_in = g.normal(.15, .45, (c.hidden, c.inputs)).astype(np.float32)
        self.w_rec = g.normal(0, .12 / np.sqrt(c.fan_in), (c.hidden, c.fan_in)).astype(np.float32)
        self.w_out = g.normal(0, .03, (c.outputs, c.hidden)).astype(np.float32)
        self.bias = np.zeros(c.outputs, np.float32)
        self.rng = np.random.default_rng(c.seed + 7_100_000)
        self.reward_mean = 0.0
        self.elig_in = np.zeros((c.plastic_rows, c.inputs), np.float32)
        self.elig_rec = np.zeros((c.plastic_rows, c.fan_in), np.float32)
        self.rows = np.arange(c.plastic_rows, dtype=np.int32)
        self.reset()

    def reset(self):
        c = self.c
        self.v = np.zeros(c.hidden, np.float32)
        self.a = np.zeros(c.hidden, np.float32)
        self.z = np.zeros(c.hidden, np.float32)
        self.f = np.zeros(c.hidden, np.float32)
        self.vpre = np.zeros(c.inputs + c.hidden, np.float32)
        self.elig_in[:] = 0
        self.elig_rec[:] = 0

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
        lg = self.w_out @ self.f + self.bias
        lg = lg - lg.max()
        e = np.exp(lg)
        return e / e.sum()

    def learn(self, target):
        c = self.c
        p = self.probs()
        loss = -float(np.log(max(float(p[target]), 1e-12)))
        err = -p.copy(); err[target] += 1
        self.w_out += np.clip(c.readout_lr * err[:, None] * self.f[None, :],
                              -c.update_clip, c.update_clip)
        self.bias += np.clip(c.readout_lr * err, -c.update_clip, c.update_clip)
        if self.hidden_learn:
            adv = float(np.clip(-loss - self.reward_mean, -2, 2))
            self.reward_mean = .95 * self.reward_mean + .05 * (-loss)
            for key, tr in (("in", self.elig_in), ("rec", self.elig_rec)):
                w = getattr(self, "w_" + key)
                w[self.rows] += np.clip(c.hidden_lr * adv * tr, -c.update_clip, c.update_clip)
                np.clip(w, -c.weight_clip, c.weight_clip, out=w)
        return loss

    def run_sequence(self, seq, track):
        self.reset()
        for x in seq:
            self.step(x, track)

    def features_batch(self, seqs, seed=901):
        """Stateless evaluation with common random numbers, both modes."""
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


def trial(seed, mode, learn_hidden):
    fx, fy = sim_xor(seed + 100_000, 128)
    tr_x, tr_y = sim_xor(seed + 200_000, 1500)
    tx, ty = sim_xor(seed + 300_000, 300)
    m = Net(seed, .05, mode, hidden_learn=learn_hidden)
    before = m.w_in.copy()
    losses = []
    for seq, lab in zip(tr_x, tr_y):
        m.run_sequence(seq, track=True)
        for _ in range(3):          # 3 update passes per example, matching prior work
            losses.append(m.learn(int(lab)))
    moved = not np.array_equal(m.w_in, before)
    # evaluation head refit from scratch on the learned hidden
    probe = ridge_probe(m.features_batch(fx), fy, m.features_batch(tx), ty)
    direct = float(np.mean(np.argmax(m.features_batch(tx) @ m.w_out.T + m.bias, 1) == ty))
    init = Net(seed, .05, mode, hidden_learn=False)
    probe0 = ridge_probe(init.features_batch(fx), fy, init.features_batch(tx), ty)
    return {"seed": seed, "mode": mode, "arm": "learn" if learn_hidden else "frozen",
            "probe": probe, "probe_initial": probe0, "probe_gain": probe - probe0,
            "direct": direct, "hidden_moved": bool(moved),
            "mean_loss": float(np.mean(losses))}


def main():
    print("=" * 76)
    print("Does the learning negative survive on the SHIPPED hard-threshold neuron?")
    print("task: sim_xor  |  3 seeds  |  same generator, same budget, both samplers")
    print("=" * 76)
    rows = []
    for mode in ("bernoulli", "hard"):
        print(f"\n--- {mode} ---")
        for seed in SEEDS:
            for arm in (True, False):
                r = trial(seed, mode, arm)
                rows.append(r)
                tag = "learn " if arm else "frozen"
                print(f"  seed{seed} {tag}  probe={r['probe']:.4f} "
                      f"gain={r['probe_gain']:+.4f} direct={r['direct']:.4f} "
                      f"hidden_moved={r['hidden_moved']}")

    print("\n" + "=" * 76)
    print("PAIRED CONTRAST: learn - frozen (probe gain)")
    print("=" * 76)
    out = {"trials": rows, "contrasts": {}}
    for mode in ("bernoulli", "hard"):
        g = [r["probe_gain"] for r in rows if r["mode"] == mode and r["arm"] == "learn"]
        f = [r["probe_gain"] for r in rows if r["mode"] == mode and r["arm"] == "frozen"]
        d = [a - b for a, b in zip(g, f)]
        iv = interval(d)
        moved = any(r["hidden_moved"] for r in rows if r["mode"] == mode and r["arm"] == "learn")
        print(f"  {mode:10s} learn-frozen = {iv['mean']*100:+7.2f}pp "
              f"[{iv['lower95']*100:+7.2f},{iv['upper95']*100:+7.2f}]  hidden_moved={moved}")
        out["contrasts"][mode] = iv

    p = Path(__file__).resolve().parents[2] / "artifacts" / "deepseek-r1"
    p.mkdir(parents=True, exist_ok=True)
    (p / "sampler-learning-comparison.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nwritten: {p / 'sampler-learning-comparison.json'}")


if __name__ == "__main__":
    main()
