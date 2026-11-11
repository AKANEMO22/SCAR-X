"""V4/V5: does the ONLINE head ever learn, or does it sit at chance?

The claim under test: on the hard path a z-scored ridge probe reads sim_xor at
~0.69-0.72, but the delta-rule head that the scalar learning rule actually
updates sits at ~0.50, and mean training loss stays near ln(2)=0.693 so the
advantage (=-loss - baseline) is always ~0.

If true, my sentence "the readout is demonstrably capable, so the failure is
isolated to the learning rule" is wrong: the failing readout and the capable
readout are DIFFERENT READOUTS.

Trains the real delta-rule head and prints accuracy + loss every 250 examples.
"""
import numpy as np

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
S = 1.5
LN2 = float(np.log(2))


def cfg(seed=41001, lr=.05):
    from scarx.neural.core import Config
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


def ridge(train_f, train_y, test_f, test_y):
    mean = train_f.mean(0); scale = train_f.std(0) + 1e-4
    A = np.column_stack(((train_f - mean) / scale, np.ones(len(train_f))))
    B = np.column_stack(((test_f - mean) / scale, np.ones(len(test_f))))
    tgt = np.eye(2)[train_y]
    w = np.linalg.solve(A.T @ A + .01 * np.eye(A.shape[1]), A.T @ tgt)
    return float(np.mean(np.argmax(B @ w, 1) == test_y))


class Online:
    """The delta-rule head the scalar rule actually trains. No z-scoring."""

    def __init__(self, seed, mode, lr=.05, learn_hidden=True):
        c = cfg(seed, lr)
        self.c = c
        self.mode = mode
        self.learn_hidden = learn_hidden
        g = np.random.default_rng(c.seed)
        self.idx = np.stack([g.choice(c.hidden, c.fan_in, replace=False) for _ in range(c.hidden)]).astype(np.int32)
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
        self.v = np.zeros(c.hidden, np.float32); self.a = np.zeros(c.hidden, np.float32)
        self.z = np.zeros(c.hidden, np.float32); self.f = np.zeros(c.hidden, np.float32)
        self.vpre = np.zeros(c.inputs + c.hidden, np.float32)
        self.elig_in[:] = 0; self.elig_rec[:] = 0

    def step(self, x, track=False):
        c = self.c
        old = self.z.copy()
        self.v = c.leak * self.v + self.w_in @ x + np.sum(self.w_rec * old[self.idx], axis=1)
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
            self.elig_rec += score[:, None] * self.vpre[c.inputs:][self.idx[r]]
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
        self.w_out += np.clip(c.readout_lr * err[:, None] * self.f[None, :], -c.update_clip, c.update_clip)
        self.bias += np.clip(c.readout_lr * err, -c.update_clip, c.update_clip)
        adv = float(np.clip(-loss - self.reward_mean, -2, 2))
        self.reward_mean = .95 * self.reward_mean + .05 * (-loss)
        if self.learn_hidden:
            for key, tr in (("in", self.elig_in), ("rec", self.elig_rec)):
                w = getattr(self, "w_" + key)
                w[self.rows] += np.clip(c.hidden_lr * adv * tr, -c.update_clip, c.update_clip)
                np.clip(w, -c.weight_clip, c.weight_clip, out=w)
        return loss, adv

    def run(self, seq, track):
        self.reset()
        for x in seq:
            self.step(x, track)

    def batch_features(self, seqs, seed=901):
        c = self.c
        x = np.asarray(seqs, np.float32); b, t, _ = x.shape
        v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy(); f = v.copy()
        rng = np.random.default_rng(seed)
        for s in range(t):
            v = c.leak * v + x[:, s] @ self.w_in.T + np.sum(self.w_rec[None, :, :] * z[:, self.idx], axis=2)
            lim = c.threshold + c.adaptation_strength * a
            p = 1 / (1 + np.exp(-np.clip((v - lim) / .4, -20, 20)))
            z = (v >= lim).astype(np.float32) if self.mode == "hard" else \
                (rng.random((b, c.hidden)) < p).astype(np.float32)
            v = v - z * lim
            a = c.adaptation_decay * a + z
            f = c.feature_decay * f + (1 - c.feature_decay) * z
        return f


def main():
    print("=" * 78)
    print("V4/V5  online delta-rule head vs z-scored ridge, hard path, sim_xor")
    print("=" * 78)
    for seed in (61001, 61007, 61013):
        fx, fy = sim_xor(seed + 100_000, 128)
        trx, try_ = sim_xor(seed + 200_000, 1500)
        tx, ty = sim_xor(seed + 300_000, 300)

        frozen = Online(seed, "hard", learn_hidden=False)
        ridge_frozen = ridge(frozen.batch_features(fx), fy, frozen.batch_features(tx), ty)
        direct_frozen = float(np.mean(np.argmax(frozen.batch_features(tx) @ frozen.w_out.T
                                               + frozen.bias, 1) == ty))

        m = Online(seed, "hard", learn_hidden=True)
        losses, advs = [], []
        for i, (seq, lab) in enumerate(zip(trx, try_)):
            m.run(seq, True)
            for _ in range(3):
                l, a = m.learn(int(lab))
                losses.append(l); advs.append(a)
            if (i + 1) % 500 == 0:
                acc = float(np.mean(np.argmax(m.batch_features(tx) @ m.w_out.T + m.bias, 1) == ty))
                print(f"  seed{seed} after {i+1:4d} ex: head_acc={acc:.4f} "
                      f"mean_loss={np.mean(losses[-1500:]):.4f} |adv|={np.mean(np.abs(advs[-1500:])):.4f}")

        ridge_learned = ridge(m.batch_features(fx), fy, m.batch_features(tx), ty)
        direct_learned = float(np.mean(np.argmax(m.batch_features(tx) @ m.w_out.T
                                               + m.bias, 1) == ty))
        print(f"  seed{seed} FINAL: ridge_frozen={ridge_frozen:.4f} ridge_learned={ridge_learned:.4f}")
        print(f"  seed{seed} FINAL: head_frozen={direct_frozen:.4f} head_learned={direct_learned:.4f}")
        print(f"  seed{seed} FINAL: mean_loss={np.mean(losses):.4f} (ln2={LN2:.4f}) "
              f"mean|adv|={np.mean(np.abs(advs)):.4f}")
        print()

    print("=" * 78)
    print("If ridge >> head, the capable readout and the failing readout are DIFFERENT.")
    print("If mean_loss ~ ln2 and mean|adv| ~ 0, the scalar rule is starved.")
    print("=" * 78)


if __name__ == "__main__":
    main()
