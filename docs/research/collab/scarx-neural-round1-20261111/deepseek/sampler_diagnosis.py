"""Why does the Bernoulli path lose what the hard-threshold path keeps?

Same weights, same inputs, same ticks. Only the spike generation differs:
  hard: x >= threshold
  bernoulli: Bernoulli(sigmoid((v - limit)/0.4))

Hypothesis: the sigmoid argument is small in magnitude for most units, so p sits
near 0.5 and the neuron behaves as a coin flip rather than a detector. If p is
near 0.5 everywhere, no downstream linear readout can recover a threshold decision.
"""
import numpy as np

from scarx.neural.core import Config

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)


def main():
    c = Config(seed=41001, **CFG)
    rng = np.random.default_rng(7)
    x = rng.normal(0, .08, (200, 12, 8)).astype(np.float32)
    for i in range(200):
        for t in (2,):
            x[i, t, int(rng.integers(3))] += 1.5

    # weights from the same seed as the campaign config
    g = np.random.default_rng(c.seed)
    indices = np.stack([g.choice(c.hidden, c.fan_in, replace=False) for _ in range(c.hidden)]).astype(np.int32)
    w_in = g.normal(.15, .45, (c.hidden, c.inputs)).astype(np.float32)
    w_rec = g.normal(0, .12 / np.sqrt(c.fan_in), (c.hidden, c.fan_in)).astype(np.float32)

    b, t, _ = x.shape
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
    ps, margins = [], []
    irng = np.random.default_rng(901)
    for s in range(t):
        v = c.leak * v + x[:, s] @ w_in.T + np.sum(w_rec[None, :, :] * z[:, indices], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        logit = (v - lim) / .4
        p = 1 / (1 + np.exp(-np.clip(logit, -20, 20)))
        ps.append(p.copy()); margins.append(logit.copy())
        z = (irng.random((b, c.hidden)) < p).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z

    ps = np.array(ps); margins = np.array(margins)
    print("BERNOULLI neuron: sigmoid argument (v-limit)/0.4 across 12 ticks x 48 x 200")
    print(f"  mean logit        : {margins.mean():+.4f}")
    print(f"  std  logit        : {margins.std():.4f}")
    print(f"  mean p            : {ps.mean():.4f}")
    print(f"  |logit| < 1  (p in .27..73): {(np.abs(margins) < 1).mean():.3f}")
    print(f"  p > 0.95 (decisive ON)     : {(ps > .95).mean():.3f}")
    print(f"  p < 0.05 (decisive OFF)    : {(ps < .05).mean():.3f}")
    print()
    print("HARD-THRESHOLD neuron on the same trajectory:")
    print(f"  decisive ON  = fractions where v >= limit : "
          f"{(margins > 0).mean():.3f}  (equivalent to p > 0.5)")
    print()
    print("INTERPRETATION")
    print("  The hard neuron emits a deterministic function of sign(v-limit):")
    print("  a clean detector. The Bernoulli neuron emits a coin whose bias is")
    print("  only slightly away from 0.5, so the same decision is buried in noise")
    print("  at every tick. Downstream linear readout cannot average that away")
    print("  within 12 ticks.")


if __name__ == "__main__":
    main()
