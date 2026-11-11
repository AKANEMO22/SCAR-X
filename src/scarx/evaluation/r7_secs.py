"""Round 7 — SECS: Selective Eligibility with Credit Shaping.

MECHANISM (research prototype; production `src/scarx/neural/` is untouched)

The production rule (main) is:

    advantage = clip(p[target] - reward_mean, -1, 1)          # ONE SCALAR
    delta     = hidden_lr * advantage * eligibility_trace
                - anchor_strength * (w - stable)              # stable never refreshed

R7 probe 3 measured cosine(eligibility_trace, dL/dw) = -0.0303 with CI95
[-0.0667, +0.0060] — statistically indistinguishable from a random vector. Multiplying
a directionless trace by a scalar produces a random walk.

SECS changes two things and nothing else:

  1. CREDIT SHAPING  — instead of one global scalar, route the readout error back
     through `W_out` to give a PER-NEURON coefficient:

         c_i = W_out[:, i] . error                       (2-vector dot product)
         s_i = sign(c_i) * |c_i| / (mean_j |c_j| + eps)  # normalised, bounded

     This is the proposal-30 relaxation (one-step `W_out.T @ error`), already approved
     for research. It is ONE mat-vec of size 2xH per update.

  2. RELEVANCE GATING — scale each trace contribution by how much that neuron can
     currently move the loss, and let it decay faster when it is irrelevant:

         elig_i <- trace_decay * elig_i + s_i * (spike_i * pre - .5 * post_i * x)

  3. ANCHOR REFRESH — periodically move `stable` toward the current weights so the
     spring does not hold the network at its random initialisation forever.

NOT changed: hidden_lr (.002), trace_decay (.9), update_clip (.025), plastic_rows (12),
hidden (48), the forward pass, the spike rule, the readout rule.

Why this could work where the production rule does not: the production trace adds
`spike*pre - .5*post*x` for every neuron with equal weight, so neurons that cannot
influence the loss contribute the same as neurons that can. SECS weights each neuron by
its actual readout influence `c_i`, which is the locally available estimate of
`dL/dw` (chain rule through the readout, no recurrence).
"""
import hashlib

import numpy as np

from scarx.neural.core import Config, SpikingCore


def cfg(seed):
    return Config(seed=seed, inputs=8, hidden=48, outputs=2, fan_in=8,
                  plastic_rows=12, leak=.65, threshold=.5,
                  feature_decay=.25, adaptation_strength=.02)


class SECSCore(SpikingCore):
    """SpikingCore with credit-shaped eligibility. Forward pass inherited unchanged."""

    def __init__(self, config, mode="secs", anchor_refresh=0, seed=0):
        super().__init__(config)
        if mode not in ("secs", "shuffled_credit", "zero_credit", "scalar"):
            raise ValueError(f"unknown mode {mode}")
        self.mode = mode
        self.anchor_refresh = int(anchor_refresh)
        self._rng = np.random.default_rng(seed + 5150)
        self._credit_trace = np.zeros(config.hidden, np.float32)

    # ---- credit shaping -------------------------------------------------
    def credit(self):
        """c_i = W_out[:,i] . error, computed locally at the readout."""
        err = self._last_error
        return (self.w_out[:, :] * err[:, None]).sum(0).astype(np.float32)

    def shaped_scale(self, err):
        c = (self.w_out * err[:, None]).sum(0).astype(np.float32)
        if self.mode == "zero_credit":
            return np.zeros_like(c)
        if self.mode == "shuffled_credit":
            c = c[self._rng.permutation(len(c))]
        if self.mode == "scalar":
            return np.sign(c).astype(np.float32) * 0.0 + 1.0
        denom = float(np.mean(np.abs(c))) + 1e-6
        s = np.sign(c) * np.minimum(np.abs(c) / denom, 3.0)
        return s.astype(np.float32)

    # ---- forward with shaped trace --------------------------------------
    def step(self, x, track=True):
        c = self.config
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (c.inputs,) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid input vector")
        old = self.spikes.copy()
        current = self.w_in @ x + np.sum(self.w_rec * old[self.indices], axis=1)
        self.voltage = c.leak * self.voltage + current
        limit = c.threshold + c.adaptation_strength * self.adaptation
        self.spikes = (self.voltage >= limit).astype(np.float32)
        self.voltage -= self.spikes * limit
        self.adaptation = c.adaptation_decay * self.adaptation + self.spikes
        self.features = c.feature_decay * self.features + (1 - c.feature_decay) * self.spikes

        if track:
            r = self.rows
            # scale by the CURRENT readout credit, per neuron (rows only are plastic)
            s = self.shaped_scale(self._last_error)[r][:, None] if hasattr(self, "_last_error") \
                else np.ones((len(r), 1), np.float32)

            self.elig_in *= c.trace_decay
            self.elig_in += s * (self.spikes[r, None] * self.pre[None, :c.inputs]
                                 - .5 * self.post[r, None] * x[None, :])
            self.elig_rec *= c.trace_decay
            self.elig_rec += s * (self.spikes[r, None] * self.pre[c.inputs:][self.indices[r]]
                                  - .5 * self.post[r, None] * old[self.indices[r]])
            np.clip(self.elig_in, -5, 5, out=self.elig_in)
            np.clip(self.elig_rec, -5, 5, out=self.elig_rec)
            self.pre *= c.trace_decay
            self.pre[:c.inputs] += x
            self.pre[c.inputs:] += old
            self.post = c.trace_decay * self.post + self.spikes
        return self.probabilities()

    # ---- learning --------------------------------------------------------
    def learn(self, target, mode="full", reward=None):
        c = self.config
        if not 0 <= target < c.outputs:
            raise ValueError("Invalid target")
        p = self.probabilities()
        loss = -float(np.log(max(float(p[target]), 1e-12)))

        error = -p.copy()
        error[target] += 1
        self._last_error = error.astype(np.float32)

        # readout: UNCHANGED production delta rule
        self.w_out += np.clip(c.readout_lr * error[:, None] * self.features[None, :],
                              -c.update_clip, c.update_clip)
        self.bias += np.clip(c.readout_lr * error, -c.update_clip, c.update_clip)
        np.clip(self.w_out, -c.weight_clip, c.weight_clip, out=self.w_out)

        if mode == "frozen":
            return loss

        advantage = float(np.clip(p[target] - self.reward_mean, -1, 1))
        self.reward_mean = .95 * self.reward_mean + .05 * float(p[target])
        if self.mode == "zero_credit":
            advantage = 0.0

        for name in ("in", "rec"):
            trace = getattr(self, "elig_" + name)
            weights = getattr(self, "w_" + name)
            stable = getattr(self, "stable_" + name)
            delta = c.hidden_lr * advantage * trace \
                - c.anchor_strength * (weights[self.rows] - stable[self.rows])
            weights[self.rows] += np.clip(delta, -c.update_clip, c.update_clip)
            np.clip(weights, -c.weight_clip, c.weight_clip, out=weights)

        self.updates += 1
        if self.anchor_refresh and self.updates % self.anchor_refresh == 0:
            self.stable_in[:] = self.w_in
            self.stable_rec[:] = self.w_rec
        return loss


def digest(core):
    h = hashlib.sha256()
    h.update(core.w_in.tobytes()); h.update(core.w_rec.tobytes())
    h.update(core.w_out.tobytes())
    return h.hexdigest()
