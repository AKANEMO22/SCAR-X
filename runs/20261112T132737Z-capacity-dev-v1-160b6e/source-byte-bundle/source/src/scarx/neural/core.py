"""ALIF, sparse recurrence and three-factor plasticity without autograd.

Only the readout sees a vector of output errors. Hidden synapses receive one
scalar reward advantage and their own pre/post traces. This is a heuristic
reward-modulated STDP rule, NOT an unbiased gradient estimator or e-prop.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Config:
    inputs: int = 8
    hidden: int = 64
    outputs: int = 4
    fan_in: int = 8
    plastic_rows: int = 16
    seed: int = 17
    leak: float = .85
    threshold: float = .65
    adaptation_decay: float = .95
    adaptation_strength: float = .08
    trace_decay: float = .9
    feature_decay: float = .92
    readout_lr: float = .08
    hidden_lr: float = .002
    anchor_strength: float = .0001
    consolidation_rate: float = .02
    update_clip: float = .025
    weight_clip: float = 3.
    ram_limit: int = 2_000_000_000
    neuron: str = "alif"

    def validate(self):
        if min(self.inputs, self.hidden, self.outputs, self.fan_in, self.plastic_rows) < 1:
            raise ValueError("Network dimensions must be positive")
        if self.fan_in > self.hidden or self.plastic_rows > self.hidden:
            raise ValueError("fan_in and plastic_rows must not exceed hidden")
        if self.neuron not in ("alif", "rate"):
            raise ValueError("Unknown neuron")
        if not 0 < self.ram_limit <= 2_000_000_000:
            raise ValueError("RAM limit must be in (0, 2,000,000,000]")
        for name in ("leak", "adaptation_decay", "trace_decay", "feature_decay", "consolidation_rate"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(name)
        for name in ("threshold", "adaptation_strength", "readout_lr", "hidden_lr", "anchor_strength", "update_clip", "weight_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name)
        # Conservative allocation estimate includes checkpoint/load copies and runtime.
        estimated = 24 * (self.hidden*self.inputs + self.hidden*self.fan_in + self.hidden*self.outputs)
        estimated += 16*self.plastic_rows*(self.inputs+self.fan_in) + 100_000_000
        if estimated > self.ram_limit:
            raise MemoryError(f"Estimated working allocation {estimated:,} exceeds RAM budget")


class SpikingCore:
    weight_names = ("w_in", "w_rec", "w_out", "bias", "stable_in", "stable_rec", "stable_out", "stable_bias")
    state_names = ("voltage", "adaptation", "spikes", "pre", "post", "features", "elig_in", "elig_rec", "rows")

    def __init__(self, config=Config()):
        config.validate()
        self.config = config
        rng = np.random.default_rng(config.seed)
        h, d, k, o = config.hidden, config.inputs, config.fan_in, config.outputs
        self.indices = np.stack([rng.choice(h, k, replace=False) for _ in range(h)]).astype(np.int32)
        self.w_in = rng.normal(.15, .45, (h, d)).astype(np.float32)
        self.w_rec = rng.normal(0, .12 / np.sqrt(k), (h, k)).astype(np.float32)
        self.w_out = rng.normal(0, .03, (o, h)).astype(np.float32)
        self.bias = np.zeros(o, dtype=np.float32)
        self.stable_in, self.stable_rec = self.w_in.copy(), self.w_rec.copy()
        self.stable_out, self.stable_bias = self.w_out.copy(), self.bias.copy()
        self.updates = 0
        self.reward_mean = 0.
        self.reset_state()

    def reset_state(self):
        c = self.config
        self.voltage = np.zeros(c.hidden, np.float32)
        self.adaptation = np.zeros(c.hidden, np.float32)
        self.spikes = np.zeros(c.hidden, np.float32)
        self.pre = np.zeros(c.inputs+c.hidden, np.float32)
        self.post = np.zeros(c.hidden, np.float32)
        self.features = np.zeros(c.hidden, np.float32)
        # Rotate blocks between segments, never during a segment.
        start = (self.updates*c.plastic_rows) % c.hidden
        self.rows = (np.arange(c.plastic_rows, dtype=np.int32)+start) % c.hidden
        self.elig_in = np.zeros((c.plastic_rows, c.inputs), np.float32)
        self.elig_rec = np.zeros((c.plastic_rows, c.fan_in), np.float32)

    def step(self, x, track=True):
        c = self.config
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (c.inputs,) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid input vector")
        old = self.spikes.copy()
        current = self.w_in @ x + np.sum(self.w_rec*old[self.indices], axis=1)
        self.voltage = c.leak*self.voltage + current
        if c.neuron == "alif":
            limit = c.threshold + c.adaptation_strength*self.adaptation
            self.spikes = (self.voltage >= limit).astype(np.float32)
            self.voltage -= self.spikes*limit
        else:
            self.spikes = np.tanh(self.voltage).astype(np.float32)
        self.adaptation = c.adaptation_decay*self.adaptation + self.spikes
        self.features = c.feature_decay*self.features + (1-c.feature_decay)*self.spikes
        if track:
            r = self.rows
            # Traces from previous steps produce causal and anti-causal pairs.
            self.elig_in *= c.trace_decay
            self.elig_in += self.spikes[r,None]*self.pre[None,:c.inputs] - .5*self.post[r,None]*x[None,:]
            self.elig_rec *= c.trace_decay
            self.elig_rec += self.spikes[r,None]*self.pre[c.inputs:][self.indices[r]] - .5*self.post[r,None]*old[self.indices[r]]
            np.clip(self.elig_in, -5, 5, out=self.elig_in)
            np.clip(self.elig_rec, -5, 5, out=self.elig_rec)
            self.pre *= c.trace_decay
            self.pre[:c.inputs] += x
            self.pre[c.inputs:] += old
            self.post = c.trace_decay*self.post + self.spikes
        return self.probabilities()

    def probabilities(self):
        logits = self.w_out @ self.features + self.bias
        logits -= np.max(logits)
        p = np.exp(logits)
        return p / p.sum()

    def forward(self, sequence, track=False):
        self.reset_state()
        p = self.probabilities()
        for x in sequence:
            p = self.step(x, track=track)
        return p.copy()

    def learn(self, target, mode="full", reward=None):
        if mode not in ("full", "frozen", "stdp", "readout"):
            raise ValueError("Unknown learning mode")
        if not 0 <= target < self.config.outputs:
            raise ValueError("Invalid target")
        if reward is not None and not np.isfinite(reward):
            raise ValueError("Nonfinite reward")
        c = self.config
        p = self.probabilities()
        loss = -float(np.log(max(float(p[target]), 1e-12)))
        if mode == "frozen":
            return loss
        # Output-local delta rule; no feedback matrix or reverse traversal.
        error = -p.copy()
        error[target] += 1
        delta = c.readout_lr * error[:,None]*self.features[None,:]
        self.w_out += np.clip(delta-c.anchor_strength*(self.w_out-self.stable_out), -c.update_clip, c.update_clip)
        self.bias += np.clip(c.readout_lr*error-c.anchor_strength*(self.bias-self.stable_bias), -c.update_clip, c.update_clip)
        raw_reward = float(p[target]) if reward is None else float(reward)
        if not np.isfinite(raw_reward):
            raise ValueError("Nonfinite reward")
        advantage = float(np.clip(raw_reward-self.reward_mean, -1, 1))
        self.reward_mean = .95*self.reward_mean + .05*raw_reward
        if mode in ("full", "stdp"):
            modulation = advantage if mode == "full" else 1.
            for name, trace in (("in", self.elig_in), ("rec", self.elig_rec)):
                weights = getattr(self, "w_"+name)
                stable = getattr(self, "stable_"+name)
                delta = c.hidden_lr*modulation*trace-c.anchor_strength*(weights[self.rows]-stable[self.rows])
                weights[self.rows] += np.clip(delta, -c.update_clip, c.update_clip)
                np.clip(weights, -c.weight_clip, c.weight_clip, out=weights)
        np.clip(self.w_out, -c.weight_clip, c.weight_clip, out=self.w_out)
        np.clip(self.bias, -c.weight_clip, c.weight_clip, out=self.bias)
        self.updates += 1
        return loss

    def consolidate(self):
        for a, b in (("stable_in","w_in"),("stable_rec","w_rec"),("stable_out","w_out"),("stable_bias","bias")):
            getattr(self,a)[:] += self.config.consolidation_rate*(getattr(self,b)-getattr(self,a))

    def arrays(self):
        return {name: getattr(self,name) for name in self.weight_names+self.state_names+("indices",)}

    def memory_bytes(self):
        return sum(a.nbytes for a in self.arrays().values())

    def fingerprint(self):
        digest = hashlib.sha256()
        for name in self.weight_names:
            digest.update(getattr(self, name).tobytes())
        return digest.hexdigest()

    def save(self, path):
        from .checkpoint import save_checkpoint
        save_checkpoint(self, path)

    def dumps(self):
        from .checkpoint import dumps_checkpoint
        return dumps_checkpoint(self)

    @classmethod
    def loads(cls,payload):
        from .checkpoint import loads_checkpoint
        return loads_checkpoint(payload)

    @classmethod
    def load(cls, path):
        from .checkpoint import load_checkpoint
        return load_checkpoint(path)

