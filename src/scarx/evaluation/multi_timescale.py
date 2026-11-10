"""MTB research prototype: multi-timescale readout receptive field.

RESEARCH ONLY. Never imported by serving/product code. No autograd, no BPTT, no
RTRL, no reverse traversal, no hidden->hidden error transport.

Question this prototype exists to answer
----------------------------------------
Does giving the *readout* a bank of leaky integrators spanning several
timescales improve learning, when neuron dynamics, spiking, recurrence and the
hidden learning rule are held bit-identical?

Mechanism (the ONLY difference from production)
-----------------------------------------------
Production readout substrate is one first-order leaky integral:

    f[t] = d*f[t-1] + (1-d)*z[t]          # z = spikes, d = feature_decay

This class offers K such integrals with distinct d_k, concatenated into one
vector, feeding ONE shared readout weight matrix. Hidden dynamics are untouched,
so for a fixed seed every arm produces the same spike train — this is asserted,
not assumed.

Signal path (unchanged from the approved local rule)
---------------------------------------------------
    score[i]   = (z[i] - p[i]) / 0.4
    elig_in   += score[i] * V_pre[x]
    elig_rec  += score[i] * V_pre[prev]
    advantage  = clip(reward - reward_mean, -2, 2)
    dW_hidden  = clip(hidden_lr * advantage * elig)

The readout uses the readout-local delta rule on its own wider input. No error
is transported into hidden neurons; the neuron-local forward traces are the only
hidden learning signal.
"""
from dataclasses import asdict, replace
import hashlib
import io
import json
import zipfile

import numpy as np

from ..neural.core import Config

CHECKPOINT_FORMAT = "scarx-mtb-v1"

# The registered decay bank. d=0.25 is the production value and is included so
# that the candidate contains the incumbent basis as a subset.
DECAY_BANK = (0.10, 0.25, 0.50, 0.75, 0.90)


class MTCore:
    """ALIF core with a multi-timescale readout substrate.

    Neuron equations are copied verbatim from ``Config``/``SpikingCore`` so that
    a bit-identical spike train is achievable; ``spike_hashes`` exposes the check.
    """

    weight_names = ("w_in", "w_rec", "w_out", "bias", "stable_in", "stable_rec",
                    "stable_out", "stable_bias")
    state_names = ("voltage", "adaptation", "spikes", "pre", "post", "features",
                   "elig_in", "elig_rec", "rows", "voltage_pre")

    def __init__(self, config, decays=DECAY_BANK, rule="score", shuffle=False,
                 readout_projection=None, deterministic=False):
        if config.neuron != "alif":
            raise ValueError("MTB requires the ALIF neuron")
        if config.hidden > 128:
            raise ValueError("MTB is bounded to the registered hidden ceiling")
        decays = tuple(float(d) for d in decays)
        if not 1 <= len(decays) <= 8 or any(not 0 < d < 1 for d in decays):
            raise ValueError("Invalid decay bank")
        if rule not in ("score", "readout", "frozen"):
            raise ValueError("Invalid rule")
        self.config = config
        self.decays = decays
        self.rule = rule
        self.shuffle = bool(shuffle)
        # deterministic=True reproduces the shipping UnifiedCore neuron (threshold
        # crossing). deterministic=False reproduces the stochastic research neuron
        # used by reports 20/35. Both are supported so parity can be checked
        # against whichever reference a reviewer requires.
        self.deterministic = bool(deterministic)
        self.readout_projection = None if readout_projection is None else np.asarray(readout_projection, np.float32)

        rng = np.random.default_rng(config.seed)
        h, d, k, o = config.hidden, config.inputs, config.fan_in, config.outputs
        self.indices = np.stack([rng.choice(h, k, replace=False) for _ in range(h)]).astype(np.int32)
        self.w_in = rng.normal(.15, .45, (h, d)).astype(np.float32)
        self.w_rec = rng.normal(0, .12 / np.sqrt(k), (h, k)).astype(np.float32)

        self.substrate_width = self._substrate_width()
        self.w_out = rng.normal(0, .03, (o, self.substrate_width)).astype(np.float32)
        self.bias = np.zeros(o, dtype=np.float32)
        self.stable_in, self.stable_rec = self.w_in.copy(), self.w_rec.copy()
        self.stable_out, self.stable_bias = self.w_out.copy(), self.bias.copy()

        self.rng = np.random.default_rng(config.seed + 7_100_000)
        self.mod_rng = np.random.default_rng(config.seed + 7_200_000)
        self.updates = 0
        self.reward_mean = 0.
        self.ticks = 0
        self.reset_state()

    # ---------------- substrate ----------------

    def _substrate_width(self):
        base = len(self.decays) * self.config.hidden
        if self.readout_projection is not None:
            return int(self.readout_projection.shape[1])
        return base

    def reset_state(self):
        c = self.config
        self.voltage = np.zeros(c.hidden, np.float32)
        self.adaptation = np.zeros(c.hidden, np.float32)
        self.spikes = np.zeros(c.hidden, np.float32)
        self.pre = np.zeros(c.inputs + c.hidden, np.float32)
        self.post = np.zeros(c.hidden, np.float32)
        self.features = np.zeros((len(self.decays), c.hidden), np.float32)
        self.voltage_pre = np.zeros(c.inputs + c.hidden, np.float32)
        start = (self.updates * c.plastic_rows) % c.hidden
        self.rows = (np.arange(c.plastic_rows, dtype=np.int32) + start) % c.hidden
        self.elig_in = np.zeros((c.plastic_rows, c.inputs), np.float32)
        self.elig_rec = np.zeros((c.plastic_rows, c.fan_in), np.float32)

    def _flatten(self, f):
        """Normalise a feature argument to one of two unambiguous shapes.

        Accepted: the live state ``(K, H)``, an already-flat bank ``(K*H,)``,
        or a batch of flat banks ``(N, K*H)``. Anything else is a caller error.
        """
        f = np.asarray(f, np.float32)
        k, h = len(self.decays), self.config.hidden
        if f.ndim == 1:
            if f.shape != (k * h,):
                raise ValueError(f"Flattened bank must be ({k * h},), got {f.shape}")
            return f
        if f.ndim == 2:
            if f.shape == (k, h):
                return f.reshape(-1)
            if f.shape[1] == k * h:
                return f
        raise ValueError(f"Unsupported feature shape {f.shape} for K={k}, H={h}")

    def substrate(self, features=None):
        basis = self._flatten(self.features if features is None else features)
        if self.readout_projection is not None:
            basis = basis @ self.readout_projection
        return basis

    # ---------------- dynamics (verbatim production equations) ----------------

    def step(self, x, track=True):
        c = self.config
        x = np.asarray(x, np.float32)
        if x.shape != (c.inputs,) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid input vector")
        old = self.spikes.copy()
        self.voltage = c.leak * self.voltage + self.w_in @ x + np.sum(self.w_rec * old[self.indices], axis=1)
        limit = c.threshold + c.adaptation_strength * self.adaptation
        probability = 1 / (1 + np.exp(-np.clip((self.voltage - limit) / .4, -20, 20)))
        if self.deterministic:
            # shipping UnifiedCore behaviour: hard threshold crossing
            self.spikes = (self.voltage >= limit).astype(np.float32)
        else:
            # stochastic research neuron used by reports 20/35
            self.spikes = (self.rng.random(c.hidden) < probability).astype(np.float32)
        self.voltage -= self.spikes * limit
        self.adaptation = c.adaptation_decay * self.adaptation + self.spikes
        for i, d in enumerate(self.decays):
            self.features[i] = d * self.features[i] + (1 - d) * self.spikes
        if track:
            r = self.rows
            self.voltage_pre *= c.leak
            self.voltage_pre[:c.inputs] += x
            self.voltage_pre[c.inputs:] += old
            score = (self.spikes[r] - probability[r]) / .4
            self.elig_in += score[:, None] * self.voltage_pre[None, :c.inputs]
            self.elig_rec += score[:, None] * self.voltage_pre[c.inputs:][self.indices[r]]
            np.clip(self.elig_in, -10, 10, out=self.elig_in)
            np.clip(self.elig_rec, -10, 10, out=self.elig_rec)
            self.pre *= c.trace_decay
            self.pre[:c.inputs] += x
            self.pre[c.inputs:] += old
            self.post = c.trace_decay * self.post + self.spikes
        self.ticks += 1
        return self.probabilities()

    def probabilities(self, features=None):
        basis = self.substrate(features)
        logits = basis @ self.w_out.T + self.bias
        logits = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=-1, keepdims=True)

    def forward(self, sequence, track=False):
        self.reset_state()
        p = self.probabilities()
        for x in sequence:
            p = self.step(x, track=track)
        return p.copy()

    # ---------------- learning ----------------

    def _readout_update(self, features, target):
        c = self.config
        p = np.asarray(self.probabilities(features), np.float32)
        if p.ndim != 1:
            raise ValueError("Scalar readout update only")
        basis = self.substrate(features)
        error = -p.copy()
        error[target] += 1
        delta = c.readout_lr * error[:, None] * basis[None, :]
        self.w_out += np.clip(delta, -c.update_clip, c.update_clip)
        self.bias += np.clip(c.readout_lr * error, -c.update_clip, c.update_clip)
        np.clip(self.w_out, -c.weight_clip, c.weight_clip, out=self.w_out)
        np.clip(self.bias, -c.weight_clip, c.weight_clip, out=self.bias)

    def learn(self, target, reward=None):
        if not 0 <= target < self.config.outputs:
            raise ValueError("Invalid target")
        if reward is not None and not np.isfinite(reward):
            raise ValueError("Nonfinite reward")
        c = self.config
        p = self.probabilities()
        loss = -float(np.log(max(float(p[target]), 1e-12)))
        self._readout_update(self.features, target)
        raw = -loss if reward is None else float(reward)
        if not np.isfinite(raw):
            raise ValueError("Nonfinite reward")
        advantage = float(np.clip(raw - self.reward_mean, -2, 2))
        self.reward_mean = .95 * self.reward_mean + .05 * raw
        if self.shuffle:
            advantage *= float(self.mod_rng.choice([-1, 1]))
        if self.rule == "score":
            for key in ("in", "rec"):
                trace = getattr(self, "elig_" + key)
                weight = getattr(self, "w_" + key)
                signal = advantage * trace
                weight[self.rows] += np.clip(c.hidden_lr * signal, -c.update_clip, c.update_clip)
                np.clip(weight, -c.weight_clip, c.weight_clip, out=weight)
        self.updates += 1
        return loss

    # ---------------- evaluation (stateless, common random numbers) ----------------

    def _spike_from_voltage(self, voltage, limit, probability, rng, shape):
        if self.deterministic:
            return (voltage >= limit).astype(np.float32)
        return (rng.random(shape) < probability).astype(np.float32)

    def features_many(self, sequences, seed=901):
        x = np.asarray(sequences, np.float32)
        c = self.config
        if x.ndim != 3 or x.shape[2] != c.inputs or not np.isfinite(x).all():
            raise ValueError("Invalid batch")
        b, t, _ = x.shape
        v = np.zeros((b, c.hidden), np.float32)
        a = v.copy()
        z = v.copy()
        f = np.zeros((len(self.decays), b, c.hidden), np.float32)
        rng = np.random.default_rng(seed)
        for s in range(t):
            v = c.leak * v + x[:, s] @ self.w_in.T + np.sum(self.w_rec[None, :, :] * z[:, self.indices], axis=2)
            limit = c.threshold + c.adaptation_strength * a
            prob = 1 / (1 + np.exp(-np.clip((v - limit) / .4, -20, 20)))
            z = (rng.random((b, c.hidden)) < prob).astype(np.float32)
            v = v - z * limit
            a = c.adaptation_decay * a + z
            for i, d in enumerate(self.decays):
                f[i] = d * f[i] + (1 - d) * z
        out = f.transpose(1, 0, 2).reshape(b, -1)
        if self.readout_projection is not None:
            out = out @ self.readout_projection
        return out

    def predict_many(self, sequences, seed=901):
        f = self.features_many(sequences, seed)
        return np.argmax(f @ self.w_out.T + self.bias, axis=1)

    def spike_train_many(self, sequences, seed=901):
        """Spike trains for the bit-identity check across arms."""
        x = np.asarray(sequences, np.float32)
        c = self.config
        b, t, _ = x.shape
        v = np.zeros((b, c.hidden), np.float32)
        a = v.copy()
        z = v.copy()
        rng = np.random.default_rng(seed)
        out = []
        for s in range(t):
            v = c.leak * v + x[:, s] @ self.w_in.T + np.sum(self.w_rec[None, :, :] * z[:, self.indices], axis=2)
            limit = c.threshold + c.adaptation_strength * a
            prob = 1 / (1 + np.exp(-np.clip((v - limit) / .4, -20, 20)))
            z = (rng.random((b, c.hidden)) < prob).astype(np.float32)
            v = v - z * limit
            a = c.adaptation_decay * a + z
            out.append(z.copy())
        return np.stack(out, 1)

    def spike_hash(self, sequences, seed=901):
        return hashlib.sha256(self.spike_train_many(sequences, seed).tobytes()).hexdigest()

    # ---------------- persistence ----------------

    def arrays(self):
        out = {name: getattr(self, name) for name in self.weight_names + self.state_names + ("indices",)}
        if self.readout_projection is not None:
            out["readout_projection"] = self.readout_projection
        return out

    def memory_bytes(self):
        return sum(a.nbytes for a in self.arrays().values())

    def fingerprint(self):
        digest = hashlib.sha256()
        for name in self.weight_names:
            digest.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        return digest.hexdigest()

    def dumps(self):
        arrays = self.arrays()
        meta = {
            "format": CHECKPOINT_FORMAT,
            "config": asdict(self.config),
            "decays": list(self.decays),
            "rule": self.rule,
            "shuffle": self.shuffle,
            "updates": self.updates,
            "ticks": self.ticks,
            "reward_mean": self.reward_mean,
            "rng": self.rng.bit_generator.state,
            "mod_rng": self.mod_rng.bit_generator.state,
            "sha256": {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                       for k, v in arrays.items()},
        }
        stream = io.BytesIO()
        np.savez(stream, metadata=np.array(json.dumps(meta)), **arrays)
        return stream.getvalue()

    @classmethod
    def loads(cls, payload):
        if len(payload) > 100_000_000:
            raise ValueError("Oversized MTB checkpoint")
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if sum(i.file_size for i in z.infolist()) > 100_000_000:
                raise ValueError("Oversized MTB checkpoint")
            if z.getinfo("metadata.npy").file_size > 65536:
                raise ValueError("Oversized metadata")
            if len(z.namelist()) != len(set(z.namelist())):
                raise ValueError("Duplicate checkpoint members")
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if meta["format"] != CHECKPOINT_FORMAT:
                raise ValueError("MTB format")
            projection = z["readout_projection"] if "readout_projection" in z.files else None
            m = cls(Config(**meta["config"]), meta["decays"], meta["rule"], meta["shuffle"], projection)
            if set(z.files) != set(m.arrays()) | {"metadata"}:
                raise ValueError("Fields")
            for k, template in m.arrays().items():
                a = z[k]
                if a.shape != template.shape or a.dtype != template.dtype:
                    raise ValueError("Shape/dtype")
                if not np.isfinite(a).all():
                    raise ValueError("Corrupt array")
                if hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() != meta["sha256"][k]:
                    raise ValueError("Corrupt array")
                template[:] = a
            m.updates = int(meta["updates"])
            m.ticks = int(meta["ticks"])
            m.reward_mean = float(meta["reward_mean"])
            m.rng.bit_generator.state = meta["rng"]
            m.mod_rng.bit_generator.state = meta["mod_rng"]
            if m.updates < 0 or m.ticks < 0 or not np.isfinite(m.reward_mean):
                raise ValueError("Invalid state")
            if np.any(m.indices < 0) or np.any(m.indices >= m.config.hidden):
                raise ValueError("Invalid state")
            if np.any(m.rows < 0) or np.any(m.rows >= m.config.hidden):
                raise ValueError("Invalid state")
        return m

    def save(self, path):
        from pathlib import Path
        import os
        import tempfile
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.dumps()
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path):
        from pathlib import Path
        path = Path(path)
        if path.stat().st_size > 100_000_000:
            raise ValueError("Oversized MTB checkpoint")
        return cls.loads(path.read_bytes())


def make_dimension_matched_projection(seed, base_width, out_width):
    """Fixed random projection used ONLY by the matched-dimension control.

    Same temporal content as production (single timescale), expanded to the
    candidate's width so parameter count and readout compute are equal. Without
    this control, any MTB gain could be dismissed as "more parameters".
    """
    if out_width < base_width:
        raise ValueError("Matched projection must not shrink the basis")
    rng = np.random.default_rng(seed + 8_300_000)
    w = rng.normal(0, 1 / np.sqrt(base_width), (base_width, out_width))
    return w.astype(np.float32)


PARITY_ALT_SEED = 991
