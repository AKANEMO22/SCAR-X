"""Research-only locally taught input synapses; ALIF kernel is unchanged.

Fixed semantic encoder is shared with controls. No torch, reverse gradients,
BPTT or analog nearest-prototype decision path at inference.
"""
import io
import json
import zipfile
import numpy as np
from ..hybrid.learner import LocalALIF

FORMAT = 'scarx-semantic-prototype-ALIF-v1'
CODEC = 'train-centered-nativeQwen1024-PCA64-unitFP32-v1'
ARMS = ('prototype', 'readout_only', 'frozen', 'shuffled')


class SemanticALIF:
    def __init__(self, seed, arm, budget=None, device='gpu'):
        if arm not in ARMS:
            raise ValueError('Unknown research condition')
        self.model = LocalALIF(seed, mode='frozen')
        weights = self.model.core.w_in
        weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-8)
        self.model.core.stable_in[:] = weights
        self.model.f_var.fill(1)
        self.model.f_mean.fill(0)
        self.initial_input = weights.copy()
        self.initial_recurrent = self.model.core.w_rec.copy()
        self.counts = np.zeros(128, dtype=np.int64)
        self.signal_rng = np.random.default_rng(seed + 93_000_000)
        self.arm, self.seed, self.budget = arm, int(seed), budget
        self.updates = 0
        self.hidden_updates = 0
        self.spikes = 0
        self.forward_attempts = 0
        if device == 'gpu':
            self.model.to_device('gpu')
        elif device != 'cpu':
            raise ValueError('Unknown device')

    def forward(self, vector):
        if self.budget is not None:
            self.budget.charge('forward')
        self.forward_attempts += 1
        xp = self.model.xp
        x = xp.asarray(vector, dtype=xp.float32)
        if x.shape != (64,) or not xp.isfinite(x).all():
            raise ValueError('Expected finite64D vector')
        _, features, _, _, _ = self.model._forward_native(xp.repeat(x[None, :], 8, axis=0))
        self.spikes = self.model.active_spikes()
        if self.arm == 'readout_only':
            scores = self.model.core.w_out[2:8] @ features + self.model.core.bias[2:8]
        else:
            scores = 4 * self.model.core.features[:126].reshape(6, 21).max(axis=1)
        scores = scores - scores.max()
        p = xp.exp(scores)
        p /= p.sum()
        if not xp.isfinite(p).all():
            raise FloatingPointError('Nonfinite ALIF output')
        return self.model.host(p).copy(), features

    def learn(self, vector, label):
        if type(label) is not int or not 0 <= label < 6:
            raise ValueError('Invalid training label')
        if self.arm == 'frozen':
            raise ValueError('Frozen condition has zero update attempts')
        if self.budget is not None:
            self.budget.charge('update')
        p, features = self.forward(vector)
        loss = -float(np.log(max(float(p[label]), 1e-12)))
        xp = self.model.xp
        core = self.model.core
        if self.arm == 'readout_only':
            before = core.w_out[2:8].copy()
            error = -xp.asarray(p)
            error[label] += 1
            core.w_out[2:8] += xp.clip(.04 * error[:, None] * features[None, :], -.025, .025)
            core.bias[2:8] += xp.clip(.04 * error, -.025, .025)
            delta = float(xp.linalg.norm(core.w_out[2:8] - before))
            chosen, assigned = None, label
        else:
            assigned = int(self.signal_rng.integers(6)) if self.arm == 'shuffled' else label
            pool = np.arange(assigned * 21, (assigned + 1) * 21)
            used = pool[self.counts[pool] > 0]
            unused = pool[self.counts[pool] == 0]
            x = xp.asarray(vector, dtype=xp.float32)
            if len(used):
                drive = core.w_in[xp.asarray(used)] @ x
                nearest = int(xp.argmax(drive))
                winner, similarity = int(used[nearest]), float(drive[nearest])
            else:
                winner, similarity = None, -1.
            chosen = int(unused[0]) if len(unused) and (winner is None or similarity < .95) else winner
            if chosen is None:
                raise ValueError('No valid neuron')
            before = core.w_in[chosen].copy()
            count = int(self.counts[chosen])
            value = x.copy() if count == 0 else before + np.float32(1 / (count + 1)) * (x - before)
            value /= xp.maximum(xp.linalg.norm(value), np.float32(1e-8))
            core.w_in[chosen] = value
            self.counts[chosen] += 1
            delta = float(xp.linalg.norm(value - before))
            self.hidden_updates += 1
        self.updates += 1
        return {'loss': loss, 'prediction_before': int(np.argmax(p)), 'probability': p.tolist(),
                'target': label, 'plasticity_label': assigned, 'chosen_neuron': chosen,
                'weight_delta_L2': delta, 'updates': self.updates, 'hidden_updates': self.hidden_updates,
                'active_spikes_last_tick': self.spikes, 'device': self.model.device}

    def reset_hidden(self):
        self.model.core.w_in[:] = self.model.xp.asarray(self.initial_input)
        self.model.core.w_rec[:] = self.model.xp.asarray(self.initial_recurrent)

    def dumps(self, controller=None):
        meta = {'format': FORMAT, 'codec': CODEC, 'arm': self.arm, 'seed': self.seed,
                'updates': self.updates, 'hidden_updates': self.hidden_updates,
                'signal_rng': self.signal_rng.bit_generator.state, 'controller': controller or {}}
        buffer = io.BytesIO()
        np.savez(buffer, model=np.frombuffer(self.model.dumps(), dtype=np.uint8), counts=self.counts,
                 initial_input=self.initial_input, initial_recurrent=self.initial_recurrent,
                 metadata=np.frombuffer(json.dumps(meta, allow_nan=False).encode(), dtype=np.uint8))
        return buffer.getvalue()

    @classmethod
    def loads(cls, payload, budget=None, device='gpu'):
        if not isinstance(payload, bytes) or len(payload) > 2_000_000:
            raise ValueError('Research checkpoint byte limit')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != 5 or len(set(names)) != 5 or sum(i.file_size for i in archive.infolist()) > 2_000_000:
                raise ValueError('Duplicate/oversized checkpoint member')
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            if set(arrays.files) != {'model', 'counts', 'initial_input', 'initial_recurrent', 'metadata'}:
                raise ValueError('Unexpected research checkpoint members')
            for key in ('model', 'metadata'):
                if arrays[key].dtype != np.uint8 or arrays[key].ndim != 1:
                    raise ValueError('Checkpoint byte array dtype')
            meta = json.loads(arrays['metadata'].tobytes())
            if meta.get('format') != FORMAT or meta.get('codec') != CODEC:
                raise ValueError('Wrong research checkpoint/codec')
            obj = cls(meta['seed'], meta['arm'], budget, device='cpu')
            for key in ('counts', 'initial_input', 'initial_recurrent'):
                value, expected = arrays[key], getattr(obj, key)
                if value.shape != expected.shape or value.dtype != expected.dtype or not np.isfinite(value).all():
                    raise ValueError('Checkpoint array type/shape: ' + key)
                setattr(obj, key, value.copy())
            if np.any(obj.counts < 0):
                raise ValueError('Negative prototype counts')
            for key in ('updates', 'hidden_updates'):
                if type(meta[key]) is not int or not 0 <= meta[key] <= 4500:
                    raise ValueError('Bad update counter')
                setattr(obj, key, meta[key])
            if (obj.hidden_updates != (obj.updates if obj.arm in ('prototype', 'shuffled') else 0)
                    or int(obj.counts.sum()) != obj.hidden_updates):
                raise ValueError('Inconsistent synapse update counters')
            obj.signal_rng.bit_generator.state = meta['signal_rng']
            obj.model, _ = LocalALIF.loads(arrays['model'].tobytes(), device=device)
            return obj, meta['controller']
