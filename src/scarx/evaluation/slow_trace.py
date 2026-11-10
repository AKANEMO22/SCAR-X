"""Research prototype: one-step readout error times a slow forward trace.

The neuron's own feature decay stays at the Generation-1 value 0.25. The extra
trace does not enter the voltage. This is not a serving rule and not BPTT.
"""
import hashlib
import io
import json
import zipfile

import numpy as np

from ..neural.core import Config
from .readout_eligibility import ReadoutEligibility, spike_slope


SLOW = np.float32(0.95)
FORMAT = 'scarx-grok-slow-trace-v1'


class SlowTraceEligibility(ReadoutEligibility):
    def __init__(self, config, condition='correct', mode='slow'):
        if mode not in ('fast', 'slow'):
            raise ValueError('Slow-trace mode must be fast or slow')
        if config.feature_decay != 0.25:
            raise ValueError('Feature decay is locked at the Generation-1 value 0.25')
        super().__init__(config, condition)
        self.mode = mode
        self.slow = np.zeros(config.hidden, np.float32)
        self.e_s = np.zeros((config.hidden, config.inputs + config.fan_in), np.float32)

    def begin(self, track=True):
        super().begin(track)
        self.slow.fill(0)
        self.e_s.fill(0)

    def _presynaptic_sensitivity(self, x):
        c = self.core.config
        x = np.asarray(x, np.float32)
        old_z = self.core.spikes
        theta = c.threshold + c.adaptation_strength * self.core.adaptation
        current = self.core.w_in @ x + np.sum(self.core.w_rec * old_z[self.core.indices], axis=1)
        u = c.leak * self.core.voltage + current
        psi = spike_slope(u - theta)
        pre = np.concatenate((np.broadcast_to(x, (c.hidden, c.inputs)), old_z[self.core.indices]), axis=1)
        u_e = c.leak * self.e_v + pre
        a_old = self.e_a.copy()
        return psi[:, None] * (u_e - c.adaptation_strength * a_old)

    def step(self, x):
        sensitivity = self._presynaptic_sensitivity(x) if self.mode == 'slow' and self.track else None
        features = super().step(x)
        if self.mode == 'slow':
            self.slow[:] = SLOW * self.slow + (1 - SLOW) * self.core.spikes
            if sensitivity is not None:
                self.e_s[:] = SLOW * self.e_s + (1 - SLOW) * sensitivity
                if not np.isfinite(self.e_s).all():
                    raise ValueError('Nonfinite slow eligibility')
        return features

    def accumulate(self, target):
        if self.mode == 'fast':
            return super().accumulate(target)
        if not self.segment_active or not self.steps_in_segment or not self.head.frozen or type(target) is not int or not 0 <= target < len(self.output_mask) or not self.output_mask[target]:
            raise ValueError('Calibrated supervised endpoint required')
        feature = self.slow
        p = self.head.probabilities(feature, self.output_mask)
        error = -p.copy()
        error[target] += 1
        scale = np.maximum(np.sqrt(self.head.m2 / (self.head.count - 1)), .05).astype(np.float32)
        raw = (feature - self.head.mean.astype(np.float32)) / scale
        signal = (self.head.weights.T @ error) * (np.abs(raw) < 5) / scale
        permutation = self.signal_rng.permutation(self.core.config.hidden)
        if self.condition == 'shuffled':
            signal = signal[permutation]
        if self.condition == 'zero':
            signal = np.zeros_like(signal)
        if self.condition in ('correct', 'shuffled'):
            self.bank += signal[:, None] * self.e_s
            if not np.isfinite(self.bank).all():
                raise ValueError('Nonfinite slow direction')
        loss = self.head.learn(feature, target, self.output_mask)
        self.core.updates += 1
        self.endpoints += 1
        return loss

    def features(self, sequence, reset_each_tick=False):
        if self.mode == 'fast':
            return super().features(sequence, reset_each_tick)
        x = self._sequence(sequence, self.core.config.inputs)
        self.begin(False)
        for row in x:
            if reset_each_tick:
                self.core.reset_state()
                self.slow.fill(0)
            self.step(row)
        self.segment_active = False
        return self.slow.copy()

    def memory_bytes(self):
        return super().memory_bytes() + self.slow.nbytes + self.e_s.nbytes

    def state_view(self, sequence):
        x = self._sequence(sequence, self.core.config.inputs)
        self.begin(False)
        for row in x:
            self.step(row)
        self.segment_active = False
        return self.core.features.copy(), self.slow.copy(), self.core.adaptation.copy()

    def dumps(self):
        base = ReadoutEligibility.dumps(self)
        extra = io.BytesIO()
        np.savez(extra, slow=self.slow, e_s=self.e_s)
        extra = extra.getvalue()
        meta = {
            'format': FORMAT,
            'mode': self.mode,
            'base_sha256': hashlib.sha256(base).hexdigest(),
            'extra_sha256': hashlib.sha256(extra).hexdigest(),
        }
        meta['metadata_sha256'] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as archive:
            archive.writestr(zipfile.ZipInfo('base.zip'), base)
            archive.writestr(zipfile.ZipInfo('extra.npz'), extra)
            archive.writestr(zipfile.ZipInfo('metadata.json'), json.dumps(meta))
        raw = out.getvalue()
        if len(raw) > 5_000_000:
            raise MemoryError('Slow-trace checkpoint exceeds 5MB')
        return raw

    @classmethod
    def loads(cls, payload):
        if len(payload) > 5_000_000:
            raise ValueError('Slow-trace checkpoint bound')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(infos) != 3 or len(set(names)) != 3 or set(names) != {'base.zip', 'extra.npz', 'metadata.json'}:
                raise ValueError('Slow-trace checkpoint members')
            if sum(item.file_size for item in infos) > 5_000_000:
                raise ValueError('Slow-trace checkpoint bound')
            if archive.getinfo('metadata.json').file_size > 65536 or archive.getinfo('extra.npz').file_size > 2_000_000:
                raise ValueError('Slow-trace checkpoint bound')
            base = archive.read('base.zip')
            extra = archive.read('extra.npz')
            meta = json.loads(archive.read('metadata.json'))
        checksum = meta.pop('metadata_sha256')
        if meta.get('format') != FORMAT or meta.get('mode') not in ('fast', 'slow'):
            raise ValueError('Slow-trace format')
        if hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest() != checksum:
            raise ValueError('Slow-trace checksum')
        if hashlib.sha256(base).hexdigest() != meta['base_sha256'] or hashlib.sha256(extra).hexdigest() != meta['extra_sha256']:
            raise ValueError('Slow-trace payload checksum')
        loaded = ReadoutEligibility.loads(base)
        if loaded.core.config.feature_decay != 0.25:
            raise ValueError('Slow-trace checkpoint is not the locked Generation-1 decay')
        model = cls(loaded.core.config, loaded.condition, meta['mode'])
        model.core = loaded.core
        model.head = loaded.head
        model.core.w_out = model.head.weights
        model.core.bias = model.head.bias
        for name in ReadoutEligibility.array_names:
            getattr(model, name)[:] = getattr(loaded, name)
        for name in ('total_ticks', 'steps_in_segment', 'endpoints', 'hidden_updates', 'segments_learned', 'segment_active', 'track', 'last_direction_norm'):
            setattr(model, name, getattr(loaded, name))
        model.signal_rng.bit_generator.state = loaded.signal_rng.bit_generator.state
        with np.load(io.BytesIO(extra), allow_pickle=False) as arrays:
            if set(arrays.files) != {'slow', 'e_s'}:
                raise ValueError('Slow-trace extra fields')
            slow = arrays['slow']
            eligibility = arrays['e_s']
            if slow.dtype != np.float32 or eligibility.dtype != np.float32:
                raise ValueError('Slow-trace dtype')
            if slow.shape != model.slow.shape or eligibility.shape != model.e_s.shape:
                raise ValueError('Slow-trace extra shape')
            if not np.isfinite(slow).all() or not np.isfinite(eligibility).all() or np.any(slow < 0) or np.any(slow > 1):
                raise ValueError('Slow-trace extra values')
            model.slow[:] = slow
            model.e_s[:] = eligibility
        return model


def g1_config(seed):
    return Config(
        inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=seed,
        leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02,
        adaptation_decay=.95, readout_lr=.025, hidden_lr=.0001, anchor_strength=0,
        weight_clip=20., neuron='alif',
    )


def order_sequences(seed, count):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    first = rng.integers(0, 2, count)
    for i, channel in enumerate(first):
        other = 1 - int(channel)
        values[i, 0, int(channel)] += 1.5
        values[i, 1, other] += 1.5
        values[i, 10, 4] += 1.5
        values[i, 11, 5] += 1.5
    return values, first.astype(np.int32)
