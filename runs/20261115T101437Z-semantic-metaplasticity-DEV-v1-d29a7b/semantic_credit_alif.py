"""Research-only signed one-step error times local forward ALIF eligibility.

GPU-only numerical path. No autograd/BPTT, cross-neuron temporal Jacobian,
momentum or change to the original ALIF forward dynamics.
"""
import io
import json
import zipfile
import numpy as np
from ..hybrid.learner import LocalALIF

FORMAT = 'scarx-semantic-credit-alif-GPU-v1'
CODEC = 'train-centered-nativeQwen1024-PCA64-unitFP32-v1'


def initial_model(seed):
    # CPU initialization only; never CPU forward or training.
    model = LocalALIF(seed, mode='frozen')
    w = model.core.w_in
    w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), np.float32(1e-8))
    model.core.stable_in[:] = w
    model.f_mean.fill(0)
    model.f_var.fill(1)
    model.core.w_out.fill(0)
    for c in range(6):
        model.core.w_out[2 + c, c * 21:(c + 1) * 21] = np.float32(20 / 21)
    model.core.stable_out[:] = model.core.w_out
    return model


class CreditALIF:
    def __init__(self, seed, arm, eta=.05, budget=None, device='gpu'):
        if device != 'gpu':
            raise ValueError('GPU-only research: CPU forward/training forbidden')
        if arm not in ('hidden', 'readout_only', 'frozen') or eta not in (.05, .2):
            raise ValueError('Unregistered credit condition/rate')
        self.seed, self.arm, self.eta, self.budget = int(seed), arm, float(eta), budget
        self.model = initial_model(seed).to_device('gpu')
        self.updates, self.hidden_updates = 0, 0

    def check_gpu(self):
        if (self.model.device != 'gpu' or not type(self.model.core.w_in).__module__.startswith('cupy')
                or self.model.core.w_in.dtype != np.float32):
            raise RuntimeError('CUDA FP32 required, no silent CPU fallback')

    def forward(self, vector, track=False):
        self.check_gpu()
        if self.budget is not None:
            self.budget.charge('forward')
        xp = self.model.xp
        x = xp.asarray(vector, dtype=xp.float32)
        if x.shape != (64,) or not xp.isfinite(x).all():
            raise ValueError('Invalid semantic vector')
        p, f, _, _, _ = self.model._forward_native(xp.repeat(x[None, :], 8, axis=0), track=track)
        return p, f

    def learn(self, vector, target):
        self.check_gpu()
        if self.arm == 'frozen' or type(target) is not int or not 0 <= target < 6:
            raise ValueError('Invalid learning request')
        if self.budget is not None:
            self.budget.charge('update')
        xp = self.model.xp
        p, f = self.forward(vector, track=self.arm == 'hidden')
        error = -p.copy()
        error[target] += 1
        loss = -float(xp.log(xp.maximum(p[target], np.float32(1e-12))))
        if self.arm == 'hidden':
            signal = self.model.core.w_out[2:8].T @ error
            direction = signal[:, None] * self.model.e_f[:, :64]
            delta = xp.clip(np.float32(self.eta) * direction, -.025, .025)
            before = self.model.core.w_in.copy()
            self.model.core.w_in += delta
            xp.clip(self.model.core.w_in, -3, 3, out=self.model.core.w_in)
            delta_norm = float(xp.linalg.norm(self.model.core.w_in - before))
            self.hidden_updates += 1
            signal_norm = float(xp.linalg.norm(signal))
            eligibility_norm = float(xp.linalg.norm(self.model.e_f[:, :64]))
        else:
            before = self.model.core.w_out.copy()
            self.model.core.w_out[2:8] += xp.clip(.04 * error[:, None] * f[None, :], -.025, .025)
            self.model.core.bias[2:8] += xp.clip(.04 * error, -.025, .025)
            xp.clip(self.model.core.w_out, -3, 3, out=self.model.core.w_out)
            xp.clip(self.model.core.bias, -3, 3, out=self.model.core.bias)
            delta_norm = float(xp.linalg.norm(self.model.core.w_out - before))
            signal_norm = eligibility_norm = 0.
        self.updates += 1
        if not all(xp.isfinite(a).all() for a in self.model.arrays().values()):
            raise FloatingPointError('Nonfinite local credit state')
        return {'loss': loss, 'prediction_before': int(xp.argmax(p)), 'target': target,
                'probabilities': self.model.host(p).tolist(), 'weight_delta_L2': delta_norm,
                'signal_norm': signal_norm, 'eligibility_norm': eligibility_norm,
                'updates': self.updates, 'hidden_updates': self.hidden_updates,
                'active_spikes_last_tick': self.model.active_spikes(), 'device': 'gpu'}

    def dumps(self, controller=None):
        meta = {'format': FORMAT, 'codec': CODEC, 'seed': self.seed, 'arm': self.arm,
                'eta': self.eta, 'updates': self.updates, 'hidden_updates': self.hidden_updates,
                'controller': controller or {}}
        output = io.BytesIO()
        np.savez(output, model=np.frombuffer(self.model.dumps(), dtype=np.uint8),
                 metadata=np.frombuffer(json.dumps(meta, allow_nan=False).encode(), dtype=np.uint8))
        return output.getvalue()

    @classmethod
    def loads(cls, payload, budget=None, device='gpu'):
        if device != 'gpu':
            raise ValueError('GPU-only research: no CPU restore execution')
        if not isinstance(payload, bytes) or len(payload) > 1_000_000:
            raise ValueError('Checkpoint size/type')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != 2 or len(set(names)) != 2 or sum(i.file_size for i in archive.infolist()) > 1_000_000:
                raise ValueError('Duplicate/oversized research checkpoint')
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            if set(data.files) != {'model', 'metadata'}:
                raise ValueError('Checkpoint member mismatch')
            for key in data.files:
                if data[key].dtype != np.uint8 or data[key].ndim != 1:
                    raise ValueError('Checkpoint byte arrays required')
            meta = json.loads(data['metadata'].tobytes())
            if meta.get('format') != FORMAT or meta.get('codec') != CODEC:
                raise ValueError('Wrong research format/codec')
            if (meta.get('arm') not in ('hidden', 'readout_only', 'frozen') or meta.get('eta') not in (.05, .2)
                    or type(meta.get('seed')) is not int):
                raise ValueError('Invalid research identity')
            for key in ('updates', 'hidden_updates'):
                if type(meta.get(key)) is not int or not 0 <= meta[key] <= 1500:
                    raise ValueError('Invalid checkpoint counter')
            expected = meta['updates'] if meta['arm'] == 'hidden' else 0
            if meta['hidden_updates'] != expected or (meta['arm'] == 'frozen' and meta['updates']):
                raise ValueError('Inconsistent research counter')
            obj = cls.__new__(cls)
            obj.seed, obj.arm, obj.eta, obj.budget = meta['seed'], meta['arm'], meta['eta'], budget
            obj.model, _ = LocalALIF.loads(data['model'].tobytes(), device='gpu')
            obj.updates, obj.hidden_updates = meta['updates'], meta['hidden_updates']
            obj.check_gpu()
            return obj, meta['controller']
