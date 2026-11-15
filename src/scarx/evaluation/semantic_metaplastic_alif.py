"""GPU-only local importance/slow anchor research, with norm-matched control."""
import io
import json
import zipfile
import numpy as np
from .semantic_credit_alif import CreditALIF

FORMAT = 'scarx-local-importance-anchor-GPU-v1'


class ProtectedALIF:
    def __init__(self, seed, rule, budget=None, device='gpu'):
        if rule not in ('protected', 'norm_matched', 'unprotected') or device != 'gpu':
            raise ValueError('Registered GPU-only local protection rules required')
        self.credit = CreditALIF(seed, 'hidden', .2, budget, device)
        self.rule = rule

    @property
    def model(self):
        return self.credit.model

    def forward(self, vector):
        return self.credit.forward(vector)

    def learn(self, vector, target, target_norm=None):
        self.credit.check_gpu()
        if type(target) is not int or not 0 <= target < 6:
            raise ValueError('Bad label')
        if self.credit.budget is not None:
            self.credit.budget.charge('update')
        xp = self.model.xp
        p, _ = self.credit.forward(vector, track=True)
        error = -p.copy()
        error[target] += 1
        signal = self.model.core.w_out[2:8].T @ error
        direction = signal[:, None] * self.model.e_f[:, :64]
        before = self.model.core.w_in.copy()
        if self.rule == 'protected':
            importance = self.model.importance[:, :64]
            importance[:] = .99 * importance + .01 * direction**2
            q = xp.clip(importance / xp.maximum(importance.mean(axis=1, keepdims=True), np.float32(1e-8)), 0, 10)
            delta = xp.clip(.2 * direction / (1 + q) - .002 * q * (before - self.model.core.stable_in), -.025, .025)
        elif self.rule == 'norm_matched':
            if target_norm is None or not np.isfinite(target_norm) or target_norm < 0:
                raise ValueError('Required finite realized candidate update norm')
            def proposed(scale):
                value = xp.clip(scale * .2 * direction, -.025, .025)
                return xp.clip(before + value, -3, 3) - before
            low, high = 0., 1.
            for _ in range(30):
                if float(xp.linalg.norm(proposed(high))) >= target_norm:
                    break
                high *= 2
            else:
                raise ValueError('Norm-matched direction cannot reach candidate L2')
            for _ in range(24):
                middle = (low + high) / 2
                if float(xp.linalg.norm(proposed(middle))) < target_norm:
                    low = middle
                else:
                    high = middle
            delta = proposed((low + high) / 2)
        else:
            delta = xp.clip(.2 * direction, -.025, .025)
        self.model.core.w_in[:] = xp.clip(before + delta, -3, 3)
        actual_norm = float(xp.linalg.norm(self.model.core.w_in - before))
        norm_error = abs(actual_norm - target_norm) if self.rule == 'norm_matched' else None
        if norm_error is not None and norm_error > max(1e-6, 1e-4 * target_norm):
            raise ValueError('Matched update norm tolerance failure')
        if self.rule == 'protected':
            self.model.core.stable_in += .005 * (self.model.core.w_in - self.model.core.stable_in)
        self.credit.updates += 1
        self.credit.hidden_updates += 1
        if not all(xp.isfinite(a).all() for a in self.model.arrays().values()):
            raise FloatingPointError('Nonfinite protected synaptic state')
        return {'loss': -float(xp.log(xp.maximum(p[target], np.float32(1e-12)))),
                'prediction_before': int(xp.argmax(p)), 'target': target,
                'weight_delta_L2': actual_norm, 'norm_match_error': norm_error,
                'importance_sum': float(xp.sum(self.model.importance)),
                'anchor_drift_L2': float(xp.linalg.norm(self.model.core.w_in - self.model.core.stable_in)),
                'updates': self.credit.updates, 'hidden_updates': self.credit.hidden_updates,
                'active_spikes_last_tick': self.model.active_spikes(), 'device': 'gpu'}

    def dumps(self, controller=None):
        output = io.BytesIO()
        np.savez(output, inner=np.frombuffer(self.credit.dumps(controller), dtype=np.uint8),
                 metadata=np.frombuffer(json.dumps({'format': FORMAT, 'rule': self.rule}).encode(), dtype=np.uint8))
        return output.getvalue()

    @classmethod
    def loads(cls, payload, budget=None, device='gpu'):
        if device != 'gpu' or not isinstance(payload, bytes) or len(payload) > 1_000_000:
            raise ValueError('GPU-only protected checkpoint size/type')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != 2 or len(set(names)) != 2 or sum(i.file_size for i in archive.infolist()) > 1_000_000:
                raise ValueError('Duplicate/oversized protected checkpoint')
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            if set(data.files) != {'inner', 'metadata'}:
                raise ValueError('Wrong protected checkpoint members')
            if any(data[k].dtype != np.uint8 or data[k].ndim != 1 for k in data.files):
                raise ValueError('Protected checkpoint byte arrays required')
            meta = json.loads(data['metadata'].tobytes())
            if meta.get('format') != FORMAT or meta.get('rule') not in ('protected', 'norm_matched', 'unprotected'):
                raise ValueError('Wrong protection format/rule')
            obj = cls.__new__(cls)
            obj.rule = meta['rule']
            obj.credit, controller = CreditALIF.loads(data['inner'].tobytes(), budget, device)
            if obj.credit.arm != 'hidden' or obj.credit.eta != .2:
                raise ValueError('Protected inner rule mismatch')
            return obj, controller
