"""P0-only isolated ALIF history ablation. Never imported by serving code.

The pinned live learner supplies unchanged forward/credit/readout/statistics.
Only its three history/energy/delta statements are replaced in a copied method.
No autodiff, backward, new dataset, DEV training or live checkpoint migration.
"""

import ast
from collections import Counter
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import textwrap
import time
import zipfile

import numpy as np
from scarx.hybrid.learner import LocalALIF

BASE_SHA = 'c38f7dd9412da3bc14a02a3a83611941d15aee1c558c05f4bd26671a6872111a'
ARMS = ('frozen', 'readout_only', 'current_hidden', 'current_only', 'matched')
SEEDS = (260313, 260337)
FORMAT = 'scarx-history-ablation-p0-v1'


class P0Budget:
    """Append-only precharges, shared across attempts/restarts; no refunds."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self):
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if not raw or not raw.endswith(b'\n') or len(raw) > 400_000:
            raise ValueError('Incomplete/corrupt budget; do not reset it')
        rows = [json.loads(line) for line in raw.splitlines()]
        previous = '0' * 64
        counts = Counter()
        for index, row in enumerate(rows, 1):
            if set(row) != {'format', 'attempt', 'arm', 'seed', 'source', 'epoch', 'previous', 'sha256'}:
                raise ValueError('Budget record schema')
            digest = row['sha256']
            data = {k: v for k, v in row.items() if k != 'sha256'}
            if (row['format'] != FORMAT or row['attempt'] != index or row['arm'] not in ARMS
                    or row['seed'] not in SEEDS or row['previous'] != previous
                    or digest != hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
                    or not np.isfinite(row['epoch']) or row['epoch'] <= 0):
                raise ValueError('Budget integrity failure; no automatic recovery')
            counts[row['arm'], row['seed']] += 1
            if index > 600 or counts[row['arm'], row['seed']] > 100:
                raise ValueError('Persisted budget exceeds registered bounds')
            previous = digest
        return rows

    def charge(self, arm, seed, source):
        if arm not in ARMS or seed not in SEEDS or source not in ('cpu', 'gpu', 'production_reference'):
            raise ValueError('Unregistered P0 attempt')
        lock = self.path.with_suffix('.lock')
        handle = lock.open('x')
        try:
            with handle:
                rows = self.records()
                now = time.time()
                if rows and (now < rows[-1]['epoch'] or now - rows[0]['epoch'] > 300):
                    raise RuntimeError('Registered P0 wall budget expired; keep all attempts')
                count = sum(r['arm'] == arm and r['seed'] == seed for r in rows)
                if len(rows) >= 600 or count >= 100:
                    raise RuntimeError('Registered P0 attempt budget exhausted')
                row = {'format': FORMAT, 'attempt': len(rows) + 1, 'arm': arm, 'seed': seed,
                       'source': source, 'epoch': now, 'previous': rows[-1]['sha256'] if rows else '0' * 64}
                row['sha256'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                with self.path.open('ab') as handle:
                    handle.write((json.dumps(row, sort_keys=True) + '\n').encode())
                    handle.flush()
                    os.fsync(handle.fileno())
                return row['attempt']
        finally:
            lock.unlink()


class NormMismatch(ValueError):
    pass


def history_delta(xp, weights, trace, energy, direction, arm):
    """Original FP32 weights/updates; float64 only for scalar norm diagnostics."""
    if arm not in ('current_hidden', 'current_only', 'matched'):
        raise ValueError('Not a hidden plasticity arm')
    if any(a.shape != (128, 72) or a.dtype != xp.float32 or not bool(xp.isfinite(a).all())
           for a in (weights, trace, energy, direction)):
        raise ValueError('Expected finite FP32 H128 synapse arrays')
    if bool(xp.any(energy <= 0)) or bool(xp.any(xp.abs(weights) > 3)):
        raise ValueError('Invalid energy/weight bounds')
    next_energy = .99 * energy + .01 * direction ** 2
    historical = .9 * trace + .1 * direction
    next_trace = historical if arm == 'current_hidden' else .1 * direction
    denominator = xp.sqrt(next_energy) + 1e-4
    current = .008 * historical / denominator
    raw = .008 * next_trace / denominator
    def effective(vector):
        return xp.clip(weights + xp.clip(vector, -.01, .01), -3, 3) - weights
    def norm(value):
        return float(xp.sqrt(xp.sum(value.astype(xp.float64) ** 2)))
    target = norm(effective(current))
    alpha, iterations = 1.0, 0
    if arm == 'matched':
        if target == 0:
            alpha = 0.0
        elif norm(effective(raw)) != target:
            maximum = norm(effective(256.0 * raw))
            if maximum < .99 * target:
                raise NormMismatch(f'Target={target}; maximum support={maximum}; no attribution')
            low, high = 0.0, 256.0
            best = (abs(norm(effective(raw)) - target), 1.0)
            for iterations in range(1, 21):
                middle = (low + high) / 2
                size = norm(effective(middle * raw))
                best = min(best, (abs(size - target), middle))
                if size < target:
                    low = middle
                else:
                    high = middle
            alpha = best[1]
    delta = xp.clip(alpha * raw, -.01, .01)
    achieved = norm(effective(alpha * raw))
    error = abs(achieved - target) / target if target else achieved
    if arm == 'matched' and error > .01:
        raise NormMismatch(f'Postclip norm error={error}; no attribution')
    return next_trace, next_energy, delta, {
        'alpha': alpha, 'bisection_iterations': iterations, 'target_norm': target,
        'actual_norm': achieved, 'relative_norm_error': error,
    }


def copied_learn_source():
    path = Path(inspect.getfile(LocalALIF))
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Pinned base learner changed; do not adapt silently')
    tree = ast.parse(textwrap.dedent(inspect.getsource(LocalALIF.learn)))
    function = tree.body[0]
    replacements = 0
    for node in ast.walk(function):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self.mode in ('learned', 'shuffled')":
            expected = ['self.syn_trace[:]', 'self.syn_energy[:]', 'delta']
            assert [ast.unparse(n.targets[0]) for n in node.body[2:5]] == expected
            node.body[2:5] = ast.parse('delta = self._history_update(direction)').body
            replacements += 1
    if replacements != 1:
        raise ValueError('Expected exactly one copied history-block replacement')
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


_namespace = {'np': np}
exec(compile(copied_learn_source(), '<pinned-P0-learn-copy>', 'exec'), _namespace)


class HistoryALIF(LocalALIF):
    def __init__(self, seed, arm, budget):
        if seed not in SEEDS or arm not in ARMS or not isinstance(budget, P0Budget):
            raise ValueError('Unregistered isolated P0 model')
        super().__init__(seed, 'frozen' if arm == 'frozen' else 'readout_only' if arm == 'readout_only' else 'learned')
        self.arm, self.budget, self.last_math = arm, budget, None

    def _history_update(self, direction):
        xp = self.xp
        weights = xp.concatenate((self.core.w_in, self.core.w_rec), axis=1)
        before = {'weights': self.host(weights).copy(), 'trace': self.host(self.syn_trace).copy(),
                  'energy': self.host(self.syn_energy).copy(), 'direction': self.host(direction).copy()}
        trace, energy, delta, details = history_delta(xp, weights, self.syn_trace, self.syn_energy, direction, self.arm)
        self.syn_trace[:] = trace
        self.syn_energy[:] = energy
        self.last_math = {**before, 'next_trace': self.host(trace).copy(),
                          'next_energy': self.host(energy).copy(), 'delta': self.host(delta).copy(), **details}
        return delta

    def learn(self, *args, **kwargs):
        self.budget.charge(self.arm, self.seed, self.device)
        return _namespace['learn'](self, *args, **kwargs)

    def dumps(self, controller=None):
        payload = LocalALIF.dumps(self, controller)
        metadata = {'format': FORMAT, 'arm': self.arm, 'base_sha256': BASE_SHA,
                    'encoder': 'signed-hash64-v1', 'state_sha256': hashlib.sha256(payload).hexdigest()}
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name, raw in [('research.json', json.dumps(metadata, sort_keys=True).encode()), ('state.npz', payload)]:
                archive.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), raw)
        return output.getvalue()

    @classmethod
    def loads(cls, payload, *, budget, expected_arm, device='gpu'):
        if not isinstance(payload, bytes) or len(payload) > 16_000_000 or expected_arm not in ARMS:
            raise ValueError('Invalid P0 checkpoint envelope')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            if (len(infos) != 2 or {i.filename for i in infos} != {'research.json', 'state.npz'}
                    or any(i.compress_type != zipfile.ZIP_STORED for i in infos)
                    or sum(i.file_size for i in infos) > 16_000_000):
                raise ValueError('Invalid/duplicate/oversized research members')
            metadata = json.loads(archive.read('research.json'))
            state = archive.read('state.npz')
        if (set(metadata) != {'format', 'arm', 'base_sha256', 'encoder', 'state_sha256'}
                or metadata['format'] != FORMAT or metadata['arm'] != expected_arm
                or metadata['base_sha256'] != BASE_SHA or metadata['encoder'] != 'signed-hash64-v1'
                or hashlib.sha256(state).hexdigest() != metadata['state_sha256']):
            raise ValueError('P0 checkpoint identity/checksum mismatch')
        base, controller = LocalALIF.loads(state)
        model = cls(base.seed, expected_arm, budget)
        if (model.mode != base.mode or base.hidden_updates > base.updates
                or base.core.updates != base.updates
                or not np.array_equal(base.gamma, model.gamma)
                or not np.array_equal(base.core.indices, model.core.indices)
                or any(np.any(np.abs(getattr(base.core, k)) > 3) for k in ('w_in', 'w_rec', 'w_out', 'bias'))
                or (expected_arm in ('frozen', 'readout_only') and base.hidden_updates != 0)
                or any(not isinstance(r['evidence'].get('id'), str) or not r['evidence']['id'] for r in base.replay)):
            raise ValueError('Arm/mode mismatch')
        model.__dict__.update(base.__dict__)
        model.arm, model.budget, model.last_math = expected_arm, budget, None
        model.to_device(device)
        return model, controller
