"""Locked screen: slow trace visible to the head, per-neuron error, frozen gate.

Rules fixed before the run:
- Head on the slow trace reads s, decay 0.95. Fast features stay at 0.25.
- Signal is W_out.T @ error times the forward sensitivity of s.
- A task supports a learning claim only if slow-frozen development accuracy
  is below 0.65. Otherwise the result is recorded as a precondition failure.
- Seeds 81011, 81017, 81023. Two passes over 64 training sequences.
- No threshold change after seeing scores. No serving-core import.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import psutil

from scarx.evaluation.slow_trace import SlowTraceEligibility, g1_config, order_sequences


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs' / 'grok-slow-head-signal-v1'
SEEDS = (81011, 81017, 81023)
ARMS = (('slow-frozen', 'frozen'), ('slow-correct', 'correct'), ('slow-shuffled', 'shuffled'))


def pair_binding(seed, count):
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        v0 = int(y[i])
        values = {0: v0, 1: 1 - v0}
        key_order = rng.permutation(2)
        positions = (int(rng.integers(1, 3)), int(rng.integers(4, 7)))
        for position, key in zip(positions, key_order):
            key = int(key)
            x[i, position, key] += 1.5
            x[i, position, 2 + values[key]] += 1.5
    perm = rng.permutation(count)
    return x[perm], y[perm].astype(np.int32)


def strict_binding(seed, count):
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        query = int(rng.integers(2))
        a = int(y[i]) ^ query
        values = (a, 1 - a)
        key_order = rng.permutation(2)
        positions = (int(rng.integers(1, 3)), int(rng.integers(4, 7)))
        for position, key in zip(positions, key_order):
            key = int(key)
            x[i, position, key] += 1.5
            x[i, position, 2 + values[key]] += 1.5
        x[i, 10, 4 + query] += 1.5
        x[i, 11, 7] += 1
    perm = rng.permutation(count)
    return x[perm], y[perm].astype(np.int32)


GENERATORS = {
    'delayed_order': lambda seed, count: order_sequences(seed, count),
    'pair_binding': pair_binding,
    'strict_binding': strict_binding,
}


def train_and_score(seed, condition, train, train_y, dev, dev_y, calibration):
    model = SlowTraceEligibility(g1_config(seed), condition, 'slow')
    before = model.core.w_in.copy()
    model.calibrate(calibration)
    for _epoch in range(2):
        for row, label in zip(train, train_y):
            model.learn(row, int(label))
    predicted = [int(model.predict(row)) for row in dev]
    accuracy = float(np.mean(np.asarray(predicted) == dev_y))
    return {
        'accuracy': accuracy,
        'hidden_changed': bool(np.any(model.core.w_in != before)),
        'predictions': predicted,
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    process = psutil.Process()
    peak = process.memory_info().rss
    calibration, _ignored = order_sequences(81003, 128)
    report = {'tasks': {}, 'threads': {k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
    for task, generate in GENERATORS.items():
        train, train_y = generate(81002, 64)
        dev, dev_y = generate(81004, 64)
        rows = []
        for seed in SEEDS:
            row = {'seed': seed}
            for name, condition in ARMS:
                scored = train_and_score(seed, condition, train, train_y, dev, dev_y, calibration)
                row[name] = scored['accuracy']
                row[name + '_hidden_changed'] = scored['hidden_changed']
                row[name + '_predictions'] = scored['predictions']
                peak = max(peak, process.memory_info().rss)
            rows.append(row)
        frozen = [row['slow-frozen'] for row in rows]
        correct = [row['slow-correct'] for row in rows]
        shuffled = [row['slow-shuffled'] for row in rows]
        frozen_mean = float(np.mean(frozen))
        report['tasks'][task] = {
            'seeds': rows,
            'development_labels': dev_y.astype(int).tolist(),
            'frozen_mean': frozen_mean,
            'correct_mean': float(np.mean(correct)),
            'shuffled_mean': float(np.mean(shuffled)),
            'precondition_frozen_below_0_65': frozen_mean < 0.65,
            'data_sha256': hashlib.sha256(train.tobytes() + dev.tobytes()).hexdigest(),
        }
    report['peak_rss_bytes'] = peak
    report['child_processes'] = 0
    OUT.mkdir(parents=True)
    (OUT / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = {
        task: {k: report['tasks'][task][k] for k in ('frozen_mean', 'correct_mean', 'shuffled_mean', 'precondition_frozen_below_0_65')}
        for task in report['tasks']
    }
    print(json.dumps({'peak_rss_bytes': peak, 'tasks': summary}, indent=2))


if __name__ == '__main__':
    main()
