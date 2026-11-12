"""Formal screen for protocol v1, with confirmation scored on split 71005.

Amendment 1 corrects the earlier run, which scored confirmation seeds on the
development split 71004. Seeds, budget and the 0.05 bound are unchanged.
The historical result directory is not rewritten.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import psutil

from scarx.evaluation.slow_trace import SlowTraceEligibility, g1_config, order_sequences
from scarx.evaluation.tasks21 import ridge_probe


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/collab/scarx-neural-round1-20261111/grok/protocol-v1.md'
OUT = ROOT / 'runs/grok-slow-trace-v1a-confirmation-71005'
HISTORICAL = ROOT / 'runs/grok-slow-trace-v1/result.json'
T_95_DF7 = 2.3646242515923784
ARMS = (
    ('fast-frozen', 'fast', 'frozen'),
    ('fast-correct', 'fast', 'correct'),
    ('slow-frozen', 'slow', 'frozen'),
    ('slow-correct', 'slow', 'correct'),
    ('slow-shuffled', 'slow', 'shuffled'),
    ('slow-zero', 'slow', 'zero'),
)
DEV_SEEDS = (71011, 71017, 71023)
CONFIRM_SEEDS = (71037, 71039, 71051, 71057, 71063, 71079, 71091, 71099)


def score_arm(model, values, labels):
    predicted = [int(model.predict(row)) for row in values]
    return float(np.mean(np.asarray(predicted) == labels)), predicted


def probe(model, values):
    fast, slow = [], []
    for row in values:
        features, trace, _adaptation = model.state_view(row)
        fast.append(features)
        slow.append(trace)
    return np.stack(fast), np.stack(slow)


def train_one(seed, mode, condition, train, labels):
    model = SlowTraceEligibility(g1_config(seed), condition, mode)
    calibration, _ignored = order_sequences(71003, 128)
    model.calibrate(calibration)
    for _epoch in range(2):
        for row, label in zip(train, labels):
            model.learn(row, int(label))
    return model


def mean(rows, arm):
    return float(np.mean([row[arm] for row in rows]))


def interval(gains):
    values = np.asarray(gains, np.float64)
    count = int(values.size)
    if count < 2:
        raise ValueError('Need at least two seed gains')
    center = float(values.mean())
    deviation = float(values.std(ddof=1))
    margin = T_95_DF7 * deviation / np.sqrt(count)
    return {
        'n': count,
        'mean': center,
        'sample_std_ddof1': deviation,
        't_975_df7': T_95_DF7,
        'lower': center - margin,
        'upper': center + margin,
        'gains': [float(item) for item in values],
    }


def screen_passes(rows):
    if any(row['slow-correct'] < 0.65 for row in rows):
        return False
    if mean(rows, 'slow-correct') - mean(rows, 'slow-frozen') <= 0:
        return False
    if mean(rows, 'slow-correct') - mean(rows, 'fast-correct') < 0.05:
        return False
    return True


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} already exists; refusing to overwrite a recorded run')
    if not HISTORICAL.is_file():
        raise FileNotFoundError('Historical development-scored result is missing')
    started_utc = datetime.now(timezone.utc).isoformat()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    peak_commit = getattr(process.memory_info(), 'private', process.memory_info().vms)
    started = time.perf_counter()
    protocol = PROTOCOL.read_bytes()
    train, train_labels = order_sequences(71002, 64)
    development, development_labels = order_sequences(71004, 64)
    confirmation_values, confirmation_labels = order_sequences(71005, 64)
    gate_model = SlowTraceEligibility(g1_config(71001), 'frozen', 'slow')
    fast_train, slow_train = probe(gate_model, train)
    fast_dev, slow_dev = probe(gate_model, development)
    fast_score = ridge_probe(fast_train, train_labels, fast_dev, development_labels)
    slow_score = ridge_probe(slow_train, train_labels, slow_dev, development_labels)
    if fast_score >= 0.65:
        gate = 'STOP_FAST_FEATURE_ALREADY_SOLVES_TASK'
    elif slow_score < 0.40:
        gate = 'STOP_SLOW_TRACE_HAS_NO_INFORMATION'
    else:
        gate = 'LEARNING_SCREEN_INTERPRETABLE'
    peak_rss = max(peak_rss, process.memory_info().rss)
    peak_commit = max(peak_commit, getattr(process.memory_info(), 'private', process.memory_info().vms))
    learning = []
    if gate == 'LEARNING_SCREEN_INTERPRETABLE':
        for seed in DEV_SEEDS:
            scores = {'seed': seed}
            for name, mode, condition in ARMS:
                model = train_one(seed, mode, condition, train, train_labels)
                accuracy_dev, predicted = score_arm(model, development, development_labels)
                accuracy_train, _train_predicted = score_arm(model, train, train_labels)
                scores[name] = accuracy_dev
                scores[name + '_train'] = accuracy_train
                scores[name + '_predictions'] = predicted
                info = process.memory_info()
                peak_rss = max(peak_rss, info.rss)
                peak_commit = max(peak_commit, getattr(info, 'private', info.vms))
            learning.append(scores)
    confirmation = []
    opened = bool(learning) and screen_passes(learning)
    if opened:
        for seed in CONFIRM_SEEDS:
            scores = {'seed': seed}
            for name, mode, condition in ARMS:
                model = train_one(seed, mode, condition, train, train_labels)
                accuracy_dev, predicted = score_arm(model, confirmation_values, confirmation_labels)
                scores[name] = accuracy_dev
                scores[name + '_predictions'] = predicted
            confirmation.append(scores)
    gain_versus_frozen = interval([row['slow-correct'] - row['slow-frozen'] for row in confirmation]) if confirmation else None
    gain_versus_fast = interval([row['slow-correct'] - row['fast-correct'] for row in confirmation]) if confirmation else None
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    high_water = getattr(process.memory_info(), 'peak_wset', None)
    result = {
        'amendment': 'v1a scores confirmation seeds on split 71005. It does not replace runs/grok-slow-trace-v1.',
        'historical_result': HISTORICAL.relative_to(ROOT).as_posix(),
        'command': [sys.executable, 'scripts/grok_slow_trace_screen.py'],
        'started_utc': started_utc,
        'source_commit': commit,
        'protocol_sha256': hashlib.sha256(protocol).hexdigest(),
        'protocol_commit_required_before_run': 'c709bbe01f97c38c84924a554bb787db801701bd',
        'threads': {name: os.environ[name] for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        'information_gate': {
            'fast_feature_development_probe': fast_score,
            'slow_trace_development_probe': slow_score,
            'decision': gate,
        },
        'development_labels': development_labels.astype(int).tolist(),
        'confirmation_split_seed': 71005,
        'confirmation_labels': confirmation_labels.astype(int).tolist(),
        'learning_screen': learning,
        'confirmation_opened_by_development_screen': opened,
        'confirmation': confirmation,
        'confirmation_gain_versus_slow_frozen': gain_versus_frozen,
        'confirmation_gain_versus_fast_correct': gain_versus_fast,
        'confirmation_lower_bound_passes_0_05': bool(gain_versus_frozen and gain_versus_fast and gain_versus_frozen['lower'] >= 0.05 and gain_versus_fast['lower'] >= 0.05),
        'peak_rss_bytes_sampled': peak_rss,
        'peak_private_bytes_sampled': peak_commit,
        'os_peak_working_set_bytes': None if high_water is None else int(high_water),
        'memory_note': 'os_peak_working_set_bytes is the Windows peak working set for this process. The sampled fields are only the maxima of the explicit samples.',
        'child_processes': 0,
        'seconds': time.perf_counter() - started,
        'data_sha256': {
            'train': hashlib.sha256(train.tobytes() + train_labels.tobytes()).hexdigest(),
            'development': hashlib.sha256(development.tobytes() + development_labels.tobytes()).hexdigest(),
            'confirmation': hashlib.sha256(confirmation_values.tobytes() + confirmation_labels.tobytes()).hexdigest(),
        },
        'exit_code': 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
