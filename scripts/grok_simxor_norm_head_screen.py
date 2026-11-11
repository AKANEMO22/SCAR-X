"""Locked screen: normalized online head, then per-neuron versus scalar.

Rules fixed before the run. No threshold change after scores.

Task: sim_xor. Two bits on one tick, label = XOR. Hard-threshold ALIF,
feature_decay 0.25, H=48. The spike step is SpikingCore.step. The head is the
existing normalized ConditionedReadout (learning rate 0.08, clip 0.025).

Arms, same head and same forward eligibility e_f:
- frozen: head learns, hidden does not
- correct: per-neuron W_out.T @ error times e_f
- shuffled: that signal permuted across neurons
- zero: no hidden update
- scalar: one advantage, clip(-loss - baseline, -2, 2), times every e_f row

Precondition, recorded and not retuned: mean online accuracy of frozen >= 0.65.
Only then may a learning claim be read. The claim passes this 3-seed screen
only if the repository bootstrap (phase21.interval, 10000 resamples, rng 2101)
of correct-minus-frozen has lower95 >= 0.05, and the correct mean is above
both shuffled and zero. Scalar is reported and is not a second pass rule.
A refit ridge before and after is recorded and is not the pass gate.
Confirmation is not opened here. Seeds 91011, 91017, 91023.
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

from scarx.evaluation.phase21 import interval
from scarx.evaluation.readout_eligibility import ReadoutEligibility
from scarx.evaluation.slow_trace import g1_config
from scarx.evaluation.tasks21 import ridge_probe


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs' / 'grok-simxor-norm-head-v1'
SEEDS = (91011, 91017, 91023)
S = 1.5


class ScalarForwardEligibility(ReadoutEligibility):
    """Same normalized head and e_f. One scalar times every synapse."""

    def __init__(self, config):
        super().__init__(config, 'correct')
        self.reward_mean = 0.0

    def accumulate(self, target):
        if (not self.segment_active or not self.steps_in_segment or not self.head.frozen
                or type(target) is not int or not 0 <= target < len(self.output_mask)
                or not self.output_mask[target]):
            raise ValueError('Calibrated supervised endpoint required')
        feature = self.core.features
        probabilities = self.head.probabilities(feature, self.output_mask)
        loss_value = -float(np.log(max(float(probabilities[target]), 1e-12)))
        advantage = float(np.clip(-loss_value - self.reward_mean, -2.0, 2.0))
        self.reward_mean = 0.95 * self.reward_mean + 0.05 * (-loss_value)
        self.signal_rng.permutation(self.core.config.hidden)
        self.bank += np.float32(advantage) * self.e_f
        if not np.isfinite(self.bank).all():
            raise ValueError('Nonfinite local direction')
        loss = self.head.learn(feature, target, self.output_mask)
        self.core.updates += 1
        self.endpoints += 1
        return loss


def sim_xor(seed, count):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    labels = rng.integers(0, 2, count).astype(np.int32)
    for i in range(count):
        first = int(rng.integers(2))
        second = first ^ int(labels[i])
        tick = int(rng.integers(1, 4))
        values[i, tick, 0] += S * first
        values[i, tick, 1] += S * second
        values[i, 10, 7] += S
    return values, labels


def make_model(seed, arm):
    config = g1_config(seed)
    if config.feature_decay != 0.25 or config.neuron != 'alif':
        raise ValueError('Locked hard-threshold decay')
    if arm == 'scalar':
        return ScalarForwardEligibility(config)
    return ReadoutEligibility(config, arm)


def train_and_score(seed, arm, train, train_y, dev, dev_y, calibration, ridge_x, ridge_y):
    model = make_model(seed, arm)
    before = model.core.w_in.copy()
    model.calibrate(list(calibration))
    losses = []
    for _epoch in range(2):
        for sequence, label in zip(train, train_y):
            losses.append(model.learn(sequence, int(label)))
    predicted = [int(model.predict(sequence)) for sequence in dev]
    accuracy = float(np.mean(np.asarray(predicted) == dev_y))
    ridge_after = ridge_probe(
        model.features_many(ridge_x), ridge_y, model.features_many(dev), dev_y)
    return {
        'accuracy': accuracy,
        'hidden_changed': bool(np.any(model.core.w_in != before)),
        'hidden_updates': int(model.hidden_updates),
        'mean_loss': float(np.mean(losses)),
        'ridge_after': ridge_after,
        'predictions': predicted,
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    process = psutil.Process()
    peak = process.memory_info().rss
    calibration, _unused = sim_xor(91003, 128)
    train, train_y = sim_xor(91002, 64)
    dev, dev_y = sim_xor(91004, 64)
    ridge_x, ridge_y = sim_xor(91006, 128)
    rows = []
    for seed in SEEDS:
        initial = make_model(seed, 'frozen')
        ridge_before = ridge_probe(
            initial.features_many(ridge_x), ridge_y, initial.features_many(dev), dev_y)
        row = {'seed': seed, 'ridge_before': ridge_before}
        for arm in ('frozen', 'correct', 'shuffled', 'zero', 'scalar'):
            scored = train_and_score(
                seed, arm, train, train_y, dev, dev_y, calibration, ridge_x, ridge_y)
            row[arm] = scored['accuracy']
            row[arm + '_hidden_changed'] = scored['hidden_changed']
            row[arm + '_hidden_updates'] = scored['hidden_updates']
            row[arm + '_mean_loss'] = scored['mean_loss']
            row[arm + '_ridge_after'] = scored['ridge_after']
            row[arm + '_predictions'] = scored['predictions']
            peak = max(peak, process.memory_info().rss)
        rows.append(row)
    frozen = [row['frozen'] for row in rows]
    correct = [row['correct'] for row in rows]
    shuffled = [row['shuffled'] for row in rows]
    zero = [row['zero'] for row in rows]
    scalar = [row['scalar'] for row in rows]
    gains = [a - b for a, b in zip(correct, frozen)]
    contrast = interval(gains)
    frozen_mean = float(np.mean(frozen))
    precondition = frozen_mean >= 0.65
    screen_pass = bool(
        precondition
        and contrast['lower95'] >= 0.05
        and float(np.mean(correct)) > float(np.mean(shuffled))
        and float(np.mean(correct)) > float(np.mean(zero)))
    info = process.memory_info()
    report = {
        'task': 'sim_xor',
        'feature_decay': 0.25,
        'spike': 'hard-threshold SpikingCore.step',
        'head': 'ConditionedReadout normalized',
        'interval': 'phase21.interval bootstrap 10000 rng 2101',
        'learning_gate': 'precondition frozen_mean>=0.65 and correct-frozen lower95>=0.05 and correct above shuffled and zero',
        'confirmation_opened': False,
        'seeds': rows,
        'development_labels': dev_y.astype(int).tolist(),
        'frozen_mean': frozen_mean,
        'correct_mean': float(np.mean(correct)),
        'shuffled_mean': float(np.mean(shuffled)),
        'zero_mean': float(np.mean(zero)),
        'scalar_mean': float(np.mean(scalar)),
        'correct_minus_frozen': contrast,
        'precondition_online_frozen_at_least_0_65': precondition,
        'screen_pass': screen_pass,
        'data_sha256': {
            'calibration': hashlib.sha256(calibration.tobytes()).hexdigest(),
            'train': hashlib.sha256(train.tobytes()).hexdigest(),
            'development': hashlib.sha256(dev.tobytes()).hexdigest(),
            'ridge_fit': hashlib.sha256(ridge_x.tobytes()).hexdigest(),
        },
        'threads': {name: os.environ[name] for name in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        'peak_rss_bytes_sampled': int(peak),
        'peak_wset_bytes': int(getattr(info, 'peak_wset', 0)),
    }
    OUT.mkdir(parents=True)
    (OUT / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'frozen_mean': report['frozen_mean'],
        'correct_mean': report['correct_mean'],
        'shuffled_mean': report['shuffled_mean'],
        'zero_mean': report['zero_mean'],
        'scalar_mean': report['scalar_mean'],
        'correct_minus_frozen_lower95': contrast['lower95'],
        'precondition_online_frozen_at_least_0_65': precondition,
        'screen_pass': screen_pass,
        'peak_rss_bytes_sampled': report['peak_rss_bytes_sampled'],
    }, indent=2))


if __name__ == '__main__':
    main()
