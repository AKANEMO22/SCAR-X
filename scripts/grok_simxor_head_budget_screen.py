"""New screen. Does not revise runs/grok-simxor-norm-head-v1.

The 2x64 screen left the normalized head at chance, so its error was not a
teaching signal. This run first fits that head with hidden synapses frozen.
Only a head that already clears 0.65 can support a learning claim.

Locked before the run:
- sim_xor, hard-threshold SpikingCore.step, feature_decay 0.25, H=48.
- Head: existing normalized ConditionedReadout.
- Seeds 92011, 92017, 92023. Train 256, dev 256, ridge fit 256, calibration 128.
- Phase 1: 4 epochs, hidden frozen. Precondition is mean online accuracy >= 0.65.
- Phase 2 starts from that head and the original hidden weights. 2 further
  epochs of frozen, correct, shuffled, zero, and scalar.
- correct is W_out.T @ error times e_f. scalar is one advantage times e_f.
- A learning pass requires the precondition, repository bootstrap lower95 of
  phase2 correct-minus-frozen >= 0.05, and correct above shuffled and zero.
- Ridge of decay-0.25 features and of the spike sum are recorded, not gates.
- No threshold change after scores. Confirmation is not opened.
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
OUT = ROOT / 'runs' / 'grok-simxor-head-budget-v1'
SEEDS = (92011, 92017, 92023)
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


def copy_head(src, dst):
    dst.head.weights[:] = src.head.weights
    dst.head.bias[:] = src.head.bias
    dst.head.mean[:] = src.head.mean
    dst.head.m2[:] = src.head.m2
    dst.head.count = src.head.count
    dst.head.frozen = src.head.frozen
    dst.head.updates = src.head.updates
    dst.core.updates = src.core.updates


def run_epochs(model, sequences, labels, epochs):
    losses = []
    for _epoch in range(epochs):
        for sequence, label in zip(sequences, labels):
            losses.append(model.learn(sequence, int(label)))
    return losses


def online_accuracy(model, sequences, labels):
    predicted = [int(model.predict(sequence)) for sequence in sequences]
    return predicted, float(np.mean(np.asarray(predicted) == labels))


def spike_sum(model, sequences):
    core = model.core
    config = core.config
    values = np.asarray(sequences, np.float32)
    batch, ticks, _width = values.shape
    voltage = np.zeros((batch, config.hidden), np.float32)
    adaptation = np.zeros_like(voltage)
    spikes = np.zeros_like(voltage)
    total = np.zeros_like(voltage)
    for step in range(ticks):
        voltage = (config.leak * voltage + values[:, step] @ core.w_in.T
                   + np.sum(core.w_rec[None, :, :] * spikes[:, core.indices], axis=2))
        limit = config.threshold + config.adaptation_strength * adaptation
        spikes = (voltage >= limit).astype(np.float32)
        voltage = voltage - spikes * limit
        adaptation = config.adaptation_decay * adaptation + spikes
        total += spikes
    return total


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    process = psutil.Process()
    peak = process.memory_info().rss
    calibration, _unused = sim_xor(92003, 128)
    train, train_y = sim_xor(92002, 256)
    dev, dev_y = sim_xor(92004, 256)
    ridge_x, ridge_y = sim_xor(92006, 256)
    rows = []
    for seed in SEEDS:
        phase1 = make_model(seed, 'frozen')
        phase1.calibrate(list(calibration))
        ridge_decay = ridge_probe(
            phase1.features_many(ridge_x), ridge_y, phase1.features_many(dev), dev_y)
        ridge_spikes = ridge_probe(
            spike_sum(phase1, ridge_x), ridge_y, spike_sum(phase1, dev), dev_y)
        phase1_losses = run_epochs(phase1, train, train_y, 4)
        phase1_predictions, phase1_accuracy = online_accuracy(phase1, dev, dev_y)
        row = {
            'seed': seed,
            'ridge_decay025_before': ridge_decay,
            'ridge_spike_sum_before': ridge_spikes,
            'phase1_frozen': phase1_accuracy,
            'phase1_mean_loss': float(np.mean(phase1_losses)),
            'phase1_predictions': phase1_predictions,
        }
        for arm in ('frozen', 'correct', 'shuffled', 'zero', 'scalar'):
            model = make_model(seed, arm)
            model.calibrate(list(calibration))
            copy_head(phase1, model)
            before = model.core.w_in.copy()
            losses = run_epochs(model, train, train_y, 2)
            predicted, accuracy = online_accuracy(model, dev, dev_y)
            row[arm] = accuracy
            row[arm + '_hidden_changed'] = bool(np.any(model.core.w_in != before))
            row[arm + '_hidden_updates'] = int(model.hidden_updates)
            row[arm + '_mean_loss'] = float(np.mean(losses))
            row[arm + '_ridge_after'] = ridge_probe(
                model.features_many(ridge_x), ridge_y, model.features_many(dev), dev_y)
            row[arm + '_predictions'] = predicted
            peak = max(peak, process.memory_info().rss)
        rows.append(row)
    phase1_scores = [row['phase1_frozen'] for row in rows]
    frozen = [row['frozen'] for row in rows]
    correct = [row['correct'] for row in rows]
    shuffled = [row['shuffled'] for row in rows]
    zero = [row['zero'] for row in rows]
    scalar = [row['scalar'] for row in rows]
    contrast = interval([a - b for a, b in zip(correct, frozen)])
    phase1_mean = float(np.mean(phase1_scores))
    precondition = phase1_mean >= 0.65
    screen_pass = bool(
        precondition
        and contrast['lower95'] >= 0.05
        and float(np.mean(correct)) > float(np.mean(shuffled))
        and float(np.mean(correct)) > float(np.mean(zero)))
    info = process.memory_info()
    report = {
        'task': 'sim_xor',
        'replaces_result': False,
        'previous_screen': 'runs/grok-simxor-norm-head-v1',
        'feature_decay': 0.25,
        'spike': 'hard-threshold SpikingCore.step',
        'head': 'ConditionedReadout normalized',
        'phase1_epochs': 4,
        'phase2_epochs': 2,
        'train_count': 256,
        'interval': 'phase21.interval bootstrap 10000 rng 2101',
        'learning_gate': 'phase1 frozen mean>=0.65 and phase2 correct-frozen lower95>=0.05 and correct above shuffled and zero',
        'confirmation_opened': False,
        'seeds': rows,
        'development_labels': dev_y.astype(int).tolist(),
        'phase1_frozen_mean': phase1_mean,
        'ridge_decay025_before_mean': float(np.mean([row['ridge_decay025_before'] for row in rows])),
        'ridge_spike_sum_before_mean': float(np.mean([row['ridge_spike_sum_before'] for row in rows])),
        'frozen_mean': float(np.mean(frozen)),
        'correct_mean': float(np.mean(correct)),
        'shuffled_mean': float(np.mean(shuffled)),
        'zero_mean': float(np.mean(zero)),
        'scalar_mean': float(np.mean(scalar)),
        'correct_minus_frozen': contrast,
        'precondition_phase1_frozen_at_least_0_65': precondition,
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
        'phase1_frozen_mean': phase1_mean,
        'ridge_decay025_before_mean': report['ridge_decay025_before_mean'],
        'ridge_spike_sum_before_mean': report['ridge_spike_sum_before_mean'],
        'frozen_mean': report['frozen_mean'],
        'correct_mean': report['correct_mean'],
        'shuffled_mean': report['shuffled_mean'],
        'zero_mean': report['zero_mean'],
        'scalar_mean': report['scalar_mean'],
        'correct_minus_frozen_lower95': contrast['lower95'],
        'precondition_phase1_frozen_at_least_0_65': precondition,
        'screen_pass': screen_pass,
        'peak_rss_bytes_sampled': report['peak_rss_bytes_sampled'],
    }, indent=2))


if __name__ == '__main__':
    main()
