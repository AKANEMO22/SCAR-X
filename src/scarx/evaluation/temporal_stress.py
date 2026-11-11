"""Independent ALIF window features for the temporal-stress audit.

The step matches SpikingCore. Hidden weights are read and never written.
"""
import numpy as np

from ..neural.core import Config, SpikingCore
from .tasks21 import ridge_probe


BANK = (0.10, 0.25, 0.50, 0.75, 0.90)
PULSE = 1.5


def g1_config(seed):
    return Config(
        inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=int(seed),
        leak=0.65, threshold=0.5, feature_decay=0.25, adaptation_strength=0.02,
        adaptation_decay=0.95, readout_lr=0.025, anchor_strength=0.0, neuron="alif",
    )


def make_core(seed):
    return SpikingCore(g1_config(seed))


def rollout(core, sequences, state=None):
    """Windowed substrates. Incoming state affects spikes. Feature sums start at zero."""
    config = core.config
    values = np.asarray(sequences, np.float32)
    single = values.ndim == 2
    if single:
        values = values[None, :, :]
    batch, ticks, width = values.shape
    if width != config.inputs:
        raise ValueError("Input width")
    if state is None:
        voltage = np.zeros((batch, config.hidden), np.float32)
        adaptation = np.zeros_like(voltage)
        spikes = np.zeros_like(voltage)
    else:
        voltage, adaptation, spikes = (np.array(part, copy=True) for part in state)
        if voltage.ndim == 1:
            voltage = voltage[None, :]
            adaptation = adaptation[None, :]
            spikes = spikes[None, :]
        if voltage.shape != (batch, config.hidden):
            raise ValueError("State batch")
    features = np.zeros_like(voltage)
    total = np.zeros_like(voltage)
    banks = [np.zeros_like(voltage) for _ in BANK]
    for step in range(ticks):
        current = values[:, step] @ core.w_in.T + np.sum(
            core.w_rec[None, :, :] * spikes[:, core.indices], axis=2)
        voltage = config.leak * voltage + current
        limit = config.threshold + config.adaptation_strength * adaptation
        spikes = (voltage >= limit).astype(np.float32)
        voltage = voltage - spikes * limit
        adaptation = config.adaptation_decay * adaptation + spikes
        features = config.feature_decay * features + (1.0 - config.feature_decay) * spikes
        total += spikes
        for index, decay in enumerate(BANK):
            banks[index] = decay * banks[index] + (1.0 - decay) * spikes
    substrates = {
        "decay025": features,
        "spikesum": total,
        "bank5": np.concatenate(banks, axis=1),
    }
    if single:
        substrates = {name: value[0] for name, value in substrates.items()}
        voltage, adaptation, spikes = voltage[0], adaptation[0], spikes[0]
    return substrates, (voltage, adaptation, spikes)


def core_features(core, sequence):
    core.reset_state()
    for row in np.asarray(sequence, np.float32):
        core.step(row, track=False)
    return core.features.copy()


class RawHead:
    """Serving delta rule. No feature normalization."""

    def __init__(self, width, seed):
        rng = np.random.default_rng(int(seed) + 17)
        self.weights = rng.normal(0, 0.03, (2, width)).astype(np.float32)
        self.bias = np.zeros(2, np.float32)

    def observe(self, features):
        return None

    def freeze(self):
        return None

    def learn(self, features, target):
        probabilities = self._probabilities(features)
        error = -probabilities
        error[int(target)] += 1.0
        self.weights += np.clip(0.025 * error[:, None] * features[None, :], -0.025, 0.025)
        self.bias += np.clip(0.025 * error, -0.025, 0.025)

    def predict(self, features):
        return int(np.argmax(self._probabilities(features)))

    def _probabilities(self, features):
        logits = self.weights @ np.asarray(features, np.float32) + self.bias
        logits = logits - np.max(logits)
        exponent = np.exp(logits)
        return exponent / np.sum(exponent)


class NormHead:
    """Z-scored delta rule. Statistics freeze before evaluation."""

    def __init__(self, width, seed):
        rng = np.random.default_rng(int(seed) + 991)
        self.weights = rng.normal(0, 0.03, (2, width)).astype(np.float32)
        self.bias = np.zeros(2, np.float32)
        self.mean = np.zeros(width, np.float64)
        self.second = np.zeros(width, np.float64)
        self.count = 0
        self.frozen = False

    def observe(self, features):
        if self.frozen:
            raise RuntimeError("Calibration is frozen")
        values = np.asarray(features, np.float64)
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.second += delta * (values - self.mean)

    def freeze(self):
        if self.count < 2:
            raise RuntimeError("Need calibration samples")
        self.frozen = True

    def transform(self, features):
        if not self.frozen:
            raise RuntimeError("Freeze calibration before use")
        scale = np.maximum(np.sqrt(self.second / (self.count - 1)), 0.05).astype(np.float32)
        return np.clip((np.asarray(features, np.float32) - self.mean.astype(np.float32)) / scale, -5, 5)

    def learn(self, features, target):
        transformed = self.transform(features)
        probabilities = self._probabilities(transformed)
        error = -probabilities
        error[int(target)] += 1.0
        self.weights += np.clip(0.08 * error[:, None] * transformed[None, :], -0.025, 0.025)
        self.bias += np.clip(0.08 * error, -0.025, 0.025)
        np.clip(self.weights, -3, 3, out=self.weights)
        np.clip(self.bias, -3, 3, out=self.bias)

    def predict(self, features):
        return int(np.argmax(self._probabilities(self.transform(features))))

    def _probabilities(self, transformed):
        logits = self.weights @ transformed + self.bias
        logits = logits - np.max(logits)
        exponent = np.exp(logits)
        return exponent / np.sum(exponent)


def fit_reader(kind, features, labels, seed):
    head = RawHead(features.shape[1], seed) if kind == "raw" else NormHead(features.shape[1], seed)
    for row in features:
        head.observe(row)
    head.freeze()
    for row, label in zip(features, labels):
        head.learn(row, int(label))
    return head


def accuracy(head, features, labels):
    predicted = np.asarray([head.predict(row) for row in features], np.int32)
    return float(np.mean(predicted == np.asarray(labels))), predicted


def paired_flip(head, first, first_y, second, second_y):
    correct = 0
    for left, left_y, right, right_y in zip(first, first_y, second, second_y):
        if head.predict(left) == int(left_y) and head.predict(right) == int(right_y):
            correct += 1
    return correct / len(first_y)


def sim_xor(seed, count):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    labels = rng.integers(0, 2, count).astype(np.int32)
    for index in range(count):
        first = int(rng.integers(2))
        second = first ^ int(labels[index])
        tick = int(rng.integers(1, 4))
        values[index, tick, 0] += PULSE * first
        values[index, tick, 1] += PULSE * second
        values[index, 10, 7] += PULSE
    return values, labels


def order_pairs(seed, count):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    early = noise.copy()
    late = noise.copy()
    early[:, 2, 0] += PULSE
    early[:, 6, 1] += PULSE
    late[:, 6, 0] += PULSE
    late[:, 2, 1] += PULSE
    early[:, 11, 5] += PULSE
    late[:, 11, 5] += PULSE
    labels_early = np.ones(count, np.int32)
    labels_late = np.zeros(count, np.int32)
    return early, labels_early, late, labels_late


def delay_set(seed, count, query_tick):
    rng = np.random.default_rng(seed + 1000 * int(query_tick))
    values = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    labels = np.arange(count) % 2
    for index in range(count):
        values[index, 1, int(labels[index])] += PULSE
        values[index, int(query_tick), 5] += PULSE
    order = rng.permutation(count)
    return values[order], labels[order].astype(np.int32)


def order_with_distractors(seed, count, extras):
    rng = np.random.default_rng(seed + 50 + int(extras))
    values = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    labels = np.arange(count) % 2
    spots = (4, 5)
    for index in range(count):
        if labels[index] == 1:
            values[index, 2, 0] += PULSE
            values[index, 7, 1] += PULSE
        else:
            values[index, 7, 0] += PULSE
            values[index, 2, 1] += PULSE
        for extra in range(int(extras)):
            values[index, spots[extra], 4 + extra] += PULSE
        values[index, 11, 5] += PULSE
    order = rng.permutation(count)
    return values[order], labels[order].astype(np.int32)


def query_set(seed, count):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    labels = np.zeros(count, np.int32)
    first_key = np.zeros(count, np.int32)
    for index in range(count):
        query = index % 2
        stored = int(rng.integers(2))
        pair = (stored, 1 - stored)
        first_key[index] = pair[0]
        labels[index] = pair[query]
        order = rng.permutation(2)
        for place, key in zip((2, 6), order):
            key = int(key)
            values[index, place, key] += PULSE
            values[index, place, 2 + pair[key]] += PULSE
        values[index, 10, 4 + query] += PULSE
    order = rng.permutation(count)
    return values[order], labels[order].astype(np.int32), first_key[order].astype(np.int32)


def length_set(seed, count, length, pad_start):
    rng = np.random.default_rng(seed + 17 * int(length) + int(pad_start))
    labels = np.arange(count) % 2
    values = np.zeros((count, 16, 8), np.float32)
    for index in range(count):
        body = rng.normal(0, 0.08, (length, 8)).astype(np.float32)
        early = 1 + int(rng.integers(0, max(1, length // 3)))
        late = min(length - 2, (2 * length) // 3 + int(rng.integers(0, max(1, length // 6))))
        if late <= early:
            late = min(length - 2, early + 1)
        if labels[index] == 1:
            body[early, 0] += PULSE
            body[late, 1] += PULSE
        else:
            body[late, 0] += PULSE
            body[early, 1] += PULSE
        body[length - 1, 5] += PULSE
        if pad_start:
            values[index, 16 - length:] = body
        else:
            values[index, :length] = body
    return values, labels.astype(np.int32)


def noise_gap(seed, index):
    rng = np.random.default_rng(seed + 10007 * int(index))
    gap = rng.normal(0, 0.08, (4, 8)).astype(np.float32)
    gap[1, 6] += PULSE
    return gap


def stream_episodes(seed, count):
    tasks = ("sim_xor", "order", "delay")
    buckets = {name: [] for name in tasks}
    cursor = {name: 0 for name in tasks}
    early, early_y, late, late_y = order_pairs(seed + 12, count)
    order_x = np.concatenate((early, late), axis=0)
    order_y = np.concatenate((early_y, late_y), axis=0)
    mix = np.random.default_rng(seed + 12).permutation(len(order_y))
    pools = {
        "sim_xor": sim_xor(seed + 11, count),
        "order": (order_x[mix][:count], order_y[mix][:count]),
        "delay": delay_set(seed + 13, count, 6),
    }
    sequence = []
    for index in range(count):
        name = tasks[index % 3]
        data, labels = pools[name]
        position = cursor[name]
        cursor[name] += 1
        sequence.append((name, data[position], int(labels[position])))
        buckets[name].append(index)
    return sequence


def ridge_on(train_x, train_y, test_x, test_y):
    return ridge_probe(train_x, train_y, test_x, test_y)
