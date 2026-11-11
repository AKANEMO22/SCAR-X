"""Readout-only ablations. Copied from Grok R4B 8f200b988788dfb735a763fcbe085bf4546f731f.
Hidden ALIF weights are never updated.
"""
import numpy as np

from ..neural.core import Config, SpikingCore
from .tasks21 import ridge_probe


BANK = (0.10, 0.25, 0.50, 0.75, 0.90)
PULSE = 1.5
ARMS = ("norm", "raw", "noclip", "nofloor", "nonorm", "causal")


def g1_config(seed):
    return Config(
        inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=int(seed),
        leak=0.65, threshold=0.5, feature_decay=0.25, adaptation_strength=0.02,
        adaptation_decay=0.95, readout_lr=0.025, anchor_strength=0.0, neuron="alif",
    )


def make_core(seed):
    return SpikingCore(g1_config(seed))


def rollout(core, sequences):
    config = core.config
    values = np.asarray(sequences, np.float32)
    batch, ticks, width = values.shape
    if width != config.inputs:
        raise ValueError("Input width")
    voltage = np.zeros((batch, config.hidden), np.float32)
    adaptation = np.zeros_like(voltage)
    spikes = np.zeros_like(voltage)
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
    return {"decay025": features, "spikesum": total, "bank5": np.concatenate(banks, axis=1)}


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


def separated_xor(seed, count):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    labels = rng.integers(0, 2, count).astype(np.int32)
    for index in range(count):
        first = int(rng.integers(2))
        second = first ^ int(labels[index])
        values[index, 2, 0] += PULSE * first
        values[index, 6, 1] += PULSE * second
        values[index, 11, 5] += PULSE
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
    return early, np.ones(count, np.int32), late, np.zeros(count, np.int32)


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
    for index in range(count):
        if labels[index] == 1:
            values[index, 2, 0] += PULSE
            values[index, 7, 1] += PULSE
        else:
            values[index, 7, 0] += PULSE
            values[index, 2, 1] += PULSE
        for extra in range(int(extras)):
            values[index, (4, 5)[extra], 4 + extra] += PULSE
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
        key_order = rng.permutation(2)
        for place, key in zip((2, 6), key_order):
            key = int(key)
            values[index, place, key] += PULSE
            values[index, place, 2 + pair[key]] += PULSE
        values[index, 10, 4 + query] += PULSE
    order = rng.permutation(count)
    return values[order], labels[order].astype(np.int32), first_key[order].astype(np.int32)


def query_swap(sequences, labels):
    twin = np.array(sequences, copy=True)
    twin[:, 10, 4] = sequences[:, 10, 5]
    twin[:, 10, 5] = sequences[:, 10, 4]
    return twin, (1 - np.asarray(labels)).astype(np.int32)


def ridge_score(train_f, train_y, test_f, test_y):
    return ridge_probe(train_f, train_y, test_f, test_y)


def order_index(labels, kind):
    labels = np.asarray(labels)
    if kind == "original":
        return np.arange(len(labels))
    if kind == "perm":
        return np.random.default_rng(96001).permutation(len(labels))
    if kind == "balanced":
        zero = np.flatnonzero(labels == 0)
        one = np.flatnonzero(labels == 1)
        count = min(len(zero), len(one))
        mixed = []
        for index in range(count):
            mixed.extend((int(zero[index]), int(one[index])))
        mixed.extend(int(value) for value in zero[count:])
        mixed.extend(int(value) for value in one[count:])
        return np.asarray(mixed, np.int32)
    raise ValueError(kind)


class DeltaHead:
    """One local delta rule. `mode` changes a single named factor."""

    def __init__(self, width, seed, mode):
        if mode not in ARMS:
            raise ValueError(mode)
        rng = np.random.default_rng(int(seed) + 991)
        self.mode = mode
        self.weights = rng.normal(0, 0.03, (2, width)).astype(np.float32)
        self.bias = np.zeros(2, np.float32)
        self.mean = np.zeros(width, np.float64)
        self.second = np.zeros(width, np.float64)
        self.count = 0
        self.frozen = False
        self.lr = 0.025 if mode == "raw" else 0.08
        self.step_clip = None if mode == "noclip" else 0.025
        self.weight_clip = 20.0 if mode == "raw" else 3.0
        self.normalize = mode not in ("raw", "nonorm")
        self.causal = mode == "causal"
        self.floor = 0.0 if mode == "nofloor" else 0.05
        self.feature_clip = None if mode == "nofloor" else 5.0
        self.clip_hits = 0
        self.clip_coords = 0
        self.update_norm = 0.0
        self.updates = 0

    def observe(self, features):
        if self.frozen:
            raise RuntimeError("Calibration is frozen")
        values = np.asarray(features, np.float64)
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.second += delta * (values - self.mean)

    def basis(self, features):
        values = np.asarray(features, np.float32)
        if not self.normalize:
            return values
        if self.count < 2:
            scaled = values
        else:
            scale = np.sqrt(self.second / (self.count - 1))
            scale = np.maximum(scale, self.floor).astype(np.float32)
            scaled = (values - self.mean.astype(np.float32)) / scale
        if self.feature_clip is not None:
            scaled = np.clip(scaled, -self.feature_clip, self.feature_clip)
        return scaled.astype(np.float32)

    def probabilities(self, features):
        logits = self.weights @ self.basis(features) + self.bias
        logits = logits - np.max(logits)
        exponent = np.exp(logits)
        return exponent / np.sum(exponent)

    def learn(self, features, target):
        basis = self.basis(features)
        probabilities = self.probabilities(features)
        error = -probabilities
        error[int(target)] += 1.0
        raw = self.lr * error[:, None] * basis[None, :]
        if self.step_clip is None:
            step = raw
        else:
            step = np.clip(raw, -self.step_clip, self.step_clip)
            self.clip_hits += int(np.sum(np.abs(raw) > self.step_clip + 1e-12))
            self.clip_coords += raw.size
        self.weights += step
        bias_step = self.lr * error
        if self.step_clip is not None:
            bias_step = np.clip(bias_step, -self.step_clip, self.step_clip)
        self.bias += bias_step
        np.clip(self.weights, -self.weight_clip, self.weight_clip, out=self.weights)
        np.clip(self.bias, -self.weight_clip, self.weight_clip, out=self.bias)
        self.update_norm += float(np.linalg.norm(step))
        self.updates += 1
        if self.causal:
            self.observe(features)
        return -float(np.log(max(float(probabilities[int(target)]), 1e-12))), error, basis, raw

    def predict(self, features):
        return int(np.argmax(self.probabilities(features)))


def batch_loss(head, features, labels):
    losses = []
    correct = 0
    for row, label in zip(features, labels):
        probabilities = head.probabilities(row)
        losses.append(-np.log(max(float(probabilities[int(label)]), 1e-12)))
        correct += int(np.argmax(probabilities) == int(label))
    return float(np.mean(losses)), correct / len(labels)


def fit_arm(features, labels, seed, mode, order_kind="original"):
    head = DeltaHead(features.shape[1], seed, mode)
    order = order_index(labels, order_kind)
    if not head.causal and head.normalize:
        for index in order:
            head.observe(features[index])
        head.frozen = True
    train_hits = 0
    for index in order:
        head.learn(features[index], int(labels[index]))
        train_hits += int(head.predict(features[index]) == int(labels[index]))
    train_loss, train_accuracy = batch_loss(head, features, labels)
    return head, {
        "train_accuracy": train_accuracy,
        "train_loss": train_loss,
        "online_hit_rate": train_hits / len(labels),
        "clip_fraction": head.clip_hits / max(1, head.clip_coords),
        "mean_update_norm": head.update_norm / max(1, head.updates),
        "max_abs_weight": float(np.max(np.abs(head.weights))),
        "updates": head.updates,
    }


def evaluate(head, features, labels):
    loss, score = batch_loss(head, features, labels)
    return {"accuracy": score, "loss": loss}


def paired_flip(head, first, first_y, second, second_y):
    hits = 0
    for left, left_y, right, right_y in zip(first, first_y, second, second_y):
        hits += int(head.predict(left) == int(left_y) and head.predict(right) == int(right_y))
    return hits / len(first_y)


def geometry(features):
    values = np.asarray(features, np.float64)
    std = values.std(axis=0)
    live = std > 1e-4
    result = {
        "mean_abs": float(np.mean(np.abs(values))),
        "median_std": float(np.median(std)),
        "dead_fraction": float(np.mean(~live)),
        "sparsity": float(np.mean(np.abs(values) < 1e-3)),
        "effective_rank": None,
        "correlation_condition": None,
    }
    if int(live.sum()) >= 2:
        centered = values[:, live] - values[:, live].mean(axis=0)
        gram = centered.T @ centered / max(1, len(values) - 1)
        eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0, None)
        positive = eigenvalues[eigenvalues > 1e-8]
        result["effective_rank"] = float((eigenvalues.sum() ** 2) / (np.sum(eigenvalues ** 2) + 1e-12))
        if len(positive):
            result["correlation_condition"] = float(positive[-1] / positive[0])
    return result


def direction_error(seed=5):
    """Finite difference of cross-entropy versus the unclipped delta on a fixed basis."""
    rng = np.random.default_rng(seed)
    head = DeltaHead(6, seed, "nonorm")
    features = rng.normal(0, 1, 6).astype(np.float32)
    target = 1
    probabilities = head.probabilities(features)
    error = -probabilities
    error[target] += 1.0
    analytic = error[:, None] * features[None, :]
    numerical = np.zeros_like(head.weights)
    epsilon = 1e-4
    for row in range(2):
        for col in range(6):
            saved = float(head.weights[row, col])
            head.weights[row, col] = saved + epsilon
            plus = -np.log(max(float(head.probabilities(features)[target]), 1e-12))
            head.weights[row, col] = saved - epsilon
            minus = -np.log(max(float(head.probabilities(features)[target]), 1e-12))
            head.weights[row, col] = saved
            numerical[row, col] = (plus - minus) / (2 * epsilon)
    return float(np.max(np.abs(-numerical - analytic)))
