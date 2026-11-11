"""Forensic cue/shift helpers and one safe readout. Hidden weights stay frozen.

R4A sources copied from 10e1c7df9a585566db04e754af9ba7b419d8bab1:
r4a_probe.py sha256 10766f166c8748baebf82a40c48ac4322760363f5ff6ece2689e8bd62194e6be
r4a_relational.py sha256 cc7e34c7b10fcfa193b523ca1094a9cb316b2f395e3f66dde7cf6f2f53d54a8c
"""
import numpy as np

from . import r4a_probe as probe
from . import r4a_relational as rel

PULSE = rel.PULSE
DEAD = 1e-4
CONF_SEEDS = (96001, 96007, 96013, 96019, 96023, 96029, 96031, 96037)


def cue_at(sequences, tick):
    values = np.asarray(sequences, np.float32)
    hot = (values[:, int(tick), 5] > 0.5).astype(np.float32)
    return np.stack((1.0 - hot, hot), axis=1)


def historical_shift(sequences):
    values = np.asarray(sequences, np.float32)
    out = np.zeros_like(values)
    for tick in (2, 6, 10):
        out[:, tick + 1, :] = values[:, tick, :]
    return out


def pure_shift(sequences):
    """Move only coordinates that carry a pulse. Background stays put."""
    values = np.asarray(sequences, np.float32)
    out = np.array(values, copy=True)
    for tick in (2, 6, 10):
        mask = values[:, tick, :] > 0.5
        out[:, tick, :] = np.where(mask, values[:, tick, :] - PULSE, values[:, tick, :])
        out[:, tick + 1, :] = out[:, tick + 1, :] + np.where(mask, PULSE, 0.0)
    return out.astype(np.float32)


def r5_features(seed, sequences, capture_tick, cue_tick):
    captured = probe.capture(seed, sequences, query_tick=int(capture_tick))
    state = probe.state_vector(captured, ("spikes", "voltage", "adaptation", "features"), "post")
    mapped = probe.bilinear_query_map(state, cue_at(sequences, cue_tick))
    return mapped, captured


def accuracy_of(seed, train_x, train_y, test_x, test_y, capture_tick, cue_tick):
    train_f, _cap = r5_features(seed, train_x, capture_tick, cue_tick)
    test_f, cap = r5_features(seed, test_x, capture_tick, cue_tick)
    head = probe.train_head(train_f, train_y, seed=seed)
    predicted = head.predict(test_f)
    labels = np.asarray(test_y)
    paired = predicted.reshape(-1, 2)
    paired_y = labels.reshape(-1, 2)
    valid = paired_y[:, 0] != paired_y[:, 1]
    consistency = float(np.mean(paired[valid, 0] != paired[valid, 1])) if np.any(valid) else 0.0
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "train_accuracy": head.acc(train_f, train_y),
        "consistency": consistency,
        "cue_key1_fraction": float(np.mean(cue_at(test_x, cue_tick)[:, 1])),
        "post_digest": hashlib_state(cap),
        "updates": int(len(train_y)),
    }


def hashlib_state(captured):
    import hashlib
    payload = captured["post"]["features"].tobytes()
    return hashlib.sha256(payload).hexdigest()


class SafeHead:
    """Same delta rule as the normalized head. Only the scale of live coordinates changes."""

    def __init__(self, width, seed):
        rng = np.random.default_rng(int(seed) + 991)
        self.weights = rng.normal(0, 0.03, (2, width)).astype(np.float32)
        self.bias = np.zeros(2, np.float32)
        self.mean = np.zeros(width, np.float64)
        self.second = np.zeros(width, np.float64)
        self.count = 0

    def observe(self, row):
        values = np.asarray(row, np.float64)
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.second += delta * (values - self.mean)

    def basis(self, row):
        values = np.asarray(row, np.float32)
        if self.count < 2:
            return np.clip(values, -5, 5)
        std = np.sqrt(self.second / (self.count - 1)).astype(np.float32)
        scaled = np.zeros_like(values)
        live = std > DEAD
        scaled[live] = (values[live] - self.mean.astype(np.float32)[live]) / std[live]
        return np.clip(scaled, -5, 5)

    def learn(self, row, target):
        basis = self.basis(row)
        logits = self.weights @ basis + self.bias
        logits = logits - np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)
        error = -probabilities
        error[int(target)] += 1.0
        self.weights += np.clip(0.08 * error[:, None] * basis[None, :], -0.025, 0.025)
        self.bias += np.clip(0.08 * error, -0.025, 0.025)
        np.clip(self.weights, -3, 3, out=self.weights)
        np.clip(self.bias, -3, 3, out=self.bias)

    def predict_many(self, rows):
        return np.asarray([int(np.argmax(self._probs(row))) for row in rows], np.int32)

    def _probs(self, row):
        basis = self.basis(row)
        logits = self.weights @ basis + self.bias
        logits = logits - np.max(logits)
        probabilities = np.exp(logits)
        return probabilities / np.sum(probabilities)


def fit_safe(features, labels, seed):
    head = SafeHead(features.shape[1], seed)
    for row in features:
        head.observe(row)
    for row, label in zip(features, labels):
        head.learn(row, int(label))
    return head
