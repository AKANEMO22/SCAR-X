"""Independent scoring. Does not import DeepSeek's gate function."""
import numpy as np


def both_correct(predicted, labels, differ):
    """Fraction of pairs whose two siblings are both correct.

    `differ` is one boolean per pair, not per row. Rows are stored as
    (query A, query B) within each pair.
    """
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    differ = np.asarray(differ, dtype=bool)
    left, right = predicted[0::2], predicted[1::2]
    ya, yb = labels[0::2], labels[1::2]
    if left.shape != ya.shape or len(differ) != len(left):
        raise ValueError("pair alignment")
    both = (left == ya) & (right == yb)
    if not np.any(differ):
        return float("nan")
    return float(np.mean(both[differ]))


def train_only_mean(train):
    return np.asarray(train, np.float64).mean(axis=0)


def assert_scale_uses_train_only(mean, train):
    if not np.allclose(mean, train_only_mean(train)):
        raise AssertionError("normalization saw rows other than the training block")


def seed_interval(values):
    """Bootstrap the mean of independent seeds. Not of rows inside a seed."""
    values = np.asarray(values, np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("need one number per seed")
    draws = np.random.default_rng(2101).choice(values, (10_000, len(values)), replace=True).mean(1)
    return {
        "mean": float(values.mean()),
        "lower95": float(np.quantile(draws, 0.025)),
        "upper95": float(np.quantile(draws, 0.975)),
        "n_seeds": int(len(values)),
    }
