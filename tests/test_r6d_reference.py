"""A bad normalizer must fail the train-only check."""
import numpy as np
import pytest

from scarx.evaluation.r6d_reference import assert_scale_uses_train_only, both_correct, train_only_mean


def test_bad_normalizer_fails_the_train_only_check():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (32, 4))
    test = rng.normal(5, 1, (32, 4))
    leaked = np.concatenate((train, test)).mean(axis=0)
    with pytest.raises(AssertionError):
        assert_scale_uses_train_only(leaked, train)


def test_train_only_mean_passes_its_own_check():
    train = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert_scale_uses_train_only(train_only_mean(train), train)


def test_both_correct_counts_pairs_not_rows():
    predicted = np.array([1, 0, 1, 1])
    labels = np.array([1, 0, 1, 0])
    differ = np.array([True, False])
    # First pair is both correct. Second pair is case B and is ignored.
    assert both_correct(predicted, labels, differ) == 1.0
    assert both_correct(np.array([1, 1, 1, 1]), labels, differ) == 0.0
