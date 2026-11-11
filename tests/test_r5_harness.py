"""The published cue reads tick 10. A shifted query lives at tick 11."""
import numpy as np

from scarx.evaluation import r4a_probe as probe
from scarx.evaluation import r4a_relational as rel
from scarx.evaluation.r5_audit import cue_at, historical_shift, pure_shift


def _pair():
    sequences, labels, _pid, _flag = rel.paired_counterfactual(7, 32)
    return sequences, labels


def test_historical_cue_is_always_key0_after_the_published_shift():
    sequences, _labels = _pair()
    shifted = historical_shift(sequences)
    buggy = cue_at(shifted, probe.QUERY_TICK)
    assert np.all(buggy[:, 0] == 1.0)
    assert np.all(buggy[:, 1] == 0.0)


def test_cue_at_the_shifted_tick_uses_both_keys():
    sequences, labels = _pair()
    shifted = historical_shift(sequences)
    cue = cue_at(shifted, 11)
    assert cue.shape == (len(labels), 2)
    assert np.mean(cue[:, 1]) == 0.5
    assert np.any(cue[:, 0] == 1.0) and np.any(cue[:, 1] == 1.0)


def test_cue_comes_from_the_input_not_the_label():
    sequences, labels = _pair()
    flipped = labels.copy()
    flipped[:] = 1 - flipped
    assert np.array_equal(cue_at(sequences, 10), cue_at(sequences, 10))
    other = sequences.copy()
    other[:, 10, 5] = np.where(labels == 1, 2.0, 0.0)
    assert not np.array_equal(cue_at(sequences, 10), cue_at(other, 10))


def test_siblings_differ_only_at_the_query_tick_and_labels_flip():
    sequences, labels, _pid, _flag = rel.paired_counterfactual(7, 16)
    left, right = sequences[0::2], sequences[1::2]
    gap = np.max(np.abs(left - right), axis=(0, 2))
    assert gap[10] > 1.0
    assert np.max(gap[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]]) == 0.0
    assert np.all(labels[0::2] != labels[1::2])


def test_published_shift_erases_background_and_pure_shift_does_not():
    sequences, _labels = _pair()
    published = historical_shift(sequences)
    pure = pure_shift(sequences)
    assert np.all(published[:, 0, :] == 0.0)
    assert np.max(np.abs(pure[:, 0, :] - sequences[:, 0, :])) == 0.0
    assert np.max(np.abs(pure[:, 11, :] - sequences[:, 11, :])) > 1.0


def test_capture_does_not_change_hidden_weights():
    sequences, _labels = _pair()
    _config, _idx, before, rec = probe.weights(3)
    probe.capture(3, historical_shift(sequences), query_tick=11)
    _config, _idx, after, rec2 = probe.weights(3)
    assert np.array_equal(before, after)
    assert np.array_equal(rec, rec2)
