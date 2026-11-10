import numpy as np
import pytest

from scarx.evaluation.readout_eligibility import ReadoutEligibility
from scarx.evaluation.slow_trace import SlowTraceEligibility, g1_config, order_sequences


def calibrated(mode, condition='correct', seed=3):
    model = SlowTraceEligibility(g1_config(seed), condition, mode)
    values, _labels = order_sequences(71003, 128)
    model.calibrate(values)
    return model


def test_fast_forward_matches_the_existing_readout_eligibility():
    config = g1_config(5)
    left = SlowTraceEligibility(config, 'correct', 'fast')
    right = ReadoutEligibility(config, 'correct')
    values, _labels = order_sequences(71002, 4)
    for row in values:
        np.testing.assert_array_equal(left.features(row), right.features(row))


def test_prediction_and_frozen_learning_do_not_change_hidden_weights():
    model = calibrated('slow', 'frozen')
    before = model.core.w_in.copy()
    values, labels = order_sequences(71002, 1)
    assert model.predict(values[0]) in (0, 1)
    model.learn(values[0], int(labels[0]))
    np.testing.assert_array_equal(model.core.w_in, before)


def test_failed_update_restores_weights_trace_and_rng():
    model = calibrated('slow', 'correct')
    values, labels = order_sequences(71002, 1)
    before = model.dumps()
    weights = model.core.w_in.copy()
    slow = model.slow.copy()
    rng = model.signal_rng.bit_generator.state

    def fail():
        model.core.w_in += 1
        model.slow += 1
        raise RuntimeError('injected')

    model.finish = fail
    with pytest.raises(RuntimeError, match='injected'):
        model.learn(values[0], int(labels[0]))
    np.testing.assert_array_equal(model.core.w_in, weights)
    np.testing.assert_array_equal(model.slow, slow)
    assert model.signal_rng.bit_generator.state == rng
    assert model.dumps() == before


def test_checkpoint_roundtrip_and_foreign_format_rejected():
    model = calibrated('slow', 'correct', seed=9)
    values, _labels = order_sequences(71004, 2)
    model.slow[:] = 0.25
    model.e_s[:] = 0.125
    saved = model.dumps()
    restored = SlowTraceEligibility.loads(saved)
    np.testing.assert_array_equal(restored.slow, model.slow)
    np.testing.assert_array_equal(restored.e_s, model.e_s)
    np.testing.assert_array_equal(restored.core.w_in, model.core.w_in)
    assert restored.predict(values[0]) == model.predict(values[0])
    first = SlowTraceEligibility.loads(saved)
    second = SlowTraceEligibility.loads(saved)
    train, labels = order_sequences(71002, 1)
    assert first.learn(train[0], int(labels[0])) == second.learn(train[0], int(labels[0]))
    foreign = ReadoutEligibility(g1_config(9), 'correct').dumps()
    with pytest.raises(ValueError, match='members'):
        SlowTraceEligibility.loads(foreign)
