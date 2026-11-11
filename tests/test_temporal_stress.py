"""Metamorphic checks. A test that only compares a function with itself is not enough."""
import numpy as np

from scarx.evaluation.tasks21 import ridge_probe
from scarx.evaluation.temporal_stress import (
    NormHead,
    core_features,
    make_core,
    rollout,
    sim_xor,
)


def test_rollout_matches_spiking_core_step():
    core = make_core(94001)
    sequences = np.random.default_rng(4).normal(0, 0.2, (3, 12, 8)).astype(np.float32)
    reference = np.stack([core_features(core, sequence) for sequence in sequences])
    mine, _state = rollout(core, sequences)
    assert np.max(np.abs(reference - mine["decay025"])) < 1e-4


def test_resume_matches_uninterrupted_state():
    core = make_core(94001)
    sequences = np.random.default_rng(5).normal(0, 0.2, (2, 12, 8)).astype(np.float32)
    _built, (left_v, left_a, left_z) = rollout(core, sequences[:, :7])
    _built, (voltage, adaptation, spikes) = rollout(core, sequences[:, 7:], (left_v, left_a, left_z))
    _whole, (once_v, once_a, once_z) = rollout(core, sequences)
    assert np.max(np.abs(voltage - once_v)) < 1e-4
    assert np.max(np.abs(adaptation - once_a)) < 1e-4
    assert np.max(np.abs(spikes - once_z)) < 1e-4


def test_ridge_falls_when_labels_are_permuted():
    rng = np.random.default_rng(8)
    train = rng.normal(0, 0.1, (80, 4)).astype(np.float32)
    labels = np.arange(80) % 2
    train[:, 0] += 4 * labels
    test = rng.normal(0, 0.1, (80, 4)).astype(np.float32)
    test_y = np.arange(80) % 2
    test[:, 0] += 4 * test_y
    intact = ridge_probe(train, labels, test, test_y)
    broken = ridge_probe(train, rng.permutation(labels), test, test_y)
    assert intact > 0.80
    assert broken < intact - 0.20


def test_ridge_falls_when_test_columns_are_permuted():
    rng = np.random.default_rng(9)
    train = rng.normal(0, 1, (60, 4)).astype(np.float32)
    labels = np.arange(60) % 2
    train[:, 0] += labels
    test = rng.normal(0, 1, (60, 4)).astype(np.float32)
    test_y = np.arange(60) % 2
    test[:, 0] += test_y
    intact = ridge_probe(train, labels, test, test_y)
    shuffled = ridge_probe(train, labels, test[:, [2, 0, 3, 1]], test_y)
    assert shuffled < intact - 0.15


def test_predict_does_not_update_calibration_and_leaky_observer_is_caught():
    head = NormHead(4, 3)
    train = np.ones((5, 4), np.float32)
    for row in train:
        head.observe(row)
    head.freeze()
    before = head.mean.copy()
    head.transform(np.full(4, 9, np.float32))
    assert np.array_equal(before, head.mean)
    try:
        head.observe(np.full(4, 9, np.float32))
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    leaky = NormHead(4, 4)
    for row in train:
        leaky.observe(row)
    leaky.observe(np.full(4, 9, np.float32))
    assert not np.array_equal(before, leaky.mean)


def test_hidden_weights_survive_rollout():
    core = make_core(94011)
    before = core.w_in.copy()
    sequences, _labels = sim_xor(11, 4)
    rollout(core, sequences)
    assert np.array_equal(before, core.w_in)
