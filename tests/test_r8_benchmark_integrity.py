"""Unit tests verifying Round 8 Benchmark Integrity Assertions."""
import pytest
import numpy as np

from scarx.neural.core import Config, SpikingCore
from scarx.evaluation.r8_benchmark_integrity import (
    assert_input_dimension,
    assert_spikes_binary,
    assert_core_parameters,
    assert_matched_updates,
    generate_strict_binding,
    generate_distractor_binding,
    generate_xor_representation,
    generate_order_representation,
    generate_multikey_associative,
)


def test_input_dimension_assertion():
    valid_x = np.zeros((10, 12, 8), dtype=np.float32)
    assert_input_dimension(valid_x, 8)

    invalid_x = np.zeros((10, 12, 11), dtype=np.float32)
    with pytest.raises(AssertionError, match="Benchmark integrity violation: input dimension 11"):
        assert_input_dimension(invalid_x, 8)


def test_spikes_binary_assertion():
    valid_spikes = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    assert_spikes_binary(valid_spikes)

    invalid_spikes = np.array([0.0, 0.45, 1.0], dtype=np.float32)
    with pytest.raises(AssertionError, match="Spike outputs must be strictly binary"):
        assert_spikes_binary(invalid_spikes)


def test_core_parameters_assertion():
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, seed=123)
    core = SpikingCore(cfg)
    assert_core_parameters(core)


def test_matched_updates_assertion():
    assert_matched_updates(100, 100, "test")
    with pytest.raises(AssertionError, match="Update budget mismatch"):
        assert_matched_updates(100, 200, "test")


def test_clean_task_generators():
    for gen in (
        generate_strict_binding,
        generate_distractor_binding,
        generate_xor_representation,
        generate_order_representation,
    ):
        x, y = gen(seed=42, count=32)
        assert x.shape == (32, 12, 8)
        assert y.shape == (32,)
        assert set(np.unique(y)).issubset({0, 1})

    # Test multikey associative task
    for n_keys in (2, 4):
        x, y = generate_multikey_associative(seed=42, count=32, n_keys=n_keys, delay=4, distractor=True)
        assert x.shape[-1] == 8
        assert y.shape == (32,)
