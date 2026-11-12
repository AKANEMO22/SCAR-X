"""Unit tests for SCAR-X Round 9 benchmark integrity, generators, and gate evaluator."""
from __future__ import annotations

import numpy as np
import pytest

from scarx.evaluation.r9_benchmark_integrity import (
    assert_input_dimension,
    assert_spikes_binary,
    generate_strict_binding,
    generate_distractor_binding,
    generate_xor_representation,
    generate_order_representation,
    generate_ceiling_stress_binding,
    generate_multikey_associative_r9,
)
from scarx.evaluation.r9_gate_evaluator import (
    R9GateEvaluator,
    compute_paired_ci95,
    benjamini_hochberg,
)


def test_r9_benchmark_invariants():
    # Input dimension D=8 strictly
    x, y = generate_strict_binding(101, 32)
    assert x.shape == (32, 12, 8)
    assert y.shape == (32,)
    assert_input_dimension(x, 8)

    # Distractor binding D=8 strictly
    xd, yd = generate_distractor_binding(102, 32, distractor_amp=1.5)
    assert xd.shape == (32, 12, 8)
    assert_input_dimension(xd, 8)

    # Ceiling stress binding D=8 strictly with extended delay
    xcs, ycs = generate_ceiling_stress_binding(103, 32, delay_ticks=16, distractor_amp=2.0)
    assert xcs.shape == (32, 22, 8)
    assert_input_dimension(xcs, 8)

    # Scaled 16-key associative retrieval D=8 strictly
    x16, y16 = generate_multikey_associative_r9(104, 32, n_keys=16, delay=4)
    assert x16.shape[2] == 8
    assert_input_dimension(x16, 8)

    # Spikes binary assertion
    spikes_valid = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    assert_spikes_binary(spikes_valid)

    spikes_invalid = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    with pytest.raises(AssertionError):
        assert_spikes_binary(spikes_invalid)


def test_r9_gate_evaluator_strict_causal():
    evaluator = R9GateEvaluator({})

    # High direct accuracy but causal drop < 5pp must FAIL (no bypass!)
    task_summary_bypass_attempt = {
        "learned_candidate": {
            "direct_acc_mean": 0.98,
            "probe_acc_mean": 0.99,
            "probe_gain_lower95": 0.06,
        },
        "frozen_hidden": {"probe_acc_mean": 0.93},
        "shuffled_signal": {"probe_acc_mean": 0.85},
        "causal_revert": {"drop_mean": 0.02},  # Only 2pp drop!
    }
    res = evaluator.evaluate_task_results("bypass_task", task_summary_bypass_attempt)
    assert res["status"] == "FAIL"
    assert "causal_drop_gte_5pp" in res["failed_checks"]
    assert res["checks"]["causal_drop_gte_5pp"]["bypass_applied"] is False

    # Valid candidate with direct >= 65%, gain >= 5pp, drop >= 5pp
    task_summary_valid = {
        "learned_candidate": {
            "direct_acc_mean": 0.72,
            "probe_acc_mean": 0.75,
            "probe_gain_lower95": 0.04,
        },
        "frozen_hidden": {"probe_acc_mean": 0.62},
        "shuffled_signal": {"probe_acc_mean": 0.55},
        "causal_revert": {"drop_mean": 0.08},
    }
    res_valid = evaluator.evaluate_task_results("valid_task", task_summary_valid)
    assert res_valid["status"] == "PASS"


def test_benjamini_hochberg():
    p_vals = [0.001, 0.01, 0.03, 0.25, 0.80]
    sig = benjamini_hochberg(p_vals, alpha=0.05)
    assert sig[0] is True
    assert sig[1] is True
    assert sig[2] is True
    assert sig[3] is False
    assert sig[4] is False
