"""Unit tests for the independent programmatic R8 gate evaluator."""
from __future__ import annotations

import pytest
from scarx.evaluation.r8_gate_evaluator import R8GateEvaluator


def test_r8_evaluator_strict_gates():
    # Synthetic data where P1 strict_binding passes, distractor fails, and P2 forgets
    raw = {
        "p1_results": {
            "strict_binding": {
                "learned_mtsg": {"direct_acc_mean": 0.80, "probe_acc_mean": 0.85, "probe_gain_lower95": 0.12},
                "frozen_hidden": {"probe_acc_mean": 0.65},
                "r35_baseline": {"probe_acc_mean": 0.68},
                "shuffled_signal": {"probe_acc_mean": 0.55},
                "causal_revert": {"drop_mean": 0.20},
            },
            "distractor_binding": {
                "learned_mtsg": {"direct_acc_mean": 0.62, "probe_acc_mean": 0.66, "probe_gain_lower95": 0.08},
                "frozen_hidden": {"probe_acc_mean": 0.54},
                "r35_baseline": {"probe_acc_mean": 0.54},
                "shuffled_signal": {"probe_acc_mean": 0.50},
                "causal_revert": {"drop_mean": 0.12},
            },
            "order": {
                "learned_mtsg": {"direct_acc_mean": 1.0, "probe_acc_mean": 0.999},
                "frozen_hidden": {"probe_acc_mean": 0.965},
                "r35_baseline": {"probe_acc_mean": 0.995},
                "shuffled_signal": {"probe_acc_mean": 0.990},
                "causal_revert": {"drop_mean": 0.035},
            },
        },
        "p2_results": {
            "bf_cascade": {
                "acquisition_mean": 0.85,
                "final_mean": 0.60,
                "mean_forgetting": 0.24,
            },
            "unconsolidated": {
                "mean_forgetting": 0.30,
            },
        },
        "resource_results": {
            "peak_rss_bytes": 40_000_000,
        },
    }

    evaluator = R8GateEvaluator(raw)
    report = evaluator.evaluate_full()

    # Overall verdict must be FAIL because distractor fails (0.62 < 0.65) and P2 forgets (0.24 > 0.03)
    assert report["overall_pass"] is False
    assert report["overall_status"] == "FAIL"

    # strict_binding passes
    sb = report["p1_tasks"]["strict_binding"]
    assert sb["pass"] is True
    assert sb["status"] == "PASS"

    # distractor_binding fails on direct accuracy
    db = report["p1_tasks"]["distractor_binding"]
    assert db["pass"] is False
    assert db["status"] == "FAIL"
    assert db["checks"]["direct_accuracy_gte_65"]["pass"] is False

    # order fails when causal revert drop < 5pp or shuffled contrast < 2pp (no 95% bypass loophole)
    order = report["p1_tasks"]["order"]
    assert order["pass"] is False
    assert order["status"] == "FAIL"
    assert order["checks"]["causal_drop_gte_5pp"]["pass"] is False
    assert order["checks"]["shuffled_control_distinct"]["pass"] is False

    # Verify that with genuine causal drop >= 5pp and shuffled distinct >= 2pp, order passes
    raw["p1_results"]["order"]["causal_revert"]["drop_mean"] = 0.08
    raw["p1_results"]["order"]["shuffled_signal"]["probe_acc_mean"] = 0.95
    evaluator_fixed = R8GateEvaluator(raw)
    report_fixed = evaluator_fixed.evaluate_full()
    assert report_fixed["p1_tasks"]["order"]["pass"] is True
    assert report_fixed["p1_tasks"]["order"]["status"] == "PASS"

    # P2 fails on final accuracy and mean forgetting <= 3pp
    p2 = report["p2_stream"]
    assert p2["pass"] is False
    assert p2["status"] == "FAIL"
    assert p2["checks"]["mean_forgetting_lte_3pp"]["pass"] is False
    assert p2["checks"]["final_gte_65"]["pass"] is False

    # Resources pass
    res = report["resources"]
    assert res["pass"] is True
