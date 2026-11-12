"""Unit tests for the R9c independent programmatic gate evaluator."""
from __future__ import annotations

import pytest
from scarx.evaluation.r9c_gate_evaluator import R9cGateEvaluator


def make_valid_payload():
    return {
        "p1_results": {
            "distractor_binding": {
                "candidate": {
                    "direct_acc_mean": 0.68,
                    "per_seed_direct": [0.67, 0.69, 0.68],
                    "probe_acc_mean": 0.72,
                    "probe_gain_lower95": 0.08,
                    "ood_acc_mean": 0.66,
                },
                "frozen_hidden": {"probe_acc_mean": 0.58},
                "shuffled_signal": {"probe_acc_mean": 0.55},
                "causal_revert": {"drop_mean": 0.12},
            }
        },
        "protected_results": {
            "strict_binding": {
                "candidate": {"direct_acc_mean": 0.67, "probe_acc_mean": 0.70},
                "causal_revert": {"drop_mean": 0.14},
                "shuffled_signal": {"probe_acc_mean": 0.55},
            },
            "xor": {
                "candidate": {"direct_acc_mean": 0.96, "probe_acc_mean": 0.97},
                "causal_revert": {"drop_mean": 0.35},
                "shuffled_signal": {"probe_acc_mean": 0.88},
            },
            "order": {
                "candidate": {"direct_acc_mean": 0.98, "probe_acc_mean": 0.99},
                "causal_revert": {"drop_mean": 0.38},
                "shuffled_signal": {"probe_acc_mean": 0.92},
            },
        },
        "resource_results": {
            "peak_rss_bytes": 120_000_000,
        },
    }


def test_r9c_evaluator_valid_pass():
    payload = make_valid_payload()
    evaluator = R9cGateEvaluator(payload)
    report = evaluator.evaluate_full()
    assert report["overall_pass"] is True
    assert report["overall_status"] == "PASS"


def test_r9c_evaluator_no_bypass_for_high_direct_on_causal_drop():
    """Direct acc = 97% MUST NOT bypass causal revert drop check (< 5pp)."""
    payload = make_valid_payload()
    payload["p1_results"]["distractor_binding"]["candidate"]["direct_acc_mean"] = 0.97
    payload["p1_results"]["distractor_binding"]["candidate"]["per_seed_direct"] = [0.96, 0.97, 0.98]
    payload["p1_results"]["distractor_binding"]["causal_revert"]["drop_mean"] = 0.02  # < 5pp!

    evaluator = R9cGateEvaluator(payload)
    report = evaluator.evaluate_full()
    assert report["overall_pass"] is False
    assert report["overall_status"] == "FAIL"
    db_checks = report["p1_target_eval"]["tasks"]["distractor_binding"]["checks"]
    assert db_checks["causal_drop_gte_5pp"]["pass"] is False


def test_r9c_evaluator_no_bypass_for_high_direct_on_shuffled_contrast():
    """Direct acc = 99% MUST NOT bypass matched shuffled contrast (< 2pp)."""
    payload = make_valid_payload()
    payload["p1_results"]["distractor_binding"]["candidate"]["direct_acc_mean"] = 0.99
    payload["p1_results"]["distractor_binding"]["candidate"]["per_seed_direct"] = [0.99, 0.99, 0.99]
    payload["p1_results"]["distractor_binding"]["candidate"]["probe_acc_mean"] = 0.995
    payload["p1_results"]["distractor_binding"]["shuffled_signal"]["probe_acc_mean"] = 0.990  # diff = 0.005 < 0.02!

    evaluator = R9cGateEvaluator(payload)
    report = evaluator.evaluate_full()
    assert report["overall_pass"] is False
    db_checks = report["p1_target_eval"]["tasks"]["distractor_binding"]["checks"]
    assert db_checks["matched_shuffled_contrast_gte_2pp"]["pass"] is False


def test_r9c_evaluator_fail_closed_on_missing_data():
    """Missing fields MUST result in FAIL-CLOSED."""
    # Empty payload
    evaluator = R9cGateEvaluator({})
    report = evaluator.evaluate_full()
    assert report["overall_pass"] is False
    assert report["overall_status"] == "FAIL"

    # Missing causal revert in task data
    payload = make_valid_payload()
    del payload["p1_results"]["distractor_binding"]["causal_revert"]
    evaluator = R9cGateEvaluator(payload)
    report = evaluator.evaluate_full()
    assert report["overall_pass"] is False
    task_res = report["p1_target_eval"]["tasks"]["distractor_binding"]
    assert task_res["status"] == "DATA_MISSING_FAIL_CLOSED"
    assert "causal_revert.drop_mean" in task_res["missing_fields"]


def test_r9c_evaluator_protected_task_regression():
    """A regression in any protected task (e.g. xor drops below 75%) must fail the run."""
    payload = make_valid_payload()
    payload["protected_results"]["xor"]["candidate"]["direct_acc_mean"] = 0.68  # < 75% threshold
    evaluator = R9cGateEvaluator(payload)
    report = evaluator.evaluate_full()
    assert report["overall_pass"] is False
    assert report["protected_tasks_eval"]["pass"] is False
    assert report["protected_tasks_eval"]["tasks"]["xor"]["pass"] is False


def test_r9c_evaluator_ood_non_inferiority():
    """OOD accuracy dropping more than 3pp below ID accuracy must fail."""
    payload = make_valid_payload()
    payload["p1_results"]["distractor_binding"]["candidate"]["direct_acc_mean"] = 0.70
    payload["p1_results"]["distractor_binding"]["candidate"]["ood_acc_mean"] = 0.65  # drop = 5pp > 3pp
    evaluator = R9cGateEvaluator(payload)
    report = evaluator.evaluate_full()
    assert report["overall_pass"] is False
    db_checks = report["p1_target_eval"]["tasks"]["distractor_binding"]["checks"]
    assert db_checks["ood_non_inferiority"]["pass"] is False


def test_r9c_evaluator_per_seed_consistency():
    """If one seed drops to chance (< 52%), per-seed consistency fails."""
    payload = make_valid_payload()
    payload["p1_results"]["distractor_binding"]["candidate"]["per_seed_direct"] = [0.85, 0.70, 0.49]
    evaluator = R9cGateEvaluator(payload)
    report = evaluator.evaluate_full()
    assert report["overall_pass"] is False
    db_checks = report["p1_target_eval"]["tasks"]["distractor_binding"]["checks"]
    assert db_checks["per_seed_consistency"]["pass"] is False
