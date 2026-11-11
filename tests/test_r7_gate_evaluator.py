"""Unit tests for the independent programmatic R7 gate evaluator."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from scarx.evaluation.r7_gate_evaluator import R7GateEvaluator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_SCREEN_PATH = REPO_ROOT / "artifacts" / "gemini-r7" / "dev-screen.json"


def test_evaluator_on_real_dev_screen():
    if not DEV_SCREEN_PATH.exists():
        pytest.skip("dev-screen.json artifact not found")

    with open(DEV_SCREEN_PATH, "r", encoding="utf-8") as f:
        artifact = json.load(f)

    evaluator = R7GateEvaluator(artifact)
    report = evaluator.evaluate_all()

    # 1. Overall verdict must be FAIL based on raw metrics
    assert report["overall_verdict"] == "FAIL"

    # 2. strict_binding passes pre-registered criteria
    sb = report["p1_task_verdicts"]["strict_binding"]
    assert sb["pass"] is True
    assert sb["status"] == "PASS"

    # 3. distractor_binding must fail (EW-HE failed to learn under distractors)
    db = report["p1_task_verdicts"]["distractor_binding"]
    assert db["pass"] is False
    assert db["status"] == "FAIL"

    # 4. xor must be flagged as inferior to R35 (86.46% vs 96.48%)
    xor = report["p1_task_verdicts"]["xor"]
    assert xor["pass"] is False
    assert xor["status"] == "FAIL_VS_R35"

    # 5. P2 continual learning gate must fail due to strict 3pp/5pp criteria
    p2 = report["p2_verdict"]
    assert p2["pass"] is False
    assert p2["status"] == "FAIL"
    assert p2["checks"]["mean_forgetting_lte_3pp"]["pass"] is False
    assert p2["checks"]["worst_forgetting_lte_5pp"]["pass"] is False


def test_evaluator_synthetic_data():
    synthetic_artifact = {
        "dev_p1_summary": {
            "task_pass": {
                "conditions": {
                    "learned_ew_he": {
                        "id_direct_acc": {"mean": 0.85},
                        "id_probe_acc": {"mean": 0.85},
                    },
                    "frozen_hidden": {
                        "id_probe_acc": {"mean": 0.65},
                    },
                    "r35_baseline": {
                        "id_probe_acc": {"mean": 0.80},
                    },
                    "causal_revert": {
                        "id_causal_drop": {"mean": 0.20},
                    },
                },
                "contrasts": {
                    "ew_he_vs_frozen_id_probe": {"lower95": 0.15},
                },
            },
            "task_fail": {
                "conditions": {
                    "learned_ew_he": {
                        "id_direct_acc": {"mean": 0.55},
                        "id_probe_acc": {"mean": 0.55},
                    },
                    "frozen_hidden": {
                        "id_probe_acc": {"mean": 0.55},
                    },
                    "r35_baseline": {
                        "id_probe_acc": {"mean": 0.70},
                    },
                    "causal_revert": {
                        "id_causal_drop": {"mean": 0.01},
                    },
                },
                "contrasts": {
                    "ew_he_vs_frozen_id_probe": {"lower95": -0.05},
                },
            },
        },
        "dev_p2_summary": {
            "modes": {
                "ew_he_ca_msc": {
                    "acquisition_accuracy": {"mean": 0.90},
                    "final_accuracy": {"mean": 0.88},
                    "mean_forgetting": {"mean": 0.02},
                    "worst_forgetting": {"mean": 0.04},
                }
            }
        },
        "dev_p2_trials": [
            {
                "mode": "ew_he_ca_msc",
                "seed": 1,
                "acquisition_accuracy": 0.90,
                "final_accuracy": 0.88,
                "mean_forgetting": 0.02,
                "worst_forgetting": 0.04,
            }
        ],
    }

    evaluator = R7GateEvaluator(synthetic_artifact)
    assert evaluator.evaluate_p1_task("task_pass")["pass"] is True
    assert evaluator.evaluate_p1_task("task_fail")["pass"] is False
    assert evaluator.evaluate_p2()["pass"] is True
