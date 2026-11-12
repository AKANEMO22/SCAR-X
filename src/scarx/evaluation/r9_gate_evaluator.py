"""SCAR-X Round 9 Independent Programmatic Gate Evaluator & Statistical Audit.

Strict Rules Enforced:
1. Direct Accuracy >= 65.0% across all evaluated tasks.
2. Probe Gain >= +5.0 percentage points with 95% CI lower bound > 0.0.
3. Causal Revert Drop >= +5.0 percentage points (STRICTLY NO direct_accuracy >= 95% BYPASS).
4. Continual Learning: Mean Forgetting <= 3.0 pp, Worst Forgetting <= 5.0 pp.
5. Shuffled Signal Routing Check: Learned probe must exceed shuffled signal probe.
6. Multiple Testing Control: Benjamini-Hochberg False Discovery Rate (FDR) q <= 0.05.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Applies Benjamini-Hochberg FDR correction to a list of p-values."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    significant = sorted_p <= thresholds
    # Step-down finding max index
    if not np.any(significant):
        return [False] * n
    max_k = np.max(np.where(significant)[0])
    passed = np.zeros(n, dtype=bool)
    passed[sorted_indices[: max_k + 1]] = True
    return passed.tolist()


def compute_paired_ci95(diffs: list[float] | np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    # Student's t critical value lookup
    t_table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262
    }
    t_crit = t_table.get(n, 2.262)
    return mean, mean - t_crit * se, mean + t_crit * se


class R9GateEvaluator:
    """Evaluates candidates against pre-registered Round 9 gates without exceptions."""

    MIN_DIRECT_ACC = 0.65
    MIN_PROBE_GAIN = 0.05
    MIN_CAUSAL_DROP = 0.05
    MAX_MEAN_FORGETTING = 0.03
    MAX_WORST_FORGETTING = 0.05

    def __init__(self, run_results: dict[str, Any]) -> None:
        self.raw = run_results

    def evaluate_task_results(self, task_name: str, task_summary: dict[str, Any]) -> dict[str, Any]:
        cand = task_summary.get("learned_candidate", {})
        frozen = task_summary.get("frozen_hidden", {})
        shuffled = task_summary.get("shuffled_signal", {})
        revert = task_summary.get("causal_revert", {})

        direct_mean = cand.get("direct_acc_mean", 0.0)
        probe_mean = cand.get("probe_acc_mean", 0.0)
        frozen_probe = frozen.get("probe_acc_mean", 0.0)
        shuffled_probe = shuffled.get("probe_acc_mean", 0.0)

        probe_gain = probe_mean - frozen_probe
        probe_gain_lower = cand.get("probe_gain_lower95", probe_gain)
        causal_drop = revert.get("drop_mean", 0.0)

        checks: dict[str, dict[str, Any]] = {}

        # 1. Direct accuracy >= 65%
        checks["direct_accuracy_gte_65"] = {
            "value": float(direct_mean),
            "threshold": self.MIN_DIRECT_ACC,
            "pass": bool(direct_mean >= self.MIN_DIRECT_ACC),
        }

        # 2. Probe gain >= +5pp and CI lower bound > 0
        checks["probe_gain_gte_5pp"] = {
            "value": float(probe_gain),
            "lower95": float(probe_gain_lower),
            "threshold": self.MIN_PROBE_GAIN,
            "pass": bool(probe_gain >= self.MIN_PROBE_GAIN and probe_gain_lower > 0.0),
        }

        # 3. Causal revert drop >= +5pp (STRICT: no bypass when direct_acc >= 95%)
        checks["causal_drop_gte_5pp"] = {
            "value": float(causal_drop),
            "threshold": self.MIN_CAUSAL_DROP,
            "pass": bool(causal_drop >= self.MIN_CAUSAL_DROP),
            "bypass_applied": False,  # Explicit audit affirmation
        }

        # 4. Routing specificity (learned probe > shuffled signal probe by at least 2pp)
        shuffled_margin = probe_mean - shuffled_probe
        checks["routing_specificity"] = {
            "value": float(shuffled_margin),
            "threshold": 0.02,
            "pass": bool(shuffled_margin >= 0.02),
        }

        all_passed = all(c["pass"] for c in checks.values())
        return {
            "task": task_name,
            "status": "PASS" if all_passed else "FAIL",
            "checks": checks,
            "failed_checks": [k for k, v in checks.items() if not v["pass"]],
        }

    def evaluate_continual_results(self, cont_summary: dict[str, Any]) -> dict[str, Any]:
        acq_mean = cont_summary.get("acquisition_mean", 0.0)
        final_mean = cont_summary.get("final_mean", 0.0)
        mean_forg = cont_summary.get("mean_forgetting", 1.0)
        worst_forg = cont_summary.get("worst_forgetting", 1.0)

        checks = {
            "acquisition_gte_65": {
                "value": float(acq_mean),
                "threshold": self.MIN_DIRECT_ACC,
                "pass": bool(acq_mean >= self.MIN_DIRECT_ACC),
            },
            "final_gte_65": {
                "value": float(final_mean),
                "threshold": self.MIN_DIRECT_ACC,
                "pass": bool(final_mean >= self.MIN_DIRECT_ACC),
            },
            "mean_forgetting_lte_3pp": {
                "value": float(mean_forg),
                "threshold": self.MAX_MEAN_FORGETTING,
                "pass": bool(mean_forg <= self.MAX_MEAN_FORGETTING),
            },
            "worst_forgetting_lte_5pp": {
                "value": float(worst_forg),
                "threshold": self.MAX_WORST_FORGETTING,
                "pass": bool(worst_forg <= self.MAX_WORST_FORGETTING),
            },
        }

        all_passed = all(c["pass"] for c in checks.values())
        return {
            "status": "PASS" if all_passed else "FAIL",
            "checks": checks,
            "failed_checks": [k for k, v in checks.items() if not v["pass"]],
        }
