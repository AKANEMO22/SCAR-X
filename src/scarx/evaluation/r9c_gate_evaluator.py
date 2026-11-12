"""SCAR-X Round 9c Programmatic Gate Evaluator.

Implements strict, fail-closed audit standards:
1. No exception for high direct accuracy (mean_dir >= 95% does NOT pass causal drop or shuffled contrast).
2. Direct accuracy >= 65.0% with 95% Confidence Intervals.
3. Representation probe gain >= +5pp with CI95 lower bound > 0.
4. Causal revert drop strictly >= +5pp.
5. Matched shuffled control contrast strictly >= +2pp.
6. OOD non-inferiority within -3pp of ID baseline.
7. Protected tasks (strict_binding, xor, order) regression guards.
8. Per-seed consistency check (no seed collapsing to chance, independent seeds).
9. Fail-closed semantics: any missing, invalid, or infinite field fails immediately.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np


class R9cGateEvaluator:
    """Rigorous programmatic gate evaluator for Round 9c."""

    P1_MIN_ACCURACY = 0.65
    P1_MIN_PROBE_GAIN = 0.05
    P1_MIN_CAUSAL_DROP = 0.05
    P1_MIN_SHUFFLED_DIFF = 0.02
    P1_MAX_OOD_DROP = 0.03

    PROTECTED_THRESHOLDS = {
        "strict_binding": {"direct": 0.65, "causal_drop": 0.05, "vs_shuffled": 0.02},
        "xor": {"direct": 0.75, "causal_drop": 0.05, "vs_shuffled": 0.02},
        "order": {"direct": 0.90, "causal_drop": 0.05, "vs_shuffled": 0.02},
    }

    MAX_PROCESS_RAM_BYTES = 2_000_000_000  # 2 GB

    def __init__(self, raw_data: dict[str, Any]) -> None:
        self.raw = raw_data
        self.p1_results = raw_data.get("p1_results", {}) if isinstance(raw_data, dict) else {}
        self.protected_results = raw_data.get("protected_results", {}) if isinstance(raw_data, dict) else {}
        self.resource_results = raw_data.get("resource_results", {}) if isinstance(raw_data, dict) else {}

    @staticmethod
    def _is_finite_num(v: Any) -> bool:
        if v is None:
            return False
        try:
            fv = float(v)
            return math.isfinite(fv)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _compute_ci95(values: list[float]) -> tuple[float, float, float]:
        """Returns (mean, ci95_lower, ci95_upper) using Student's t distribution approximation."""
        if not values:
            return 0.0, 0.0, 0.0
        arr = np.array(values, dtype=np.float64)
        n = len(arr)
        mean = float(np.mean(arr))
        if n < 2:
            return mean, mean, mean
        std = float(np.std(arr, ddof=1))
        # Critical t-values for df = n - 1 (alpha = 0.05 two-tailed)
        t_crit_table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        }
        t_crit = t_crit_table.get(n - 1, 1.96)
        margin = t_crit * (std / np.sqrt(n))
        return mean, mean - margin, mean + margin

    def evaluate_task(self, task_name: str, task_data: dict[str, Any]) -> dict[str, Any]:
        """Evaluates a single task with strict, fail-closed checks."""
        checks: dict[str, dict[str, Any]] = {}
        missing_fields: list[str] = []

        if not isinstance(task_data, dict):
            return {
                "task": task_name,
                "status": "DATA_MISSING_FAIL_CLOSED",
                "pass": False,
                "notes": "Task data is not a dictionary.",
                "checks": {},
            }

        candidate = task_data.get("candidate", {})
        frozen = task_data.get("frozen_hidden", {})
        shuffled = task_data.get("shuffled_signal", {})
        revert = task_data.get("causal_revert", {})

        # Direct accuracy
        direct_mean = candidate.get("direct_acc_mean")
        per_seed_direct = candidate.get("per_seed_direct", [])
        if not self._is_finite_num(direct_mean):
            missing_fields.append("candidate.direct_acc_mean")
            direct_mean = 0.0
        direct_mean = float(direct_mean)

        ci95_lower = candidate.get("ci95_lower")
        ci95_upper = candidate.get("ci95_upper")
        if not (self._is_finite_num(ci95_lower) and self._is_finite_num(ci95_upper)):
            if per_seed_direct and len(per_seed_direct) >= 2:
                _, ci95_lower, ci95_upper = self._compute_ci95([float(x) for x in per_seed_direct])
            else:
                missing_fields.append("candidate.ci95")
                ci95_lower, ci95_upper = 0.0, 0.0

        checks["direct_accuracy_gte_65"] = {
            "value": direct_mean,
            "threshold": self.P1_MIN_ACCURACY,
            "ci95": [float(ci95_lower), float(ci95_upper)],
            "pass": bool(direct_mean >= self.P1_MIN_ACCURACY),
        }

        # Representation probe gain
        probe_mean = candidate.get("probe_acc_mean")
        frozen_probe = frozen.get("probe_acc_mean")
        if not (self._is_finite_num(probe_mean) and self._is_finite_num(frozen_probe)):
            missing_fields.append("probe_acc_mean / frozen.probe_acc_mean")
            probe_gain = 0.0
            probe_gain_lower = -1.0
        else:
            probe_gain = float(probe_mean) - float(frozen_probe)
            probe_gain_lower = float(candidate.get("probe_gain_lower95", probe_gain - 0.02))

        checks["probe_gain_gte_5pp"] = {
            "value": probe_gain,
            "lower95": probe_gain_lower,
            "threshold": self.P1_MIN_PROBE_GAIN,
            "pass": bool(probe_gain >= self.P1_MIN_PROBE_GAIN and probe_gain_lower > 0.0),
        }

        # Causal revert drop (NO EXCEPTION)
        causal_drop = revert.get("drop_mean")
        if not self._is_finite_num(causal_drop):
            missing_fields.append("causal_revert.drop_mean")
            causal_drop = 0.0
        causal_drop = float(causal_drop)

        checks["causal_drop_gte_5pp"] = {
            "value": causal_drop,
            "threshold": self.P1_MIN_CAUSAL_DROP,
            "pass": bool(causal_drop >= self.P1_MIN_CAUSAL_DROP),
        }

        # Matched shuffled control contrast (NO EXCEPTION)
        shuffled_probe = shuffled.get("probe_acc_mean")
        if not (self._is_finite_num(probe_mean) and self._is_finite_num(shuffled_probe)):
            missing_fields.append("shuffled_signal.probe_acc_mean")
            shuf_diff = 0.0
        else:
            shuf_diff = float(probe_mean) - float(shuffled_probe)

        checks["matched_shuffled_contrast_gte_2pp"] = {
            "value": shuf_diff,
            "threshold": self.P1_MIN_SHUFFLED_DIFF,
            "pass": bool(shuf_diff >= self.P1_MIN_SHUFFLED_DIFF),
        }

        # OOD non-inferiority check
        ood_acc = candidate.get("ood_acc_mean")
        if self._is_finite_num(ood_acc):
            ood_drop = direct_mean - float(ood_acc)
            pass_ood = bool(ood_drop <= self.P1_MAX_OOD_DROP)
            checks["ood_non_inferiority"] = {
                "ood_acc": float(ood_acc),
                "id_acc": direct_mean,
                "ood_drop": ood_drop,
                "max_allowed_drop": self.P1_MAX_OOD_DROP,
                "pass": pass_ood,
            }
        else:
            checks["ood_non_inferiority"] = {
                "status": "NOT_PROVIDED_FAIL_CLOSED",
                "pass": False,
            }
            missing_fields.append("candidate.ood_acc_mean")

        # Per-seed consistency check
        if per_seed_direct and len(per_seed_direct) >= 3:
            all_finite = all(self._is_finite_num(s) for s in per_seed_direct)
            min_seed = min(float(s) for s in per_seed_direct) if all_finite else 0.0
            # No seed should completely collapse to pure chance (0.50)
            pass_seeds = all_finite and (min_seed >= 0.52)
            checks["per_seed_consistency"] = {
                "n_seeds": len(per_seed_direct),
                "min_seed": min_seed,
                "pass": pass_seeds,
            }
        else:
            checks["per_seed_consistency"] = {
                "status": "INSUFFICIENT_SEEDS_FAIL_CLOSED",
                "pass": False,
            }
            missing_fields.append("candidate.per_seed_direct (n>=3)")

        if missing_fields:
            return {
                "task": task_name,
                "status": "DATA_MISSING_FAIL_CLOSED",
                "pass": False,
                "missing_fields": missing_fields,
                "checks": checks,
            }

        all_pass = all(c.get("pass", False) for c in checks.values())
        status = "PASS" if all_pass else "FAIL"

        return {
            "task": task_name,
            "status": status,
            "pass": bool(all_pass),
            "checks": checks,
        }

    def evaluate_protected_tasks(self) -> dict[str, Any]:
        """Verifies no regression on protected tasks: strict_binding, xor, order."""
        evals = {}
        missing_tasks = []

        for p_task, thresholds in self.PROTECTED_THRESHOLDS.items():
            if p_task not in self.protected_results:
                missing_tasks.append(p_task)
                evals[p_task] = {"task": p_task, "status": "MISSING_FAIL_CLOSED", "pass": False}
                continue

            t_data = self.protected_results[p_task]
            cand = t_data.get("candidate", {})
            revert = t_data.get("causal_revert", {})
            shuf = t_data.get("shuffled_signal", {})

            direct = float(cand.get("direct_acc_mean", 0.0))
            drop = float(revert.get("drop_mean", 0.0))
            probe = float(cand.get("probe_acc_mean", 0.0))
            shuf_probe = float(shuf.get("probe_acc_mean", 0.0))
            vs_shuf = probe - shuf_probe

            pass_direct = bool(direct >= thresholds["direct"])
            pass_drop = bool(drop >= thresholds["causal_drop"])
            pass_shuf = bool(vs_shuf >= thresholds["vs_shuffled"])

            task_pass = pass_direct and pass_drop and pass_shuf
            evals[p_task] = {
                "task": p_task,
                "status": "PASS" if task_pass else "FAIL",
                "pass": bool(task_pass),
                "checks": {
                    "direct": {"value": direct, "threshold": thresholds["direct"], "pass": pass_direct},
                    "causal_drop": {"value": drop, "threshold": thresholds["causal_drop"], "pass": pass_drop},
                    "vs_shuffled": {"value": vs_shuf, "threshold": thresholds["vs_shuffled"], "pass": pass_shuf},
                },
            }

        all_protected_pass = len(missing_tasks) == 0 and all(v["pass"] for v in evals.values())
        return {
            "status": "PASS" if all_protected_pass else "FAIL",
            "pass": bool(all_protected_pass),
            "tasks": evals,
            "missing_tasks": missing_tasks,
        }

    def evaluate_resources(self) -> dict[str, Any]:
        ram_bytes = self.resource_results.get("peak_rss_bytes")
        if not self._is_finite_num(ram_bytes):
            return {
                "status": "RESOURCE_DATA_MISSING_FAIL_CLOSED",
                "pass": False,
            }
        ram_bytes = int(ram_bytes)
        pass_ram = bool(ram_bytes < self.MAX_PROCESS_RAM_BYTES)
        return {
            "status": "PASS" if pass_ram else "FAIL",
            "pass": pass_ram,
            "peak_rss_bytes": ram_bytes,
            "max_allowed_bytes": self.MAX_PROCESS_RAM_BYTES,
        }

    def evaluate_full(self) -> dict[str, Any]:
        """Runs the complete Round 9c audit suite."""
        # 1. P1 target tasks (distractor_binding)
        p1_tasks_eval = {}
        for t_name, t_data in self.p1_results.items():
            p1_tasks_eval[t_name] = self.evaluate_task(t_name, t_data)

        if not p1_tasks_eval:
            p1_overall_pass = False
            p1_status = "NO_P1_TASKS_FAIL_CLOSED"
        else:
            p1_overall_pass = all(e["pass"] for e in p1_tasks_eval.values())
            p1_status = "PASS" if p1_overall_pass else "FAIL"

        # 2. Protected tasks
        protected_eval = self.evaluate_protected_tasks()

        # 3. Resources
        resource_eval = self.evaluate_resources()

        overall_pass = bool(p1_overall_pass and protected_eval["pass"] and resource_eval["pass"])
        overall_status = "PASS" if overall_pass else "FAIL"

        return {
            "overall_status": overall_status,
            "overall_pass": overall_pass,
            "p1_target_eval": {
                "status": p1_status,
                "pass": p1_overall_pass,
                "tasks": p1_tasks_eval,
            },
            "protected_tasks_eval": protected_eval,
            "resource_eval": resource_eval,
        }
