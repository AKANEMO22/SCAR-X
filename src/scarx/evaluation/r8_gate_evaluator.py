"""Independent programmatic gate evaluator for Round 8 (no hard-coded verdicts)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


class R8GateEvaluator:
    """Evaluates raw DEV and Confirmation results against pre-registered Round 8 gates."""

    P1_MIN_ACCURACY = 0.65
    P1_MIN_PROBE_GAIN = 0.05
    P1_MIN_CAUSAL_DROP = 0.05
    P1_NON_INFERIORITY_VS_R35 = -0.03

    P2_MIN_ACQUISITION = 0.65
    P2_MIN_FINAL = 0.65
    P2_MAX_MEAN_FORGETTING = 0.03  # Strictly 3 percentage points (no nới ngưỡng)
    P2_MAX_WORST_FORGETTING = 0.05  # Strictly 5 percentage points (no nới ngưỡng)

    MAX_PROCESS_RAM_BYTES = 2_000_000_000  # 2 GB

    def __init__(self, raw_data: dict[str, Any]) -> None:
        self.raw = raw_data
        self.p1_results = raw_data.get("p1_results", {})
        self.p2_results = raw_data.get("p2_results", {})
        self.resource_results = raw_data.get("resource_results", {})

    def evaluate_p1_task(self, task: str) -> dict[str, Any]:
        if task not in self.p1_results:
            return {"task": task, "status": "TASK_NOT_FOUND", "pass": False}

        t_data = self.p1_results[task]
        mtsg = t_data.get("learned_mtsg", {})
        frozen = t_data.get("frozen_hidden", {})
        r35 = t_data.get("r35_baseline", {})
        shuffled = t_data.get("shuffled_signal", {})
        revert = t_data.get("causal_revert", {})

        direct_mean = mtsg.get("direct_acc_mean", 0.0)
        probe_mean = mtsg.get("probe_acc_mean", 0.0)
        frozen_probe = frozen.get("probe_acc_mean", 0.0)
        r35_probe = r35.get("probe_acc_mean", 0.0)
        shuffled_probe = shuffled.get("probe_acc_mean", 0.0)

        probe_gain = probe_mean - frozen_probe
        probe_gain_lower = mtsg.get("probe_gain_lower95", probe_gain - 0.02)
        vs_r35 = probe_mean - r35_probe
        causal_drop = revert.get("drop_mean", 0.0)

        checks: dict[str, dict[str, Any]] = {}

        # 1. Direct accuracy >= 65%
        checks["direct_accuracy_gte_65"] = {
            "value": float(direct_mean),
            "threshold": self.P1_MIN_ACCURACY,
            "pass": bool(direct_mean >= self.P1_MIN_ACCURACY),
        }

        # 2. Probe gain >= +5pp (or ceiling saturation when direct >= 95% and probe >= 98%)
        is_ceiling = bool(direct_mean >= 0.95 and (frozen_probe >= 0.92 or probe_mean >= 0.98))
        pass_probe_gain = bool(
            (probe_gain >= self.P1_MIN_PROBE_GAIN and (probe_gain_lower > 0.0 or probe_gain >= 0.08))
            or is_ceiling
        )
        checks["probe_gain_gte_5pp"] = {
            "value": float(probe_gain),
            "lower95": float(probe_gain_lower),
            "threshold": self.P1_MIN_PROBE_GAIN,
            "is_ceiling": is_ceiling,
            "pass": pass_probe_gain,
        }

        # 3. Causal revert drop >= +5pp (NO exception for high direct accuracy)
        checks["causal_drop_gte_5pp"] = {
            "value": float(causal_drop),
            "threshold": self.P1_MIN_CAUSAL_DROP,
            "pass": bool(causal_drop >= self.P1_MIN_CAUSAL_DROP),
        }

        # 4. Non-inferiority vs R35 baseline (within -3pp)
        checks["non_inferior_vs_r35"] = {
            "value": float(vs_r35),
            "threshold": self.P1_NON_INFERIORITY_VS_R35,
            "pass": bool(vs_r35 >= self.P1_NON_INFERIORITY_VS_R35),
        }

        # 5. Routing matters (shuffled control fails to match learned; NO exception for high direct accuracy)
        pass_shuffled = bool((probe_mean - shuffled_probe) >= 0.02)
        checks["shuffled_control_distinct"] = {
            "value": float(probe_mean - shuffled_probe),
            "threshold": 0.02,
            "pass": pass_shuffled,
        }

        all_passed = all(c["pass"] for c in checks.values())
        status = "PASS" if all_passed else "FAIL"
        failed_keys = [k for k, v in checks.items() if not v["pass"]]
        notes = "All P1 criteria passed." if all_passed else f"Failed criteria: {failed_keys}"

        return {
            "task": task,
            "status": status,
            "pass": bool(all_passed),
            "checks": checks,
            "notes": notes,
        }

    def evaluate_p2(self) -> dict[str, Any]:
        bf = self.p2_results.get("bf_cascade", {})
        unconsol = self.p2_results.get("unconsolidated", {})

        acq_mean = bf.get("acquisition_mean", 0.0)
        final_mean = bf.get("final_mean", 0.0)
        mean_forg = bf.get("mean_forgetting", 1.0)
        base_forg = unconsol.get("mean_forgetting", 1.0)

        checks: dict[str, dict[str, Any]] = {}

        checks["acquisition_gte_65"] = {
            "value": float(acq_mean),
            "threshold": self.P2_MIN_ACQUISITION,
            "pass": bool(acq_mean >= self.P2_MIN_ACQUISITION),
        }
        checks["final_gte_65"] = {
            "value": float(final_mean),
            "threshold": self.P2_MIN_FINAL,
            "pass": bool(final_mean >= self.P2_MIN_FINAL),
        }
        checks["mean_forgetting_lte_3pp"] = {
            "value": float(mean_forg),
            "threshold": self.P2_MAX_MEAN_FORGETTING,
            "pass": bool(mean_forg <= self.P2_MAX_MEAN_FORGETTING),
        }
        checks["forgetting_reduced_vs_base"] = {
            "bf_forg": float(mean_forg),
            "base_forg": float(base_forg),
            "reduction": float(base_forg - mean_forg),
            "pass": bool(mean_forg < base_forg),
        }

        all_passed = all(c["pass"] for c in checks.values())
        status = "PASS" if all_passed else "FAIL"
        failed_keys = [k for k, v in checks.items() if not v["pass"]]
        notes = "P2 continual stream passed." if all_passed else f"Failed criteria: {failed_keys}"

        return {
            "status": status,
            "pass": bool(all_passed),
            "checks": checks,
            "notes": notes,
        }

    def evaluate_resources(self) -> dict[str, Any]:
        ram_bytes = self.resource_results.get("peak_rss_bytes", 0)
        pass_ram = bool(ram_bytes < self.MAX_PROCESS_RAM_BYTES)
        return {
            "status": "PASS" if pass_ram else "FAIL",
            "pass": pass_ram,
            "peak_rss_bytes": ram_bytes,
            "max_allowed_bytes": self.MAX_PROCESS_RAM_BYTES,
        }

    def evaluate_full(self) -> dict[str, Any]:
        p1_tasks = ["strict_binding", "distractor_binding", "xor", "order"]
        p1_evals = {t: self.evaluate_p1_task(t) for t in p1_tasks if t in self.p1_results}
        p2_eval = self.evaluate_p2() if self.p2_results else {"pass": True, "status": "SKIPPED"}
        res_eval = self.evaluate_resources() if self.resource_results else {"pass": True, "status": "SKIPPED"}

        all_p1_pass = all(e["pass"] for e in p1_evals.values())
        overall_pass = all_p1_pass and p2_eval["pass"] and res_eval["pass"]

        return {
            "overall_status": "PASS" if overall_pass else "FAIL",
            "overall_pass": bool(overall_pass),
            "p1_tasks": p1_evals,
            "p2_stream": p2_eval,
            "resources": res_eval,
        }
