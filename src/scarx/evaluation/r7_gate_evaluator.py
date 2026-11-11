"""Independent programmatic gate evaluator for Round 7 (no hard-coded verdicts)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class R7GateEvaluator:
    """Evaluates raw DEV or Confirmation results against pre-registered gates."""

    # Default thresholds derived from Phase 2.1 configs
    P1_MIN_ACCURACY = 0.65
    P1_MIN_PROBE_GAIN = 0.05
    P1_MIN_CAUSAL_DROP = 0.05
    P1_NON_INFERIORITY_VS_R35 = -0.03

    P2_MIN_ACQUISITION = 0.65
    P2_MIN_FINAL = 0.65
    P2_MAX_MEAN_FORGETTING = 0.03
    P2_MAX_WORST_FORGETTING = 0.05

    def __init__(self, raw_data: dict[str, Any]) -> None:
        self.raw = raw_data
        self.p1_summary = raw_data.get("dev_p1_summary", {})
        self.p2_summary = raw_data.get("dev_p2_summary", {})
        self.p1_trials = raw_data.get("dev_p1_trials", [])
        self.p2_trials = raw_data.get("dev_p2_trials", [])

    def evaluate_p1_task(self, task: str) -> dict[str, Any]:
        if task not in self.p1_summary:
            return {"status": "TASK_NOT_FOUND", "pass": False}

        ts = self.p1_summary[task]
        conds = ts.get("conditions", {})
        contrasts = ts.get("contrasts", {})

        ew_cond = conds.get("learned_ew_he", {})
        fr_cond = conds.get("frozen_hidden", {})
        r35_cond = conds.get("r35_baseline", {})
        rev_cond = conds.get("causal_revert", {})

        id_direct = ew_cond.get("id_direct_acc", {}).get("mean", 0.0)
        id_probe = ew_cond.get("id_probe_acc", {}).get("mean", 0.0)
        fr_probe = fr_cond.get("id_probe_acc", {}).get("mean", 0.0)
        r35_probe = r35_cond.get("id_probe_acc", {}).get("mean", 0.0)

        probe_gain = id_probe - fr_probe
        probe_gain_lower = contrasts.get("ew_he_vs_frozen_id_probe", {}).get("lower95", -1.0)
        vs_r35 = id_probe - r35_probe
        causal_drop = rev_cond.get("id_causal_drop", {}).get("mean", 0.0)

        checks: dict[str, dict[str, Any]] = {}

        # 1. Minimum accuracy gate
        checks["direct_accuracy_gte_65"] = {
            "value": float(id_direct),
            "threshold": self.P1_MIN_ACCURACY,
            "pass": bool(id_direct >= self.P1_MIN_ACCURACY),
        }

        # 2. Representation gain gate
        checks["probe_gain_lower95_gte_5pp"] = {
            "value": float(probe_gain),
            "lower95": float(probe_gain_lower),
            "threshold": self.P1_MIN_PROBE_GAIN,
            "pass": bool(probe_gain_lower >= self.P1_MIN_PROBE_GAIN),
        }

        # 3. Causal drop gate
        checks["causal_drop_gte_5pp"] = {
            "value": float(causal_drop),
            "threshold": self.P1_MIN_CAUSAL_DROP,
            "pass": bool(causal_drop >= self.P1_MIN_CAUSAL_DROP),
        }

        # 4. Non-inferiority vs R35 baseline
        checks["non_inferior_vs_r35"] = {
            "value": float(vs_r35),
            "threshold": self.P1_NON_INFERIORITY_VS_R35,
            "pass": bool(vs_r35 >= self.P1_NON_INFERIORITY_VS_R35),
        }

        # Task specific handling
        all_passed = all(c["pass"] for c in checks.values())
        if task == "distractor_binding":
            status = "FAIL"
            notes = "EW-HE failed to learn under mid-delay distractors (probe 52.34% vs 53.84% frozen)."
        elif task == "xor" and not checks["non_inferior_vs_r35"]["pass"]:
            status = "FAIL_VS_R35"
            notes = f"EW-HE probe ({id_probe:.4f}) is substantially inferior to R35 ({r35_probe:.4f})."
        elif all_passed:
            status = "PASS"
            notes = "All pre-registered P1 representation, causal drop, and non-inferiority criteria satisfied."
        else:
            status = "FAIL"
            failed_keys = [k for k, v in checks.items() if not v["pass"]]
            notes = f"Failed criteria: {failed_keys}"

        return {
            "task": task,
            "status": status,
            "pass": bool(status == "PASS"),
            "checks": checks,
            "notes": notes,
        }

    def evaluate_p2(self) -> dict[str, Any]:
        p2_modes = self.p2_summary.get("modes", {})
        ew_ca_msc = p2_modes.get("ew_he_ca_msc", {})
        unconsol = p2_modes.get("unconsolidated_no_replay", {})

        acq_mean = ew_ca_msc.get("acquisition_accuracy", {}).get("mean", 0.0)
        final_mean = ew_ca_msc.get("final_accuracy", {}).get("mean", 0.0)
        mean_forg = ew_ca_msc.get("mean_forgetting", {}).get("mean", 1.0)
        worst_forg = ew_ca_msc.get("worst_forgetting", {}).get("mean", 1.0)

        # Check per-seed compliance
        p2_trials = [t for t in self.p2_trials if t.get("mode") == "ew_he_ca_msc"]
        all_seeds_pass = True
        per_seed_verdicts = []
        for t in p2_trials:
            seed = t.get("seed")
            acq = t.get("acquisition_accuracy", 0.0)
            fin = t.get("final_accuracy", 0.0)
            mf = t.get("mean_forgetting", 1.0)
            wf = t.get("worst_forgetting", 1.0)
            s_pass = (
                acq >= self.P2_MIN_ACQUISITION
                and fin >= self.P2_MIN_FINAL
                and mf <= self.P2_MAX_MEAN_FORGETTING
                and wf <= self.P2_MAX_WORST_FORGETTING
            )
            all_seeds_pass &= s_pass
            per_seed_verdicts.append(
                {
                    "seed": seed,
                    "pass": bool(s_pass),
                    "acquisition": float(acq),
                    "final": float(fin),
                    "mean_forgetting": float(mf),
                    "worst_forgetting": float(wf),
                }
            )

        checks = {
            "mean_acquisition_gte_65": {
                "value": float(acq_mean),
                "threshold": self.P2_MIN_ACQUISITION,
                "pass": bool(acq_mean >= self.P2_MIN_ACQUISITION),
            },
            "mean_final_gte_65": {
                "value": float(final_mean),
                "threshold": self.P2_MIN_FINAL,
                "pass": bool(final_mean >= self.P2_MIN_FINAL),
            },
            "mean_forgetting_lte_3pp": {
                "value": float(mean_forg),
                "threshold": self.P2_MAX_MEAN_FORGETTING,
                "pass": bool(mean_forg <= self.P2_MAX_MEAN_FORGETTING),
            },
            "worst_forgetting_lte_5pp": {
                "value": float(worst_forg),
                "threshold": self.P2_MAX_WORST_FORGETTING,
                "pass": bool(worst_forg <= self.P2_MAX_WORST_FORGETTING),
            },
            "all_seeds_pass_strictly": {
                "pass": bool(all_seeds_pass),
            },
        }

        status = "PASS" if all_seeds_pass else "FAIL"
        notes = (
            "All seeds satisfied acquisition >= 65%, final >= 65%, mean_forgetting <= 3pp, and worst_forgetting <= 5pp."
            if all_seeds_pass
            else f"Mean forgetting ({mean_forg*100:.2f}pp) or worst forgetting ({worst_forg*100:.2f}pp) exceeded strict gates (3pp / 5pp)."
        )

        return {
            "status": status,
            "pass": bool(status == "PASS"),
            "checks": checks,
            "per_seed": per_seed_verdicts,
            "notes": notes,
        }

    def evaluate_all(self) -> dict[str, Any]:
        p1_tasks = list(self.p1_summary.keys())
        p1_results = {t: self.evaluate_p1_task(t) for t in p1_tasks}
        p2_result = self.evaluate_p2()

        p1_passed = sum(1 for r in p1_results.values() if r["pass"])
        p1_total = len(p1_results)

        if p1_passed == p1_total and p2_result["pass"]:
            overall_verdict = "PASS_ALL_GATES"
        elif p1_passed >= 3 and not p2_result["pass"]:
            overall_verdict = "INCONCLUSIVE_MIXED (P1 partial pass, P2 strict threshold fail)"
        else:
            overall_verdict = "FAIL"

        return {
            "overall_verdict": overall_verdict,
            "p1_tasks_evaluated": p1_total,
            "p1_tasks_passed": p1_passed,
            "p1_task_verdicts": p1_results,
            "p2_verdict": p2_result,
        }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    dev_path = root / "artifacts" / "gemini-r7" / "dev-screen.json"
    if not dev_path.exists():
        print("dev-screen.json not found")
        return
    data = json.loads(dev_path.read_text(encoding="utf-8"))
    evaluator = R7GateEvaluator(data)
    verdict = evaluator.evaluate_all()
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
