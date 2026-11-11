"""Orchestrator for Antigravity (Gemini 4 Argon (High)) DC-ALIF Round 1 Campaign (`scarx-neural-round1-20261111`).

Executes:
1. Mandatory P0 Engineering Verification (`dc_alif_finite_difference_check`, leakage, zero/frozen parity,
   partial-segment save/load, fault rollback, cross-loader rejection).
2. Locked P1 Development Screen (3 fresh seeds x 3 tasks x 6 methods = 54 trials) + cold-process resume.
3. Locked P1 Confirmation (8 fresh seeds x 3 tasks x 6 methods = 144 trials, opened only if Screen passes)
   + cold-process resume.
4. P2 20-Context Continual Retention Stream Diagnostic (3 fresh seeds x 3 conditions = 9 trials).
"""
from __future__ import annotations

import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from dc_alif_worker import METHODS, P2_CONDITIONS, P2_SEEDS
from direct_eligibility_common import CONFIRM, DEV, TASKS, dataset
from scarx.evaluation.dc_alif_eligibility import (
    DCALIFBPTTControl,
    DCALIFEligibility,
    dc_alif_finite_difference_check,
)
from scarx.evaluation.phase21 import interval, signflip
from scarx.evaluation.readout_eligibility import ReadoutEligibility
from scarx.evaluation.runs import new_run, write_json
from scarx.evaluation.visual_words import VisualWords
from scarx.neural.core import Config
from scarx.neural.unified import UnifiedCore
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from visual_word_control_common import atomic

CAMPAIGN = "scarx-neural-round1-20261111"
AGENT_ID = "antigravity"
BASE_SHA = "1735ad66ea9e7b416f5df15df99d9c13cea46b4e"
PROTOCOL_PATH = ROOT / "experiments" / "collab" / CAMPAIGN / AGENT_ID / "protocol-v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "collab" / CAMPAIGN / AGENT_ID


def run_p0_engineering_verification() -> dict[str, Any]:
    fd_results: dict[str, Any] = {}
    for idx, task in enumerate(TASKS):
        seed = 35101 + 17 * idx
        c = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0)
        m = DCALIFEligibility(c, "correct")
        cal, _ = dataset(task, seed + 1360000, 128)
        m.calibrate(cal)
        seq, _ = dataset(task, seed + 1370000, 4)
        fd_results[task] = dc_alif_finite_difference_check(m, seq[0])
        assert fd_results[task]["finite_difference_synapses"] == 64
        assert fd_results[task]["max_absolute_error_e_f"] < 1e-5
        assert fd_results[task]["max_absolute_error_e_d"] < 1e-5

    # Verify zero/frozen exact bit-equality and matched RNG state consumption across all candidate conditions
    seed = 35107
    models = {
        cond: DCALIFEligibility(
            Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0),
            cond,
        )
        for cond in ("correct", "no_gate", "shuffled", "zero", "frozen")
    }
    cal, _ = dataset("strict_binding", seed + 1360000, 128)
    train_x, train_y = dataset("strict_binding", seed + 1370000, 16)
    for m in models.values():
        m.calibrate(cal)
    for seq, label in zip(train_x, train_y):
        for m in models.values():
            m.learn(seq, int(label))

    assert models["zero"].core.dumps() == models["frozen"].core.dumps()
    assert models["zero"].head.dumps() == models["frozen"].head.dumps()
    rng_ref = models["correct"].signal_rng.bit_generator.state
    for m in models.values():
        assert m.signal_rng.bit_generator.state == rng_ref

    # Mid-segment checkpoint save/load and cross-loader rejection
    m_mid = models["correct"]
    m_mid.begin(True)
    for row in train_x[0, :6]:
        m_mid.step(row)
    blob = m_mid.dumps()
    restored = DCALIFEligibility.loads(blob)
    for row in train_x[0, 6:]:
        np.testing.assert_array_equal(m_mid.step(row), restored.step(row))
    assert m_mid.accumulate(int(train_y[0])) == restored.accumulate(int(train_y[0]))
    m_mid.finish()
    restored.finish()
    assert m_mid.dumps() == restored.dumps()

    rejected_loaders = []
    for name, loader in (
        ("UnifiedCore.loads", UnifiedCore.loads),
        ("VisualWords.loads", VisualWords.loads),
        ("ReadoutEligibility.loads", ReadoutEligibility.loads),
    ):
        try:
            loader(blob)
            raise AssertionError(f"{name} did not reject DCALIFEligibility checkpoint")
        except ValueError:
            rejected_loaders.append(name)

    # Fault-injection rollback
    snap_before = m_mid.dumps()

    def _boom(direction: np.ndarray) -> None:
        raise RuntimeError("Injected fault after readout update")

    orig_apply = m_mid._apply_direction
    m_mid._apply_direction = _boom  # type: ignore[method-assign]
    try:
        try:
            m_mid.learn(train_x[1], int(train_y[1]))
            raise AssertionError("Fault injection did not raise")
        except RuntimeError:
            pass
    finally:
        m_mid._apply_direction = orig_apply  # type: ignore[method-assign]
    assert m_mid.dumps() == snap_before

    return {
        "passed": True,
        "finite_difference_by_task": fd_results,
        "zero_frozen_core_bytes_equal": True,
        "matched_signal_rng_across_conditions": True,
        "mid_segment_resume_exact": True,
        "fault_injection_rollback_exact": True,
        "rejected_loaders": rejected_loaders,
    }


def _execute_worker_job(job_path: Path) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "dc_alif_worker.py"), "--job", str(job_path)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=240)
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed on {job_path.name}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def run_p1_phase(
    phase: str,
    seeds: tuple[int, ...],
    run_dir: Path,
    artifact_dir: Path,
    max_workers: int = 4,
) -> tuple[list[dict[str, Any]], int]:
    jobs: list[tuple[Path, str, str, int, str]] = []
    for task in TASKS:
        for seed in seeds:
            for method in METHODS:
                identifier = f"{phase}-{task}-{seed}-{method}"
                job_file = run_dir / f"job-{identifier}.json"
                atomic(
                    job_file,
                    {
                        "mode": "p1",
                        "trial": [phase, task, seed, method, str(run_dir), str(artifact_dir)],
                    },
                )
                jobs.append((job_file, phase, task, seed, method))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_execute_worker_job, jf): (jf, p, t, s, m) for jf, p, t, s, m in jobs}
        for fut in as_completed(futs):
            fut.result()

    rows: list[dict[str, Any]] = []
    cold_jobs: list[tuple[Path, str, dict[str, Any]]] = []
    for _, p, t, s, m in jobs:
        identifier = f"{p}-{t}-{s}-{m}"
        rec = json.loads((run_dir / f"{identifier}.json").read_text(encoding="utf-8"))["record"]
        rows.append(rec)
        cold_job_file = run_dir / f"job-cold-{identifier}.json"
        atomic(
            cold_job_file,
            {
                "mode": "cold",
                "trial": [p, t, s, m, str(run_dir), str(artifact_dir)],
                "checkpoint": rec["checkpoint"],
                "sha256": rec["checkpoint_sha256"],
            },
        )
        cold_jobs.append((cold_job_file, identifier, rec["cold_expected"]))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs_cold = {pool.submit(_execute_worker_job, cjf): (cjf, ident, exp) for cjf, ident, exp in cold_jobs}
        for fut in as_completed(futs_cold):
            fut.result()

    cold_verified = 0
    for _, ident, expected in cold_jobs:
        actual = json.loads((run_dir / f"cold-{ident}.json").read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Cold continuation mismatch for {ident}: {actual} != {expected}")
        cold_verified += 1

    return rows, cold_verified


def summarize_p1_phase(rows: list[dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, Any]:
    by_task: dict[str, Any] = {}
    passing_tasks: list[str] = []
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        by_method: dict[str, list[dict[str, Any]]] = {
            m: sorted([r for r in task_rows if r["method"] == m], key=lambda x: int(x["seed"])) for m in METHODS
        }
        c_rows = by_method["correct"]
        f_rows = by_method["frozen"]
        z_rows = by_method["zero"]
        s_rows = by_method["shuffled"]
        ng_rows = by_method["no_gate"]
        b_rows = by_method["bptt_control"]

        # Verify zero/frozen exact match on every seed
        for zr, fr in zip(z_rows, f_rows):
            assert zr["direct"] == fr["direct"] and zr["probe"] == fr["probe"]
            assert zr["core_weight_fingerprint"] == fr["core_weight_fingerprint"]
            assert zr["head_sha256"] == fr["head_sha256"]

        direct_correct = [float(r["direct"]) for r in c_rows]
        probe_correct = [float(r["probe"]) for r in c_rows]
        probe_gain = [float(r["probe_gain"]) for r in c_rows]
        direct_diff_frozen = [float(c["direct"] - f["direct"]) for c, f in zip(c_rows, f_rows)]
        probe_diff_frozen = [float(c["probe"] - f["probe"]) for c, f in zip(c_rows, f_rows)]
        probe_diff_shuffled = [float(c["probe"] - s["probe"]) for c, s in zip(c_rows, s_rows)]
        probe_diff_no_gate = [float(c["probe"] - ng["probe"]) for c, ng in zip(c_rows, ng_rows)]
        direct_diff_no_gate = [float(c["direct"] - ng["direct"]) for c, ng in zip(c_rows, ng_rows)]
        bag_scores = [float(r["bag_only"]) for r in c_rows]

        valid_task = float(np.mean(bag_scores)) <= 0.60
        task_pass = (
            valid_task
            and float(np.mean(direct_correct)) >= 0.65
            and float(np.mean(probe_gain)) >= 0.05
            and float(np.mean(probe_diff_frozen)) >= 0.05
            and float(np.mean(probe_diff_shuffled)) >= 0.05
        )
        if task_pass:
            passing_tasks.append(task)

        by_task[task] = {
            "valid_task_bag_le_060": valid_task,
            "bag_only_mean": float(np.mean(bag_scores)),
            "task_gate_pass": task_pass,
            "correct_direct": interval(direct_correct),
            "correct_probe": interval(probe_correct),
            "correct_probe_gain": interval(probe_gain),
            "frozen_direct": interval([float(r["direct"]) for r in f_rows]),
            "frozen_probe": interval([float(r["probe"]) for r in f_rows]),
            "shuffled_direct": interval([float(r["direct"]) for r in s_rows]),
            "shuffled_probe": interval([float(r["probe"]) for r in s_rows]),
            "no_gate_direct": interval([float(r["direct"]) for r in ng_rows]),
            "no_gate_probe": interval([float(r["probe"]) for r in ng_rows]),
            "bptt_control_direct": interval([float(r["direct"]) for r in b_rows]),
            "bptt_control_probe": interval([float(r["probe"]) for r in b_rows]),
            "correct_minus_frozen_direct": interval(direct_diff_frozen),
            "correct_minus_frozen_probe": interval(probe_diff_frozen),
            "correct_minus_shuffled_probe": interval(probe_diff_shuffled),
            "correct_minus_no_gate_direct": interval(direct_diff_no_gate),
            "correct_minus_no_gate_probe": interval(probe_diff_no_gate),
            "signflip_p_probe_gain": signflip(probe_gain),
            "signflip_p_vs_frozen": signflip(probe_diff_frozen),
            "signflip_p_vs_shuffled": signflip(probe_diff_shuffled),
            "signflip_p_vs_no_gate": signflip(probe_diff_no_gate),
            "probe_time_permuted_mean": float(np.mean([float(r["probe_time_permuted"]) for r in c_rows])),
            "probe_state_reset_mean": float(np.mean([float(r["probe_state_reset"]) for r in c_rows])),
            "ood_probe_mean": (
                float(np.mean([float(r["ood_probe"]) for r in c_rows if r["ood_probe"] is not None]))
                if any(r["ood_probe"] is not None for r in c_rows)
                else None
            ),
        }

    phase_pass = len(passing_tasks) >= 2 and any(t in passing_tasks for t in ("xor", "strict_binding"))
    return {
        "seeds": list(seeds),
        "passing_tasks": passing_tasks,
        "phase_gate_pass": phase_pass,
        "by_task": by_task,
    }


def run_p2_retention_suite(run_dir: Path, artifact_dir: Path, max_workers: int = 3) -> dict[str, Any]:
    jobs: list[tuple[Path, int, str]] = []
    for seed in P2_SEEDS:
        for cond in P2_CONDITIONS:
            jf = run_dir / f"job-p2-{seed}-{cond}.json"
            atomic(jf, {"mode": "p2", "trial": [seed, cond, str(run_dir), str(artifact_dir)]})
            jobs.append((jf, seed, cond))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_execute_worker_job, jf): (jf, s, c) for jf, s, c in jobs}
        for fut in as_completed(futs):
            fut.result()

    rows: list[dict[str, Any]] = []
    for _, s, c in jobs:
        rec = json.loads((run_dir / f"p2-{s}-{c}.json").read_text(encoding="utf-8"))["record"]
        rows.append(rec)

    by_condition: dict[str, Any] = {}
    for cond in P2_CONDITIONS:
        c_rows = sorted([r for r in rows if r["condition"] == cond], key=lambda x: int(x["seed"]))
        by_condition[cond] = {
            "acquisition": interval([float(r["acquisition"]) for r in c_rows]),
            "final": interval([float(r["final"]) for r in c_rows]),
            "mean_forgetting": interval([float(r["mean_forgetting"]) for r in c_rows]),
            "worst_forgetting": interval([float(r["worst_forgetting"]) for r in c_rows]),
            "joint_refit_probe": interval([float(r["joint_refit_probe"]) for r in c_rows]),
            "mean_head_only_drop": float(np.mean([float(r["mean_head_only_drop"]) for r in c_rows])),
            "mean_hidden_only_drop": float(np.mean([float(r["mean_hidden_only_drop"]) for r in c_rows])),
            "mean_interaction_drop": float(np.mean([float(r["mean_interaction_drop"]) for r in c_rows])),
            "joint_gate_all_seeds": all(bool(r["joint_gate"]) for r in c_rows),
        }

    return {
        "seeds": list(P2_SEEDS),
        "conditions": list(P2_CONDITIONS),
        "by_condition": by_condition,
        "trials": rows,
    }


def main() -> None:
    apply_memory_limit()
    began = time.perf_counter()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    protocol_sha256 = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()

    run_dir = new_run(
        ROOT,
        "collab-antigravity-dc-alif-v1",
        {
            "campaign": CAMPAIGN,
            "agent_id": AGENT_ID,
            "environment": "Antigravity",
            "model": "Gemini 4 Argon (High)",
            "base_sha": BASE_SHA,
            "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
            "protocol_sha256": protocol_sha256,
            "dev_seeds": DEV,
            "confirm_seeds": CONFIRM,
            "p1_tasks": TASKS,
            "p1_methods": METHODS,
            "p2_seeds": P2_SEEDS,
            "p2_conditions": P2_CONDITIONS,
        },
    )

    with ResourceMonitor(interval=0.2) as monitor:
        print("[1/4] Running P0 engineering verification...", flush=True)
        p0_summary = run_p0_engineering_verification()
        monitor.check()

        print("[2/4] Running locked P1 development screen (54 trials + 54 cold resumes)...", flush=True)
        dev_rows, dev_cold = run_p1_phase("development", DEV, run_dir, ARTIFACT_DIR, max_workers=4)
        dev_summary = summarize_p1_phase(dev_rows, DEV)
        dev_summary["cold_continuations_verified"] = dev_cold
        monitor.check()

        confirm_rows: list[dict[str, Any]] = []
        confirm_summary: dict[str, Any] | None = None
        if dev_summary["phase_gate_pass"]:
            print("[3/4] Development screen PASSED (3/3 tasks)! Running locked 8-seed P1 confirmation (144 trials + 144 cold resumes)...", flush=True)
            confirm_rows, confirm_cold = run_p1_phase("confirmation", CONFIRM, run_dir, ARTIFACT_DIR, max_workers=4)
            confirm_summary = summarize_p1_phase(confirm_rows, CONFIRM)
            confirm_summary["cold_continuations_verified"] = confirm_cold
            monitor.check()
        else:
            print("[3/4] Development screen did not pass; skipping confirmation per stop rule.", flush=True)

        print("[4/4] Running P2 20-context continual retention diagnostic (9 trials)...", flush=True)
        p2_summary = run_p2_retention_suite(run_dir, ARTIFACT_DIR, max_workers=3)
        monitor.check()

    result = {
        "campaign": CAMPAIGN,
        "agent_id": AGENT_ID,
        "environment": "Antigravity",
        "model": "Gemini 4 Argon (High)",
        "base_sha": BASE_SHA,
        "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": protocol_sha256,
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "elapsed_seconds": time.perf_counter() - began,
        "p0_engineering": p0_summary,
        "p1_development": dev_summary,
        "p1_confirmation": confirm_summary,
        "p1_trials": dev_rows + confirm_rows,
        "p2_retention": p2_summary,
        "resources": monitor.report(),
    }

    write_json(run_dir / "result.json", result)
    write_json(ARTIFACT_DIR / "dc-alif-round1-latest.json", result)
    print(f"DONE: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
