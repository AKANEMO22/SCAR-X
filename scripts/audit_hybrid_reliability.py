"""Read-only re-score of immutable engineering receipts; no model execution."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def percentile(values, fraction):
    values = sorted(x for x in values if x is not None)
    if not values:
        return None
    pos = (len(values) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "run_id": run.name,
        "source_bundle_exact": True,
        "scope": "software endurance and routing; not lifelong or LoRA qualification",
        "receipts": {},
        "quality": {},
        "metrics": {},
    }
    for rel, digest in manifest["source_sha256"].items():
        if hashlib.sha256((run / "source" / rel).read_bytes()).hexdigest() != digest:
            report["source_bundle_exact"] = False
    for name in ("soak", "cpu", "resume"):
        p = run / (name + ".json")
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            required = (
                {
                    "warm_answer",
                    "GPU_ASR",
                    "automatic_ALIF_verify_and_replay",
                    "explicit_sleep_and_cold_wake",
                    "Hermes_workspace-vi",
                    "Hermes_workspace-en",
                    "Hermes_workspace-short",
                    "Hermes_actual-project",
                    "Hermes_compression",
                }
                if name == "soak"
                else (
                    {"cold_state", "real_answer", "cpu_ASR"}
                    if name == "cpu"
                    else {"cold_state", "real_answer"}
                )
            )
            report["receipts"][name] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "failures": r["failures"],
                "seconds": r["seconds"],
                "gateway_stopped": r.get("gateway_stopped"),
                "placement": r.get("placement"),
                "all_cases_executed": all(
                    c.get("completed", False) for c in r.get("cases", [])
                ),
                "agent_processes_completed": all(
                    c.get("result", {}).get("completed", True)
                    for c in r.get("cases", [])
                    if c["case"].startswith("Hermes_")
                ),
                "required_cases_present": required
                <= {c["case"] for c in r.get("cases", [])},
            }
            if name == "soak":
                report["metrics"]["mixed_elapsed_seconds"] = r.get(
                    "mixed_elapsed_seconds", 0
                )
                report["metrics"]["mixed_requests"] = r.get("mixed_requests", 0)
                report["thirty_minutes_completed"] = (
                    r.get("mixed_elapsed_seconds", 0) >= manifest["duration_seconds"]
                )
                answers = [
                    c["result"]
                    for c in r["cases"]
                    if c.get("completed") and c["case"].startswith("mixed_")
                ]
                report["metrics"]["TTFT_s"] = {
                    "median": (
                        statistics.median(
                            [
                                a["ttft_seconds"]
                                for a in answers
                                if a.get("ttft_seconds") is not None
                            ]
                        )
                        if answers
                        else None
                    ),
                    "p95": percentile([a.get("ttft_seconds") for a in answers], 0.95),
                }
                report["metrics"]["total_response_s"] = {
                    "median": (
                        statistics.median([a["total_seconds"] for a in answers])
                        if answers
                        else None
                    ),
                    "p95": percentile([a["total_seconds"] for a in answers], 0.95),
                }
                report["metrics"]["updates"] = r.get("fingerprint", {}).get("updates")
                report["metrics"]["hidden_updates"] = r.get("fingerprint", {}).get(
                    "hidden_updates"
                )
                report["metrics"]["full_rollback_exact"] = r.get(
                    "full_rollback_exact", False
                )
                report["metrics"]["resources"] = {
                    k: v
                    for k, v in r.get("resources", {}).items()
                    if k.startswith("peak") or k == "limit_bytes"
                }
                report["quality"]["simple_math_outputs"] = [
                    {"answer": a["answer"], "correct": a["answer"].strip() == "15"}
                    for i, a in enumerate(answers, 1)
                    if i % 2 == 0
                ]
                for c in r["cases"]:
                    if c.get("completed") and c["case"] == "GPU_ASR":
                        report["metrics"]["GPU_ASR_s"] = c["seconds"]
                    if (
                        c.get("completed")
                        and c["case"] == "explicit_sleep_and_cold_wake"
                    ):
                        report["metrics"]["cold_wake_TTFT_s"] = c["result"][
                            "cold_answer"
                        ]["ttft_seconds"]
            if name == "cpu":
                for c in r["cases"]:
                    if c.get("completed") and c["case"] == "cpu_ASR":
                        report["metrics"]["CPU_ASR_s"] = c["seconds"]
    workspace = []
    for p in sorted(run.glob("workspace-*.result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        workspace.append(
            {
                "job": r["job"],
                "local_read_used": r.get("local_read_used", False),
                "web_used": r.get("web_used", False),
                "tool_calls": r.get("tool_calls", []),
                "expected_terms_present": r.get("expected_terms_present", {}),
                "answer": r.get("answer", ""),
                "seconds": r["seconds"],
                "completed": r["completed"],
            }
        )
    report["quality"]["workspace_fixture_results"] = workspace
    p = run / "compression.result.json"
    if p.exists():
        r = json.loads(p.read_text(encoding="utf-8"))
        report["quality"]["compression"] = {
            k: r.get(k)
            for k in (
                "before_messages",
                "after_messages",
                "before_characters",
                "after_characters",
                "history_actually_reduced",
                "compression_completed_without_64K_error",
                "facts_retained_in_answer",
                "answer",
                "seconds",
            )
        }
        report["quality"]["compression"]["latest_correct_value_still_in_tail"] = any(
            "9237" in str(m.get("content", ""))
            for m in r.get("compacted_history", [])[-2:]
        )
    for name in ("nvidia-soak", "nvidia-cpu", "nvidia-resume"):
        p = run / (name + ".csv")
        if p.exists():
            samples = []
            for row in csv.reader(
                p.read_text(encoding="utf-8", errors="replace").splitlines()
            ):
                try:
                    samples.append((float(row[1]), float(row[2])))
                except (ValueError, IndexError):
                    continue
            report["metrics"][name] = {
                "samples": len(samples),
                "peak_utilization_percent": max((v[0] for v in samples), default=None),
                "peak_device_memory_MiB": max((v[1] for v in samples), default=None),
                "scope": "whole Nvidia device; may include other applications; use owned-process PDH/backend evidence for placement",
            }
    report["engineering_endurance_pass"] = bool(
        report["source_bundle_exact"]
        and report.get("thirty_minutes_completed")
        and all(k in report["receipts"] for k in ("soak", "cpu", "resume"))
        and all(
            not r["failures"]
            and r["gateway_stopped"]
            and r["all_cases_executed"]
            and r["agent_processes_completed"]
            and r["required_cases_present"]
            for r in report["receipts"].values()
        )
    )
    report["quality"][
        "lifelong_retention"
    ] = "not measured; unchanged20-stage scientific gates"
    report["quality"]["automatic_LoRA"] = "not qualified, disabled"
    report["quality"][
        "hidden_advantage"
    ] = "not measured by device parity or replay counts"
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    result = audit(a.run.resolve())
    raw = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out:
        with a.out.open("x", encoding="utf-8") as f:
            f.write(raw)
    print(raw)
