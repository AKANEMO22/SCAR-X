"""Immutable P0: actual file faults, actual serving encoder, no neural training."""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.config import HybridConfig
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.store import Store, encode
from scarx.evaluation.hybrid_incident_encoder import observation_sequence, FORMAT


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-incident-information-p0-" + uuid.uuid4().hex[:6]
    out = ROOT / "runs" / name
    out.mkdir()
    paths = list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        Path(__file__), ROOT / "scripts/hermes_incident_information_fixture.py",
        ROOT / "src/scarx/evaluation/hybrid_incident_encoder.py",
        ROOT / "integrations/hermes/hybrid_plugin/learning_hooks.py",
        ROOT / "experiments/hybrid-incident-information-p0-v1.md",
    ]
    source = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        raw = path.read_bytes()
        dest = out / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        source[rel.as_posix()] = sha(raw)
    (out / "manifest.json").write_text(encode({
        "source_sha256": source, "started_epoch": time.time(),
        "base_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "max_seconds": 300, "neural_forward_or_training_authorized": False,
    }), encoding="utf-8")
    trace = Store(ROOT / "local/hybrid-v1")
    trace.event("incident_information.started", {"run_id": name, "scope": "P0 actual fault/encoder audit, no training"}, "orchestrator")
    started = time.monotonic()
    report = {"run_id": name, "failures": [], "records": [],
              "neural_training": False, "live_model_changed": False}
    store = None
    try:
        helper = ROOT / "scripts/hermes_incident_information_fixture.py"
        python = ROOT / "local/hermes-reliability-test-env/Scripts/python.exe"
        with (out / "fixture.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [str(python), str(helper), str(out / "actual-file-observations.json")],
                cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                stdout=log, stderr=subprocess.STDOUT, timeout=120,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        assert process.returncode == 0, "Actual tool fixture process failed"
        actual = json.loads((out / "actual-file-observations.json").read_text(encoding="utf-8"))
        assert not actual["errors"] and len(actual["cases"]) == 3, "Incomplete actual fixture bank"
        spec = importlib.util.spec_from_file_location(
            "information_audit_hooks", ROOT / "integrations/hermes/hybrid_plugin/learning_hooks.py")
        hooks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hooks)
        store = Store(ROOT / "local" / name)
        monitor = SimpleNamespace(check=lambda: {"rss_bytes": 0}, sample=lambda: {"rss_bytes": 0})
        engine = HybridEngine(store, SimpleNamespace(), monitor, HybridConfig())
        original_arrays = {k: sha(v.tobytes()) for k, v in engine.model.arrays().items()}
        old_groups, new_groups = defaultdict(list), defaultdict(list)
        for case in actual["cases"]:
            assert not case["requested_exists"]
            assert len(case["missing_outputs"]) == 3 and len(case["probe_outputs"]) == 1
            first = json.loads(case["missing_outputs"][0])
            assert first.get("not_found") is True, "Fixture did not produce an actual missing-file result"
            assert all(json.loads(v) == first for v in case["missing_outputs"]), "Repeated result changed"
            probe = case["probe"]
            assert probe is not None and probe["semantic_resolution_verified"] is False
            if case["alternative_is_file"]:
                assert probe["read_ok"] is True
                next_step = "inspect_readable_bound_source"
            elif case["alternative_is_directory"]:
                assert not probe["read_ok"] and not probe["reported_not_found"]
                next_step = "clarify_file_versus_directory"
            else:
                assert not case["alternative_exists"] and probe["reported_not_found"]
                next_step = "request_missing_source"
            observations = []
            result = None
            for j, value in enumerate(case["missing_outputs"]):
                payload = hooks.tool_observation(
                    "read_file", {"path": case["requested_path"]}, value,
                    hooks.digest(case["id"]), f"call-{j}", "unknown",
                )
                if j == 2:
                    payload["probe"] = probe
                result = engine.observe_tool_experience(payload)
                observations.append({
                    "tool": payload["tool"], "reported_status": payload["reported_status"],
                    "error_type": payload["error_type"], "empty_result": payload["empty_result"],
                    "spillover": payload["spillover"], "streak": j + 1,
                    "probe": None if j < 2 else {k: probe[k] for k in (
                        "kind", "read_ok", "reported_not_found", "original_path_differs")},
                })
            incident = result["incident"]
            assert incident and incident["goal_id"] is None
            # Engine attaches the goal in the persisted audit row.
            saved = store.get("tool_incident", incident["id"])
            goal = engine.state["goals"][saved["goal_id"]]
            old = engine.goal_sequence(goal)
            proposed = observation_sequence(goal["question"], observations)
            old_hash, new_hash = sha(old.tobytes()), sha(proposed.tobytes())
            old_groups[old_hash].append(next_step)
            new_groups[new_hash].append(next_step)
            record = {
                "case_id": case["id"], "next_step_annotation": next_step,
                "annotation_is_resolved_task": False,
                "actual_probe": probe, "goal_question": goal["question"],
                "goal_reason": goal["reason"], "goal_flags": {
                    k: goal[k] for k in ("attempts", "reviewed", "searched", "verified_attempted", "evidence_ids")},
                "current_input_sha256": old_hash, "proposed_input_sha256": new_hash,
                "proposed_encoder": FORMAT, "current_shape": list(old.shape),
                "proposed_shape": list(proposed.shape),
            }
            np.save(out / (case["id"] + "-current.npy"), old, allow_pickle=False)
            np.save(out / (case["id"] + "-proposed.npy"), proposed, allow_pickle=False)
            report["records"].append(record)
            trace.event("incident_information.measured_case", {
                "run_id": name, **record,
                "actual_missing_tool_output": json.loads(case["missing_outputs"][0]),
                "actual_probe_tool_output": json.loads(case["probe_outputs"][0]),
                "production_neural_update": False,
            }, "orchestrator")
        n = len(report["records"])
        report["information_bound"] = {
            "N_engineering_cases": n,
            "current_input_groups": len(old_groups), "proposed_input_groups": len(new_groups),
            "current_deterministic_label_ceiling": sum(max(Counter(v).values()) for v in old_groups.values()) / n,
            "proposed_deterministic_label_ceiling": sum(max(Counter(v).values()) for v in new_groups.values()) / n,
            "scope": "exact equality bound on this finite bank; not model accuracy/generalization or a statistical CI",
        }
        report["zero_synapse_updates"] = engine.model.updates == engine.model.hidden_updates == 0
        report["arrays_unchanged"] = original_arrays == {
            k: sha(v.tobytes()) for k, v in engine.model.arrays().items()}
        payload = engine.model.dumps(engine.state)
        restored, state = LocalALIF.loads(payload)
        report["data_only_cold_restore_exact"] = restored.dumps(state) == payload
        assert report["zero_synapse_updates"] and report["arrays_unchanged"] and report["data_only_cold_restore_exact"]
        assert time.monotonic() - started < 300, "Registered budget exceeded"
        for rel, digest in source.items():
            assert sha((ROOT / rel).read_bytes()) == digest, "Source changed during P0: " + rel
        report["verdict"] = "CURRENT_INCIDENT_INPUT_LOSES_DIAGNOSTIC_INFORMATION"
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
        report["verdict"] = "P0_INCOMPLETE_DO_NOT_INFER"
    finally:
        report["seconds"] = time.monotonic() - started
        (out / "result.json").write_text(encode(report), encoding="utf-8")
        trace.event("incident_information.completed", {
            k: v for k, v in report.items() if k != "records"}, "orchestrator")
        trace.close()
        if store:
            store.close()
        print(encode({k: report[k] for k in ("run_id", "failures", "verdict", "seconds")}))
    return not report["failures"]


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(0 if run() else 1)
