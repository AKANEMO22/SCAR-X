"""Actual isolated simulator execution and recovery verification; no LLM/SNN."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
import psutil

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.tooltalk_sandbox import (
    Sandbox, load_executor, verify_single_alarm_transition,
)
from scarx.hybrid.store import Store, encode


def main():
    started = time.monotonic()
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-recovery-executor-p0-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    private = ROOT / "local" / name
    private.mkdir()
    intake_path = ROOT / "runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json"
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    upstream = Path(intake["local_snapshot"])
    source_paths = [
        Path(__file__), ROOT / "src/scarx/evaluation/tooltalk_sandbox.py",
        ROOT / "src/scarx/evaluation/tooltalk_schema.py",
        ROOT / "experiments/hybrid-recovery-executor-p0-v1.md",
    ]
    hashes = {}
    for path in source_paths:
        rel = path.relative_to(ROOT)
        raw = path.read_bytes()
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[rel.as_posix()] = hashlib.sha256(raw).hexdigest()
    for path in ("LICENSE", "NOTICE"):
        (run / path).write_bytes((upstream / path).read_bytes())
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "upstream_revision": intake["revision"],
        "upstream_intake_sha256": hashlib.sha256(intake_path.read_bytes()).hexdigest(),
        "max_seconds": 300, "LLM_SNN_training_or_forward": False,
    }), encoding="utf-8")
    trace = Store(ROOT / "local/hybrid-v1")
    trace.event("recovery_executor.started", {"run_id": name,
        "scope": "actual source APIs on in-memory owned benchmark state; no live model changes"}, "orchestrator")
    blocked = []

    def effect_guard(event, args):
        if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.posix_spawn"):
            blocked.append(event)
            raise PermissionError("P0 disallows network/process side effects")
        if event == "open":
            path, mode, flags = args
            writing = isinstance(mode, str) and any(v in mode for v in ("w", "a", "+", "x"))
            writing |= isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and isinstance(path, (str, bytes, os.PathLike)):
                target = Path(os.fsdecode(path)).resolve()
                if not any(base == target or base in target.parents for base in (run, private)):
                    blocked.append("external_write")
                    raise PermissionError("P0 disallows external writes")

    sys.addaudithook(effect_guard)
    report = {"run_id": name, "failures": [], "conversations": [],
              "model_forward_or_training": False, "live_model_changed": False}
    try:
        executor_class = load_executor(upstream, intake)
        sandbox = Sandbox(executor_class, upstream / "data/databases")
        report["tools"] = sorted(sandbox.schemas)
        report["schema_patches"] = sandbox.schema_patches
        report["nullable_schema_fields"] = sandbox.nullable_fields
        for folder in ("easy", "tooltalk"):
            for path in sorted((upstream / "data" / folder).glob("*.json")):
                case = json.loads(path.read_text(encoding="utf-8"))
                row = {"path": str(path.relative_to(upstream)), "calls": [], "errors": []}
                try:
                    sandbox.initialize(case["metadata"], case.get("user", {}))
                    for turn in case["conversation"]:
                        if turn["role"].lower() == "user":
                            sandbox.begin_turn()
                        for gold in turn.get("apis", []):
                            request = copy.deepcopy(gold["request"])
                            # Source executor injects this token; it is not a
                            # model-call parameter and must not bypass schema.
                            request["parameters"].pop("session_token", None)
                            actual = sandbox.dispatch(request, "annotated_replay")
                            row["calls"].append({
                                "expected": gold, "actual": actual,
                                "exception_matches": actual["exception"] == gold["exception"],
                                "response_exact_matches": actual["response"] == gold["response"],
                            })
                except Exception as exc:
                    row["errors"].append(type(exc).__name__ + ": " + str(exc))
                report["conversations"].append(row)
        sample = json.loads((upstream / "data/easy/AddAlarm-easy.json").read_text(encoding="utf-8"))
        sandbox.initialize(sample["metadata"], {k: v for k, v in sample["user"].items() if k != "session_token"})
        denied = sandbox.dispatch({"api_name": "AddAlarm", "parameters": {"time": "18:30:00"}})
        assert denied["executed"] and denied["exception"] == "User is not logged in"
        status = sandbox.recover("inspect_status")
        schema = sandbox.recover("show_schema")
        assert status["authenticated"] is False and "session_token" not in status
        assert schema["schema"]["name"] == "AddAlarm"
        assert sandbox.recover("clarify")["permitted"] is False
        report["denied_then_bounded_inspection"] = {
            "denied": denied, "status": status, "schema": schema,
            "credentials_or_permission_automatically_changed": False}
        sandbox.initialize(sample["metadata"], sample["user"])
        before = copy.deepcopy(sandbox.executor.databases)
        pin = sandbox.snapshot()
        (private / "sandbox-snapshot.json").write_bytes(pin)
        correct = sandbox.dispatch({"api_name": "AddAlarm", "parameters": {"time": "18:30:00"}})
        after = copy.deepcopy(sandbox.executor.databases)
        verified = verify_single_alarm_transition(before, after, sample["user"]["username"], "18:30:00")
        assert correct["exception"] is None and verified["goal_verified"]
        continuation = sandbox.snapshot()
        sandbox.restore(pin)
        assert sandbox.snapshot() == pin
        repeated = sandbox.dispatch({"api_name": "AddAlarm", "parameters": {"time": "18:30:00"}})
        assert repeated == correct and sandbox.snapshot() == continuation
        sandbox.restore(pin)
        wrong = sandbox.dispatch({"api_name": "AddAlarm", "parameters": {"time": "09:00:00"}})
        rejected = verify_single_alarm_transition(before, sandbox.executor.databases, sample["user"]["username"], "18:30:00")
        assert wrong["exception"] is None and not rejected["goal_verified"]
        report["independent_goal_verifier"] = {
            "correct_actual_call": correct, "correct_transition": verified,
            "wrong_but_successful_actual_call": wrong, "wrong_transition": rejected,
            "data_only_restore_exact": True, "RNG_next_call_exact": True,
            "snapshot_sha256": hashlib.sha256(pin).hexdigest(),
        }
        sandbox.initialize(sample["metadata"], {k: v for k, v in sample["user"].items() if k != "session_token"})
        calls = [sandbox.dispatch({"api_name": "AddAlarm", "parameters": {"time": "18:30:00"}}) for _ in range(4)]
        assert all(r["executed"] for r in calls[:3]) and calls[3]["origin"] == "runtime_guard"
        report["real_repeat_guard"] = calls
        invalid = []
        for request in [
            {"api_name": "Unknown", "parameters": {}},
            {"api_name": "AddAlarm", "parameters": {"time": "a" * 4001}},
        ]:
            try:
                sandbox.dispatch(request)
            except ValueError:
                invalid.append(True)
        assert len(invalid) == 2
        # Bad checkpoint must not partially replace actual simulator state.
        original = sandbox.snapshot()
        corrupt = json.loads(original)
        corrupt["tool_rng"]["AddAlarm"] = [1, 2]
        try:
            sandbox.restore(json.dumps(corrupt).encode())
            raise AssertionError("Corrupt RNG accepted")
        except (ValueError, TypeError):
            pass
        assert sandbox.snapshot() == original
        report["corrupt_restore_atomic"] = True
        report["extra_neural_dependencies_loaded"] = [
            n for n in sys.modules if n.startswith(("sent2vec", "torch", "transformers"))]
        assert not report["extra_neural_dependencies_loaded"]
        assert time.monotonic() - started < 300
        for rel, digest in hashes.items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
        report["verdict"] = "EXECUTOR_RECOVERY_VERIFIER_TECHNICAL_PASS"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "P0_FAIL_DO_NOT_INFER"
    finally:
        report["seconds"] = time.monotonic() - started
        report["blocked_side_effects"] = blocked
        rows = [c for row in report["conversations"] for c in row["calls"]]
        report["annotated_replay"] = {
            "conversations": len(report["conversations"]), "calls": len(rows),
            "conversation_errors": sum(bool(r["errors"]) for r in report["conversations"]),
            "exception_mismatches": sum(not c["exception_matches"] for c in rows),
            "response_mismatches": sum(not c["response_exact_matches"] for c in rows),
            "interpretation": "Dataset/harness diagnostic, not LLM/SNN performance"}
        report["dataset_qualified_for_learning"] = False
        report["dataset_qualification_reason"] = (
            "State/trace oracle and semantic/ID/order mismatches need separate "
            "qualification; raw response inequality alone is not a label error")
        (run / "result.json").write_text(encode(report), encoding="utf-8")
        trace.event("recovery_executor.completed", {
            k: v for k, v in report.items() if k not in ("conversations",)}, "orchestrator")
        trace.close()
        print(encode({k: report[k] for k in ("run_id", "verdict", "annotated_replay", "seconds")}))
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
