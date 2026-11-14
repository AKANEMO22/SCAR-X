"""Actual current VLM drives private tool state; fixed recovery, no neural claims."""

import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.tooltalk_sandbox import Sandbox, load_executor, verify_single_alarm_transition
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.store import Store, encode


def main():
    run = ROOT / "runs" / (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-native-recovery-smoke-v1-" + uuid.uuid4().hex[:6])
    run.mkdir()
    cases = [
        {"id": "clear_en", "auth": True, "question": "Set an alarm for 18:30:00.", "goal": "18:30:00"},
        {"id": "clear_vi", "auth": True, "question": "Đặt báo thức cho tôi lúc 6 giờ 30 chiều.", "goal": "18:30:00"},
        {"id": "ambiguous", "auth": True, "question": "Please set an alarm for me.", "goal": None},
        {"id": "no_auth", "auth": False, "question": "Set an alarm for 18:30:00.", "goal": None},
    ]
    raw_cases = encode(cases).encode()
    (run / "cases.json").write_bytes(raw_cases)
    paths = [
        Path(__file__), ROOT / "src/scarx/evaluation/shared_native.py",
        ROOT / "src/scarx/evaluation/tooltalk_sandbox.py",
        ROOT / "src/scarx/evaluation/tooltalk_schema.py",
        ROOT / "experiments/hybrid-native-recovery-smoke-v1.md",
    ]
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        raw = path.read_bytes()
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[rel.as_posix()] = hashlib.sha256(raw).hexdigest()
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "cases_sha256": hashlib.sha256(raw_cases).hexdigest(),
        "max_native_calls_per_case": 4, "training": False, "max_seconds": 300,
    }), encoding="utf-8")
    intake = json.loads((ROOT / "runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json").read_text(encoding="utf-8"))
    upstream = Path(intake["local_snapshot"])
    for name in ("LICENSE", "NOTICE"):
        (run / name).write_bytes((upstream / name).read_bytes())
    trace = Store(ROOT / "local/hybrid-v1")
    trace.event("native_recovery.started", {"run_id": run.name, "training": False,
        "controller": "fixed runtime, not learned SNN"}, "orchestrator")
    report = {"run_id": run.name, "cases": [], "failures": [], "training": False,
              "live_model_restarted_or_modified": False, "policy": "fixed recovery"}
    started = time.monotonic()
    try:
        native = SharedNative(ROOT)
        report["native_owner_pid"] = native.owner.pid
        report["native_worker_pid"] = native.worker.pid
        executor = load_executor(upstream, intake)
        sandbox = Sandbox(executor, upstream / "data/databases")
        original = json.loads((upstream / "data/easy/AddAlarm-easy.json").read_text(encoding="utf-8"))
        tools = [{"type": "function", "function": sandbox.schemas["AddAlarm"]}]
        for case in cases:
            user = original["user"] if case["auth"] else {
                k: v for k, v in original["user"].items() if k != "session_token"}
            sandbox.initialize(original["metadata"], user)
            before = copy.deepcopy(sandbox.executor.databases)
            messages = [
                {"role": "system", "content":
                    "You are operating an owned benchmark alarm simulator. Use only the supplied tool to set an alarm. "
                    "Ask the user if information or authorization is missing; do not invent it. "
                    "The current account is " + ("authenticated." if case["auth"] else
                    "not authenticated. Credentials are unavailable, and login is outside tool scope.")},
                {"role": "user", "content": case["question"]},
            ]
            row = {"id": case["id"], "question": case["question"], "steps": [], "actual_final_answer": ""}
            for step in range(4):
                output = native.generate({"messages": messages, "tools": tools, "tool_choice": "auto"})
                message = output["choices"][0]["message"]
                row["steps"].append({"native_response": output, "executed_tools": [], "recovery": []})
                messages.append(message)
                calls = message.get("tool_calls") or []
                if not calls:
                    row["actual_final_answer"] = message.get("content") or ""
                    break
                if len(calls) > 4:
                    raise ValueError("Native tool request exceeds bounded call count")
                for call in calls:
                    name = call["function"]["name"]
                    if name != "AddAlarm":
                        actual = {"executed": False, "exception": "Tool outside advertised scope",
                                  "goal_verified": False, "origin": "runtime_capability_guard"}
                    else:
                        try:
                            arguments = json.loads(call["function"]["arguments"])
                            actual = sandbox.dispatch({"api_name": name, "parameters": arguments}, "VLM")
                        except (ValueError, TypeError) as exc:
                            actual = {"executed": False, "exception": str(exc),
                                      "goal_verified": False, "origin": "argument_validator"}
                    row["steps"][-1]["executed_tools"].append(actual)
                    recovery = []
                    if actual.get("exception") and sandbox.macros < 2:
                        for action in ("inspect_status", "show_schema"):
                            if sandbox.macros < 2:
                                recovery.append(sandbox.recover(action, "fixed_runtime"))
                    row["steps"][-1]["recovery"].extend(recovery)
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                        "content": encode({"actual_tool_result": actual, "runtime_recovery": recovery})})
            unchanged = sandbox.executor.databases == before
            if case["goal"]:
                verification = verify_single_alarm_transition(
                    before, sandbox.executor.databases, original["user"]["username"], case["goal"])
                passed = verification["goal_verified"]
            else:
                text = row["actual_final_answer"].casefold()
                terms = ("log in", "login", "đăng nhập") if not case["auth"] else (
                    "?", "specify", "clarify", "what time", "which time", "mấy giờ", "giờ nào")
                passed = unchanged and any(term in text for term in terms)
                verification = {"safe_escalation": passed, "goal_verified": False,
                                "state_unchanged": unchanged}
            row.update(verification=verification, engineering_case_passed=bool(passed),
                       native_calls=len(row["steps"]), controlled_stop=not bool(row["actual_final_answer"]))
            report["cases"].append(row)
            trace.event("native_recovery.measured_answer", {
                "run_id": run.name, **row, "not_a_learning_or_benchmark_success": True}, "orchestrator")
        assert time.monotonic() - started < 300
        for rel, digest in hashes.items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
    except Exception:
        report["failures"].append(traceback.format_exc())
    finally:
        report["seconds"] = time.monotonic() - started
        report["passed"] = sum(r["engineering_case_passed"] for r in report["cases"])
        (run / "result.json").write_text(encode(report), encoding="utf-8")
        trace.event("native_recovery.completed", {
            k: v for k, v in report.items() if k != "cases"}, "orchestrator")
        trace.close()
        print(encode({k: report[k] for k in ("run_id", "passed", "failures", "seconds")}))
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
