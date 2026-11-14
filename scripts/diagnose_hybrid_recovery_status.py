"""Fixed code-vocabulary contrast, actual cloned tools; no neural training."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.shared_native import SharedNative
from scarx.evaluation.tooltalk_sandbox import Sandbox, load_executor
from scarx.evaluation.recovery_outcome_bank import verify
from scarx.hybrid.store import Store, encode

CONTRACT = (
    "\nOutput code vocabulary (format definitions, not evidence about this task): "
    "action=call means attempt the named tool, status=working; completion is not yet known. "
    "action=ask means request missing information or authorization. Use status=need_auth "
    "when authentication is missing, status=need_info when user information is missing. "
    "action=finish with status=completed means the requested effects have already occurred "
    "and there is actual supporting evidence. action=defer uses status=deferred. "
    "For non-call actions use tool=none and arguments={}. "
    "The message must agree with the status and actual tool outcomes. "
    "Do not turn a request for login/time into a completed status."
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-recovery-status-diagnostic-v1-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    parent = ROOT / "runs/20261114T025511Z-recovery-outcome-bank-dev-v1-196043"
    previous = json.loads((parent / "result.json").read_text(encoding="utf-8"))
    bankraw = (parent / "locked-cases.json").read_bytes()
    bank = {r["id"]:r for r in json.loads(bankraw)}
    cases = [r for r in previous["cases"] if r["kind"] in ("no_auth", "missing_time")]
    assert len(cases) == 8 and not previous["failures"]
    sources = [
        Path(__file__), ROOT / "src/scarx/evaluation/shared_native.py",
        ROOT / "src/scarx/evaluation/tooltalk_sandbox.py",
        ROOT / "src/scarx/evaluation/tooltalk_schema.py",
        ROOT / "src/scarx/evaluation/recovery_outcome_bank.py",
        ROOT / "experiments/hybrid-recovery-status-contract-diagnostic-v1.md",
    ]
    hashes = {}
    for path in sources:
        raw, rel = path.read_bytes(), path.relative_to(ROOT)
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[rel.as_posix()] = sha(raw)
    (run / "manifest.json").write_text(encode({
        "parent_run": parent.name, "parent_result_sha256": sha((parent / "result.json").read_bytes()),
        "case_bytes_sha256": sha(bankraw), "case_ids": [r["case_id"] for r in cases],
        "actions": ["inspect_status", "clarify"], "source_sha256": hashes,
        "appended_contract": CONTRACT, "new_information_or_gold_in_prompt": False,
        "training_or_promotion": False, "max_unique_native_calls": 48, "max_seconds": 300,
    }), encoding="utf-8")
    intake = json.loads((ROOT / "runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json").read_text())
    upstream = Path(intake["local_snapshot"])
    for name_ in ("LICENSE", "NOTICE"):
        (run / name_).write_bytes((upstream / name_).read_bytes())
    trace = Store(ROOT / "local/hybrid-v1")
    report = {"run_id": name, "pairs": [], "failures": [], "unique_native_calls": 0,
              "SNN_learning_or_VLM_training": False, "live_state_modified": False}
    started = time.monotonic()
    cache = {}
    try:
        native = SharedNative(ROOT)
        executor = load_executor(upstream, intake)
        sandbox = Sandbox(executor, upstream / "data/databases")
        reference = json.loads((upstream / "data/easy/AddAlarm-easy.json").read_text())
        owner = reference["user"]["username"]
        for original in cases:
            case = bank[original["case_id"]]
            user = reference["user"] if case["authorized"] else {
                k:v for k,v in reference["user"].items() if k != "session_token"}
            sandbox.initialize(reference["metadata"], user)
            before = copy.deepcopy(sandbox.executor.databases)
            actual_initial = sandbox.dispatch(case["initial_request"], origin="controlled_fault_injection")
            assert actual_initial == original["initial_result"]
            snapshot = (parent / (case["id"] + "-initial-state.json")).read_bytes()
            assert sandbox.snapshot() == snapshot
            for action in ("inspect_status", "clarify"):
                old = original["outcomes"][action]
                sandbox.restore(snapshot)
                recovery = sandbox.recover(action, origin="exhaustive_action_measurement")
                assert recovery == old["recovery_observation"]
                first_sha = old["model_steps"][0]["input_sha256"]
                prior = json.loads((parent / ("native-" + first_sha + ".json")).read_text(encoding="utf-8"))
                messages = copy.deepcopy(prior["request"]["messages"])
                messages[0]["content"] += CONTRACT
                first_request = {**prior["request"], "messages": copy.deepcopy(messages)}
                records, final = [], {}
                for step in range(3):
                    if time.monotonic() - started > 300 or report["unique_native_calls"] >= 48:
                        raise TimeoutError("Registered status diagnostic budget exhausted")
                    request = {**prior["request"], "messages": messages}
                    digest = sha(encode(request).encode())
                    if digest not in cache:
                        output = native.generate(request)
                        cache[digest] = output
                        report["unique_native_calls"] += 1
                        (run / ("native-" + digest + ".json")).write_text(encode({
                            "request": request, "result": output}), encoding="utf-8")
                    raw = cache[digest]
                    text = raw["choices"][0]["message"].get("content") or ""
                    try:
                        decision = json.loads(text)
                    except ValueError:
                        records.append({"input_sha256": digest, "unparsed": text})
                        break
                    record = {"input_sha256": digest, "decision": decision}
                    records.append(record)
                    if decision.get("action") != "call":
                        final = decision
                        break
                    if action == "clarify":
                        record["execution_rejected"] = "Original terminal recovery scope"
                        break
                    if decision.get("tool") not in ("AddAlarm", "DeleteAlarm"):
                        record["execution_rejected"] = "Outside original advertised capabilities"
                        break
                    try:
                        effect = sandbox.dispatch({
                            "api_name": decision["tool"], "parameters": decision["arguments"]},
                            origin="VLM_after_recovery")
                    except (TypeError, ValueError, KeyError) as exc:
                        effect = {"executed": False, "exception": str(exc), "goal_verified": False}
                    record["actual_tool_result"] = effect
                    public = copy.deepcopy(effect)
                    if "request" in public:
                        public["request"]["parameters"].pop("session_token", None)
                    messages += [{"role": "assistant", "content": text},
                        {"role": "user", "content": "Actual tool outcome:\n" + encode(public)
                         + "\nReturn the next step or supported final status."}]
                outcome = verify(case, before, sandbox.executor.databases, owner, final)
                row = {
                    "case_id": case["id"], "kind": case["kind"], "language": case["language"],
                    "question": case["question"], "action": action,
                    "old_first_input_sha256": first_sha, "first_input_after_only_contract_append": first_request,
                    "old_verified": old["verified_resolution"], "old_final": old["final_decision"],
                    "new_verification": outcome, "new_final": final,
                    "model_steps": records, "after_state_sha256": sha(encode(sandbox.executor.databases).encode()),
                    "state_unchanged": sandbox.executor.databases == before,
                    "label_created": False,
                }
                report["pairs"].append(row)
                trace.event("recovery_status.measured_pair", {
                    k:v for k,v in {"run_id": report["run_id"], **row}.items()
                    if k not in ("model_steps", "first_input_after_only_contract_append")}, "orchestrator")
        for rel, digest in hashes.items():
            assert sha((ROOT / rel).read_bytes()) == digest
        report["old_verified_branches"] = sum(r["old_verified"] for r in report["pairs"])
        report["new_verified_branches"] = sum(r["new_verification"]["verified_resolution"] for r in report["pairs"])
        report["verdict"] = "KNOWN_CASE_STATUS_CONTRACT_DIAGNOSTIC_NO_LEARNING_CLAIM"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "STATUS_DIAGNOSTIC_INCOMPLETE"
    finally:
        report["seconds"] = time.monotonic() - started
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("recovery_status.completed", {
            k:v for k,v in report.items() if k != "pairs"}, "orchestrator")
        trace.close()
        print(encode({k:report.get(k) for k in (
            "run_id", "verdict", "unique_native_calls", "old_verified_branches",
            "new_verified_branches", "failures", "seconds")}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
