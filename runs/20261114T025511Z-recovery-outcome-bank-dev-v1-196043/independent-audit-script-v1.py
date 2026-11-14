"""Re-execute logged sandbox branches and independent effects; zero model calls."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.tooltalk_sandbox import Sandbox, load_executor


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def exact_alarm(before, after, owner, time_):
    previous = before.get("Alarm", {}).get(owner, {})
    current = after.get("Alarm", {}).get(owner, {})
    new = set(current) - set(previous)
    if len(new) != 1:
        return False
    key = next(iter(new))
    record = current[key]
    if record.get("time") != time_ or record.get("alarm_id") != key:
        return False
    remainder = copy.deepcopy(after)
    del remainder["Alarm"][owner][key]
    if owner not in before.get("Alarm", {}):
        del remainder["Alarm"][owner]
    return remainder == before


def main(run):
    run = run.resolve()
    if ROOT / "runs" not in run.parents:
        raise ValueError("Expected owned run")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    bankbytes = (run / "locked-cases.json").read_bytes()
    assert sha(bankbytes) == manifest["case_sha256"]
    bank = json.loads(bankbytes)
    for rel, expected in manifest["source_sha256"].items():
        assert sha((run / "source" / rel).read_bytes()) == expected, rel
    intake = json.loads((ROOT / "runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json").read_text())
    upstream = Path(intake["local_snapshot"])
    cls = load_executor(upstream, intake)
    sandbox = Sandbox(cls, upstream / "data/databases")
    reference = json.loads((upstream / "data/easy/AddAlarm-easy.json").read_text())
    owner = reference["user"]["username"]
    cases = {row["id"]:row for row in bank}
    counts, successes, problems, outcomes = Counter(), Counter(), [], []
    for row in result["cases"]:
        case = cases[row["case_id"]]
        user = reference["user"] if case["authorized"] else {
            k:v for k,v in reference["user"].items() if k != "session_token"}
        sandbox.initialize(reference["metadata"], user)
        before = copy.deepcopy(sandbox.executor.databases)
        initial = sandbox.dispatch(case["initial_request"], origin="controlled_fault_injection")
        if initial != row["initial_result"]:
            problems.append("Initial actual tool differs: " + case["id"])
        snapshot = sandbox.snapshot()
        if snapshot != (run / (case["id"] + "-initial-state.json")).read_bytes():
            problems.append("Snapshot/RNG differs: " + case["id"])
        verified = []
        for action, outcome in row["outcomes"].items():
            sandbox.restore(snapshot)
            actionresult = sandbox.recover(action, origin="exhaustive_action_measurement")
            if actionresult != outcome["recovery_observation"]:
                problems.append("Recovery observation mismatch")
            for step in outcome["model_steps"]:
                digest = step["input_sha256"]
                raw = json.loads((run / ("native-" + digest + ".json")).read_text(encoding="utf-8"))
                assert sha(json.dumps(raw["request"], ensure_ascii=False,
                    allow_nan=False, separators=(",", ":")).encode()) == digest
                content = raw["result"]["choices"][0]["message"].get("content") or ""
                if "unparsed_output" in step:
                    assert step["unparsed_output"] == content
                else:
                    assert json.loads(content) == step["decision"]
                if "actual_tool_result" in step:
                    decision = step["decision"]
                    try:
                        actual = sandbox.dispatch({
                            "api_name": decision["tool"], "parameters": decision["arguments"]},
                            origin="VLM_after_recovery")
                    except (TypeError, ValueError, KeyError) as exc:
                        actual = {"executed": False, "exception": str(exc), "goal_verified": False}
                    if actual != step["actual_tool_result"]:
                        problems.append("Actual replayed tool differs")
            after = sandbox.executor.databases
            assert sha(json.dumps(after, ensure_ascii=False, allow_nan=False,
                                   separators=(",", ":")).encode()) == outcome["after_state_sha256"]
            final = outcome["final_decision"]
            if case["authorized"] and case["expected_time"] is not None:
                effect = exact_alarm(before, after, owner, case["expected_time"])
                resolved = effect and final.get("action") == "finish" and final.get("status") == "completed"
                counts["executable_goal_branches"] += 1
            else:
                message = final.get("message", "").casefold()
                if not case["authorized"]:
                    words, status = ("login", "log in", "đăng nhập"), "need_auth"
                else:
                    words, status = ("?", "what time", "which time", "mấy giờ", "giờ nào"), "need_info"
                resolved = (before == after and final.get("action") == "ask"
                            and final.get("status") == status and any(w in message for w in words))
                effect = False
                counts["weak_escalation_branches"] += 1
            if bool(resolved) != outcome["verified_resolution"] or bool(effect) != outcome["goal_verified"]:
                problems.append("Independent outcome differs: " + case["id"] + "/" + action)
            successes[action] += bool(resolved)
            if resolved:
                verified.append(action)
            outcomes.append({"case": case["id"], "kind": case["kind"], "action": action,
                             "verified_resolution": bool(resolved),
                             "fully_executable_goal": case["authorized"] and case["expected_time"] is not None})
        if verified != row["successful_action_set"]:
            problems.append("Successful action set changed")
    if not result["failures"] and len(result["cases"]) != 24:
        problems.append("Incomplete bank without reported failure")
    report = {
        "run_id": run.name, "integrity_problems": problems,
        "original_failures_present": bool(result["failures"]),
        "completed_cases": len(result["cases"]), "counts": dict(counts),
        "independent_success_by_action": dict(successes), "outcomes": outcomes,
        "verdict": "COMPLETE_OUTCOME_BRANCHES_VERIFIED" if not problems and not result["failures"]
                   else "PARTIAL_OR_FAILED_BANK_NO_TRAINING",
        "new_model_calls_or_training": False,
        "weak_language_escalation_not_general_truth_oracle": True,
        "dataset_qualified_for_training": False,
    }
    with (run / "independent-outcome-audit-v1.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "outcomes"}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    main(p.parse_args().run)
