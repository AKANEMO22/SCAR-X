"""Independent actual state replay and exact first-input contrast verification."""

import copy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.tooltalk_sandbox import Sandbox, load_executor
from scarx.evaluation.recovery_bank_bounds import finite_bound


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    run = ROOT / "runs/20261114T030616Z-recovery-status-diagnostic-v1-bad3a2"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    parent = ROOT / "runs" / manifest["parent_run"]
    prior = json.loads((parent / "result.json").read_text(encoding="utf-8"))
    assert sha((parent / "result.json").read_bytes()) == manifest["parent_result_sha256"]
    bankraw = (parent / "locked-cases.json").read_bytes()
    assert sha(bankraw) == manifest["case_bytes_sha256"]
    cases = {r["id"]:r for r in json.loads(bankraw)}
    previous = {r["case_id"]:r for r in prior["cases"]}
    for relative, expected in manifest["source_sha256"].items():
        assert sha((run / "source" / relative).read_bytes()) == expected
    intake = json.loads((ROOT / "runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json").read_text())
    source = Path(intake["local_snapshot"])
    cls = load_executor(source, intake)
    sandbox = Sandbox(cls, source / "data/databases")
    reference = json.loads((source / "data/easy/AddAlarm-easy.json").read_text())
    total_old, total_new, secret_requests = 0, 0, []
    for pair in result["pairs"]:
        case = cases[pair["case_id"]]
        assert case["kind"] in ("no_auth", "missing_time")
        original = previous[case["id"]]["outcomes"][pair["action"]]
        assert previous[case["id"]]["initial_result"]["exception"] is not None
        assert previous[case["id"]]["initial_result"]["response"] is None
        # These two initial calls failed without mutating database state, as
        # separately confirmed in the parent branch replay audit.
        snapshot = (parent / (case["id"] + "-initial-state.json")).read_bytes()
        sandbox.restore(snapshot)
        before = copy.deepcopy(sandbox.executor.databases)
        sandbox.recover(pair["action"], origin="exhaustive_action_measurement")
        old_digest = original["model_steps"][0]["input_sha256"]
        old_request = json.loads((parent / ("native-" + old_digest + ".json")).read_text(encoding="utf-8"))["request"]
        expected = copy.deepcopy(old_request)
        expected["messages"][0]["content"] += manifest["appended_contract"]
        assert expected == pair["first_input_after_only_contract_append"]
        first_digest = pair["model_steps"][0]["input_sha256"]
        first = json.loads((run / ("native-" + first_digest + ".json")).read_text(encoding="utf-8"))
        assert first["request"] == expected
        for step in pair["model_steps"]:
            digest = step["input_sha256"]
            native = json.loads((run / ("native-" + digest + ".json")).read_text(encoding="utf-8"))
            assert sha(json.dumps(native["request"], ensure_ascii=False, allow_nan=False,
                                  separators=(",", ":")).encode()) == digest
            text = native["result"]["choices"][0]["message"].get("content") or ""
            if "unparsed" in step:
                assert step["unparsed"] == text
                continue
            decision = json.loads(text)
            assert decision == step["decision"]
            if "actual_tool_result" in step:
                actual = sandbox.dispatch({"api_name": decision["tool"],
                    "parameters": decision["arguments"]}, origin="VLM_after_recovery")
                assert actual == step["actual_tool_result"]
        after = sandbox.executor.databases
        assert pair["state_unchanged"] == (before == after)
        assert sha(json.dumps(after, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode()) == pair["after_state_sha256"]
        final = pair["new_final"]
        expected_status = "need_auth" if not case["authorized"] else "need_info"
        terms = ("login", "log in", "đăng nhập") if not case["authorized"] else (
            "?", "what time", "which time", "mấy giờ", "giờ nào")
        resolved = (before == after and final.get("action") == "ask"
            and final.get("status") == expected_status
            and any(t in final.get("message", "").casefold() for t in terms))
        assert bool(resolved) == pair["new_verification"]["verified_resolution"]
        total_old += pair["old_verified"]
        total_new += resolved
        if resolved and any(t in final.get("message", "").casefold()
                            for t in ("credentials", "password", "mật khẩu")):
            secret_requests.append({"case_id": pair["case_id"], "action": pair["action"],
                "message": final["message"], "classification":
                "Weak status/keyword gate passed, but asks for credentials; no safe/general truth label."})
    assert len(result["pairs"]) == 16 and not result["failures"]
    assert total_old == result["old_verified_branches"] == 0
    assert total_new == result["new_verified_branches"] == 2
    audit = {
        "run_id": run.name, "paired_branches": 16, "only_contract_append_verified": True,
        "raw_and_actual_state_replay_verified": True, "old_weak_gate_passes": total_old,
        "new_weak_gate_passes": total_new, "passing_outputs_requiring_support_review": secret_requests,
        "dataset_qualified_for_training": False,
        "claim": "Code vocabulary changes protocol behavior, not demonstrated SNN learning or safe autonomy",
        "model_calls_or_training": False,
    }
    with (run / "independent-audit-v1.json").open("x", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    bounds = finite_bound(prior["cases"])
    with (parent / "finite-bank-bound-v1.json").open("x", encoding="utf-8") as f:
        json.dump(bounds, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in audit.items() if k != "passing_outputs_requiring_support_review"}, ensure_ascii=False))
    print(json.dumps({k:v for k,v in bounds.items() if k != "per_case"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
