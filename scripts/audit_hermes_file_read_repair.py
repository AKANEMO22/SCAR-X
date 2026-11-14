"""Compare real before/after reads and canonical per-file baseline failures."""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def failures(text):
    return {m.group(1).replace("\\", "/") for m in re.finditer(r"\bFAILED\s+(tests[^\s]+)", text)}


def main(run):
    baseline = ROOT / "runs/20261114T051425Z-file-contract-p0-v1-3ce157"
    before = json.loads((baseline / "baseline-actual-file-results.json").read_text(encoding="utf-8"))
    after = json.loads((run / "candidate-actual-file-results.json").read_text(encoding="utf-8"))
    refactor = json.loads((run / "refactor-v2/actual-three-cases.json").read_text(encoding="utf-8"))
    assert all(not r["failures"] for r in (before, after, refactor))
    assert len(before["calls"]) == len(after["calls"]) == 14 and len(refactor["calls"]) == 3
    old, new = ({r["case"]:r for r in rows["calls"]} for rows in (before,after))
    raw = lambda row: json.loads(row["raw_tool_result"])
    assert raw(old["empty"]) == raw(old["exclusive"])
    assert raw(new["empty"]) == raw(old["empty"])
    assert raw(new["exclusive"]).get("error") and new["exclusive"]["public_hook_observation"]["reported_status"] == "error"
    for case in ("readable","empty","misleading","binary","partial","full"):
        assert raw(new[case]) == raw(old[case]), case
    for case in ("missing","deleted"):
        assert raw(new[case])["not_found"] and raw(old[case])["not_found"]
    assert raw(new["directory"]).get("error") and not raw(new["directory"])["not_found"]
    for row in refactor["calls"]:
        assert raw(row) == raw(new[row["case"]])
    total_calls = len(after["calls"]) + len(refactor["calls"])
    assert total_calls <= 20
    canonical = ROOT / "local/hermes-file-read-contract-v2"
    logs = {name:(canonical / ("canonical-" + name + ".log")).read_text(encoding="utf-8")
            for name in ("baseline","candidate")}
    for name, text in logs.items():
        assert "clean env" in text and "4 files" in text and "100% complete" in text
    base_fail, candidate_fail = failures(logs["baseline"]), failures(logs["candidate"])
    regression = {name for name in base_fail if "test_scarx_file_read_contract.py::" in name}
    assert len(regression) == 9 and regression.isdisjoint(candidate_fail)
    assert not candidate_fail - base_fail
    assert re.search(r"test_scarx_file_read_contract\.py\s+\(11✓", logs["candidate"])
    for name, text in logs.items():
        with (run / f"canonical-{name}-v2.log").open("x",encoding="utf-8") as f:f.write(text)
    version = run / "refactor-v2"
    manifest = json.loads((version / "manifest.json").read_text())
    for rel, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((version / "source" / rel).read_bytes()).hexdigest() == expected
    for name, key in (("file_operations.py","candidate_sha256"),("file_operations_read_contract.py","small_sibling_sha256")):
        assert hashlib.sha256((version / name).read_bytes()).hexdigest() == manifest[key]
    result = {"run_id":run.name,"actual_calls_v1_plus_refactor":total_calls,
        "real_locked_read_now_reports_error":True,"true_empty_and_other_readable_cases_unchanged":True,
        "refactor_preserves_v1_actual_three_cases_exactly":True,
        "canonical_new_contract_tests_passed":11,"baseline_contract_test_failures_repaired":sorted(regression),
        "baseline_other_failures":sorted(base_fail-regression),"candidate_existing_failures":sorted(candidate_fail),
        "new_canonical_failures":[],"whole_upstream_suite_green":False,
        "baseline_only_unrelated_test_not_claimed_fixed":sorted(base_fail-regression-candidate_fail),
        "core_before_sha256":"d33e0bc43516f9c5da4dd1ab75bd57b03837034c5c4accb91df1d78b92131b51",
        "core_after_sha256":manifest["candidate_sha256"],"sibling_sha256":manifest["small_sibling_sha256"],
        "source_artifacts":"refactor-v2", "no_neural_learning_or_architecture_changes":True,
        "live_activation_verified":False,
        "verdict":"ENGINEERING_READ_CONTRACT_REPAIR_QUALIFIED_WITH_EXISTING_OS_TEST_FAILURES"}
    with (run / "engineering-acceptance-v1.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},ensure_ascii=False))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path)
    main(p.parse_args().run.resolve())
