"""Verify actual reader fault outcomes and preserve canonical baseline failures."""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KNOWN_DEDUP = "tests/tools/test_file_read_guards.py::TestFileDedup::test_background_review_fork_gets_content_and_read_mark_not_stub"


def failed_ids(text):
    return {m.group(1).strip().replace("\\", "/") for m in re.finditer(r"\bFAILED\s+(tests[^\r\n]+)", text)}


def main(run):
    run = run.resolve()
    assert run.is_relative_to(ROOT / "runs")
    manifest = json.loads((run / "manifest.json").read_text())
    for rel, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((run / "source" / rel).read_bytes()).hexdigest() == expected
    failed = json.loads((run / "candidate-actual.json").read_text(encoding="utf-8"))
    assert len(failed["calls"]) == 0 and failed["failures"], "Harness failure must remain separate"
    before = json.loads((run / "baseline-actual.json").read_text(encoding="utf-8"))
    after = json.loads((run / "candidate-repaired-harness-actual.json").read_text(encoding="utf-8"))
    assert all(not r["failures"] and len(r["calls"]) == 6 for r in (before, after))
    expected = ("compound_ok", "compound_page_failure", "sequential_ok",
                "sequential_page_failure", "sequential_empty", "sequential_locked")
    assert [c["case"] for c in before["calls"]] == [c["case"] for c in after["calls"]] == list(expected)
    pairs = []
    for a, b in zip(before["calls"], after["calls"]):
        av, bv = json.loads(a["raw_tool_result"]), json.loads(b["raw_tool_result"])
        assert a["controlled_transport_fault_not_neural_input"] == b["controlled_transport_fault_not_neural_input"]
        if a["case"].endswith("page_failure"):
            assert not av.get("error") and av.get("content") == "1|"
            assert bv.get("error") and b["public_hook_observation"]["reported_status"] == "error"
        elif a["case"].endswith("locked"):
            assert bv.get("error") and not bv.get("not_found")
        else:
            assert av == bv and b["public_hook_observation"]["reported_status"] == "ok"
        pairs.append({"case":a["case"],"before_status":a["public_hook_observation"]["reported_status"],
            "after_status":b["public_hook_observation"]["reported_status"],
            "controlled_fault":a["controlled_transport_fault_not_neural_input"]})
    directory = ROOT / "local/hermes-file-read-paths-v2"
    names = ("target-baseline", "target-candidate", "full-baseline", "full-candidate",
             "dedup-baseline-targeted", "dedup-candidate-targeted")
    logs = {name:(directory / (name + ".log")).read_text(encoding="utf-8") for name in names}
    for name in names:
        assert "clean env" in logs[name] and "100% complete" in logs[name]
        destination = run / ("canonical-" + name + ".log")
        if destination.exists():
            assert destination.read_text(encoding="utf-8") == logs[name]
        else:
            with destination.open("x",encoding="utf-8") as f:
                f.write(logs[name])
    base, candidate = failed_ids(logs["full-baseline"]), failed_ids(logs["full-candidate"])
    target_failures = {f for f in base if "test_scarx_file_read_paths.py::" in f}
    assert len(target_failures) == 11
    assert "23 tests passed, 0 failed" in logs["target-candidate"]
    # This exact failure predates v2 in v1's canonical baseline. It also passes
    # in both focused runs; do not rerun/drop it until a full suite turns green.
    historical = ROOT / "runs/20261114T052552Z-hermes-file-contract-repair-v1-3aeff8/canonical-baseline-v2.log"
    historical_text = historical.read_text(encoding="utf-8")
    assert KNOWN_DEDUP in failed_ids(historical_text)
    assert not failed_ids(logs["dedup-baseline-targeted"])
    assert not failed_ids(logs["dedup-candidate-targeted"])
    novel = candidate - base
    assert novel <= {KNOWN_DEDUP}
    unresolved_new = novel - {KNOWN_DEDUP}
    assert not unresolved_new
    summaries = {name:next(line for line in logs[name].splitlines() if "=== Summary:" in line) for name in names}
    for file, key in (("file_operations.py","after_core_sha256"),
                      ("file_operations_read_contract.py","after_sibling_sha256")):
        assert hashlib.sha256((run / "source-after" / file).read_bytes()).hexdigest() == manifest[key]
    result = {"run_id":run.name,"verdict":"ENGINEERING_READ_PATHS_V2_QUALIFIED_WITH_EXISTING_OS_TEST_FAILURES",
        "actual_baseline_and_candidate_cases":pairs,"actual_dispatcher_calls_so_far":12,
        "remaining_installed_validation_call_budget":6,"initial_harness_failure_calls":0,
        "canonical_new_contract_tests_passed":23,"new_canonical_failures":sorted(unresolved_new),
        "candidate_existing_failures":sorted(candidate),"original_canonical_summaries":summaries,
        "known_order_sensitive_dedup_failure":{
            "id":KNOWN_DEDUP,"historical_v1_base_already_failed":True,
            "historical_log_sha256":hashlib.sha256(historical.read_bytes()).hexdigest(),
            "v2_focused_base_and_candidate_pass":True,"claimed_fixed":False,
            "scope":"Order/process-sensitive failure in mocked file_ops read-mark test; unrelated reader changes are not a proof of fixing it."},
        "whole_upstream_suite_green":False,"core_before_sha256":manifest["before_core_sha256"],
        "sibling_before_sha256":manifest["before_sibling_sha256"],
        "core_after_sha256":manifest["after_core_sha256"],"sibling_sha256":manifest["after_sibling_sha256"],
        "source_artifacts":"source-after","installed_code_validation_complete":False,
        "neural_learning_or_model_quality_claim":False,
        "scope":"GitBash shell path engineering acceptance. No atomic snapshot guarantee, remote-shell qualification or neural causal awareness."}
    with (run / "engineering-acceptance-v1.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in (
        "actual_baseline_and_candidate_cases","original_canonical_summaries","candidate_existing_failures",
        "known_order_sensitive_dedup_failure")},ensure_ascii=False))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path)
    main(p.parse_args().run)
