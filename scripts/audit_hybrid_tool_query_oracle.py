"""Actual private prefix replay, independent SQL read predicates, no model judge."""

from collections import Counter
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
from scarx.evaluation.tooltalk_sandbox import Sandbox, load_executor
from scarx.evaluation.tool_query_oracle import query, multiset_equal, TOOLS, FORMAT
from scarx.hybrid.store import Store, encode


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-tool-query-oracle-p0-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    intake_path = ROOT / "runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json"
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    upstream = Path(intake["local_snapshot"])
    sources = [
        Path(__file__), ROOT / "src/scarx/evaluation/tool_query_oracle.py",
        ROOT / "src/scarx/evaluation/tooltalk_sandbox.py",
        ROOT / "src/scarx/evaluation/tooltalk_schema.py",
        ROOT / "experiments/hybrid-tool-query-oracle-p0-v1.md",
    ]
    hashes = {}
    for path in sources:
        rel, raw = path.relative_to(ROOT), path.read_bytes()
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[rel.as_posix()] = sha(raw)
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "intake_sha256": sha(intake_path.read_bytes()),
        "upstream_revision": intake["revision"], "training_or_inference": False,
        "oracle_scope": "Query predicates on simulator current state, not all mutations or user intent",
    }), encoding="utf-8")
    for file in ("LICENSE", "NOTICE"):
        (run / file).write_bytes((upstream / file).read_bytes())
    trace = Store(ROOT / "local/hybrid-v1")
    report = {"run_id": name, "queries": [], "failures": [],
              "training_or_new_feedback_labels": False, "dataset_qualified_for_learning": False}
    started = time.monotonic()
    try:
        cls = load_executor(upstream, intake)
        sandbox = Sandbox(cls, upstream / "data/databases")
        replay_calls = 0
        for folder in ("easy", "tooltalk"):
            for path in sorted((upstream / "data" / folder).glob("*.json")):
                case = json.loads(path.read_text(encoding="utf-8"))
                sandbox.initialize(case["metadata"], case.get("user", {}))
                call_index = 0
                for turn_index, turn in enumerate(case["conversation"]):
                    if turn["role"].casefold() == "user":
                        sandbox.begin_turn()
                    for gold in turn.get("apis", []):
                        request = copy.deepcopy(gold["request"])
                        request["parameters"].pop("session_token", None)
                        api = request["api_name"]
                        oracle = state = None
                        if api in TOOLS:
                            dbname = TOOLS[api][0]
                            state = copy.deepcopy({
                                "Account": sandbox.executor.databases["Account"],
                                dbname: sandbox.executor.databases.get(dbname, {}),
                            })
                            oracle = query(state, api, request["parameters"],
                                session_token=sandbox.executor.session_token,
                                now=sandbox.executor.now_timestamp.strftime(FORMAT))
                        actual = sandbox.dispatch(request, origin="annotated_replay")
                        replay_calls += 1
                        if api in TOOLS:
                            row = {
                                "path": path.relative_to(upstream).as_posix(),
                                "call_index": call_index, "turn_index": turn_index,
                                "api": api, "args": request["parameters"],
                                "before_state": state,
                                "before_state_sha256": sha(encode(state).encode()),
                                "session_token_source": "Owned public benchmark Account simulator",
                                "session_token": sandbox.executor.session_token,
                                "clock": sandbox.executor.now_timestamp.strftime(FORMAT),
                                "oracle": oracle, "actual": actual,
                                "annotation": gold["response"],
                                "actual_matches_query_contract": actual["exception"] is None and
                                    multiset_equal(oracle, actual["response"]),
                                "annotation_matches_query_contract": gold["exception"] is None and
                                    multiset_equal(oracle, gold["response"]),
                                "user_intent_verified": False,
                            }
                            report["queries"].append(row)
                            trace.event("tool_query_oracle.measured_query", {
                                k:v for k,v in row.items() if k not in (
                                    "before_state", "session_token")},
                                "orchestrator")
                        call_index += 1
        assert replay_calls == 266
        report["replay_calls"] = replay_calls
        report["query_count"] = len(report["queries"])
        report["actual_query_contract_mismatches"] = sum(
            not r["actual_matches_query_contract"] for r in report["queries"])
        report["annotation_query_contract_mismatches"] = sum(
            not r["annotation_matches_query_contract"] for r in report["queries"])
        report["by_API"] = {
            api: {
                "count": sum(r["api"] == api for r in report["queries"]),
                "actual_mismatches": sum(r["api"] == api and not r["actual_matches_query_contract"] for r in report["queries"]),
                "annotation_mismatches": sum(r["api"] == api and not r["annotation_matches_query_contract"] for r in report["queries"]),
            } for api in TOOLS}
        assert time.monotonic() - started < 300
        for rel, digest in hashes.items():
            assert sha((ROOT / rel).read_bytes()) == digest
        report["verdict"] = ("ACTUAL_QUERY_CONTRACTS_VERIFIED_ANNOTATIONS_NOT_UNIVERSAL"
            if report["actual_query_contract_mismatches"] == 0 else "QUERY_CONTRACT_DISAGREEMENT_REQUIRES_REVIEW")
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "QUERY_ORACLE_P0_INCOMPLETE"
    finally:
        report["seconds"] = time.monotonic() - started
        raw = encode(report).encode()
        with (run / "result.json").open("xb") as f:
            f.write(raw)
        trace.event("tool_query_oracle.completed", {
            k:v for k,v in report.items() if k != "queries"}, "orchestrator")
        trace.close()
        print(encode({k:report.get(k) for k in (
            "run_id", "verdict", "query_count", "actual_query_contract_mismatches",
            "annotation_query_contract_mismatches", "failures", "seconds")}))
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
