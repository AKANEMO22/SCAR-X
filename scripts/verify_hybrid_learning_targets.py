"""Reconstruct attribution solely from archived raw ledger/replay metadata."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from scarx.hybrid.store import Store,encode


def main(run):
    run=run.resolve();assert run.is_relative_to(ROOT/"runs")
    read=lambda name:json.loads((run/name).read_text(encoding="utf-8"))
    result,ledger,metadata,claimed=map(read,("result.json","event-ledger.json","replay-source-metadata.json","new-update-attribution.json"))
    manifest=read("manifest.json")
    for rel,digest in manifest["source_sha256"].items():
        assert hashlib.sha256((run/"source"/rel).read_bytes()).hexdigest()==digest
    parent=ROOT/"runs"/manifest["parent"]
    assert hashlib.sha256((parent/"result.json").read_bytes()).hexdigest()==manifest["parent_result_sha256"]
    assert all(a["id"]<b["id"] for a,b in zip(ledger,ledger[1:]))
    replay_numbers=set()
    for record in ledger:
        if record["kind"]=="sleep_replay_quantum" and record["payload"]["learning"].get("accepted"):
            replay_numbers.add(record["payload"]["learning"]["updates"])
    updates=[r for r in ledger if r["kind"]=="local_learning_update" and r["payload"].get("accepted")]
    new=[r for r in updates if r["payload"]["updates"] not in replay_numbers]
    assert len(new)==result["new_experience_updates"] and len(replay_numbers)==result["recorded_replay_updates"]
    assert len(updates)==result["accepted_updates"]
    assert {r["payload"]["updates"] for r in updates}==set(range(1,len(updates)+1))
    proofs={r["payload"]["id"]:r for r in ledger if r["kind"]=="machine_verification"}
    goals={r["payload"]["id"]:r["payload"] for r in ledger if r["kind"]=="learning_goal_created"}
    replay={r["evidence"]["id"]:r for r in metadata}
    assert len(replay)==len(metadata)==len(proofs)
    selectors=Counter()
    on_policy=0
    for record,claim in zip(new,claimed):
        item=record["payload"];source=item["evidence"]["id"];proof=proofs[source]
        assert record["id"]==claim["learning_event"] and proof["id"]==claim["proof_event"]
        possibilities=[r for r in ledger if r["kind"]=="autonomous_action"
                       and r["payload"]["goal_id"]==proof["payload"]["goal_id"] and r["id"]<proof["id"]]
        decision=possibilities[-1]
        assert decision["id"]==claim["decision_event"]
        assert replay[source]["head"]=="action" and replay[source]["target"]==result["action_space"].index("verify")
        assert decision["payload"]["action"]=="verify"==claim["target_meaning"]
        assert claim["allowed"]==decision["payload"]["allowed"]
        question=goals[proof["payload"]["goal_id"]]["question"]
        match=re.fullmatch(r"(\d+)\+(\d+)",question)
        assert match and int(proof["payload"]["answer"])==sum(map(int,match.groups()))
        assert item["reward"]==replay[source]["reward"]==1
        selectors[decision["payload"]["selector"]]+=1
        on_policy+=int(decision["payload"]["selector"]=="neural_policy" and len(decision["payload"]["allowed"])>1)
    assert dict(selectors)==result["new_selectors"] and on_policy==result["new_logged_neural_proposals_with_multiple_allowed"]
    assert all(r["payload"]["reward"]==1 for r in updates)
    assert result["reward_sign_counts"]=={"positive":len(updates)}
    assert result["unique_training_source_ids"]==len({r["payload"]["evidence"]["id"] for r in updates})
    assert result["event_counts"].get("tool_incident_detected",0)==0
    assert result["event_counts"].get("review_machine_verification",0)==0
    report={"run_id":run.name,"verdict":"ACTION_TARGET_AND_REPLAY_ATTRIBUTION_DATA_VERIFIED_NOT_CAUSAL_LEARNING_PASS",
            "accepted_updates":len(updates),"new_source_experiences":len(new),"replay_updates":len(replay_numbers),
            "all_new_targets_are_verify_action":True,"new_selectors":dict(selectors),
            "verified_arithmetic_source_values":len(proofs),"new_negative_feedback_examples":0,
            "verified_tool_error_recovery_examples":0,"VLM_weight_updates":0,
            "independent_raw_event_and_replay_joins_match":True,
            "not_frozen_baseline_comparison_or_skill_accuracy":True}
    with (run/"independent-data-audit-v1.json").open("x",encoding="utf-8") as output:output.write(encode(report))
    store=Store(ROOT/"local/hybrid-v1")
    try:store.event("learning_target_audit.data_verified",report,"orchestrator")
    finally:store.close()
    print(encode(report))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path);main(parser.parse_args().run)
