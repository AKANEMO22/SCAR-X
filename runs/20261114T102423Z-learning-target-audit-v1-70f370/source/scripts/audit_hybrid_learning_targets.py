"""Data-only learning/action provenance, not another toy accuracy benchmark."""

import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
import uuid
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from scarx.hybrid.store import Store,encode

KINDS=("autonomous_action","machine_verification","local_learning_update",
       "sleep_replay_quantum","review_machine_verification","tool_incident_detected",
       "question_answered","learning_goal_created")


def actions_from_archived_source(path):
    tree=ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ACTIONS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing archived action mapping")


def main():
    start=time.monotonic()
    parent=ROOT/"runs/20261114T011841Z-shared-endurance-v1-554c19"
    manifest=json.loads((parent/"manifest.json").read_text(encoding="utf-8"))
    parent_result=(parent/"result.json").read_bytes()
    run=ROOT/"runs"/(time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"-learning-target-audit-v1-"+uuid.uuid4().hex[:6])
    run.mkdir()
    sources=[Path(__file__),ROOT/"experiments/hybrid-learning-target-attribution-audit-v1.md"]
    hashes={}
    for path in sources:
        rel=path.relative_to(ROOT);dest=run/"source"/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("engine.py","learner.py","cognition.py"):
        rel="src/scarx/hybrid/"+name;raw=(parent/"source"/rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==manifest["source_sha256"][rel]
        dest=run/"parent-source"/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    (run/"manifest.json").write_text(encode({"run_id":run.name,"parent":parent.name,
        "parent_result_sha256":hashlib.sha256(parent_result).hexdigest(),"source_sha256":hashes,
        "new_neural_forward_or_learning":False}),encoding="utf-8")
    actions=actions_from_archived_source(parent/"source/src/scarx/hybrid/learner.py")
    assert actions==("review","search","verify","ask","defer","consolidate")
    database=(ROOT/"local"/parent.name/"hybrid.sqlite").resolve()
    db=sqlite3.connect(database.as_uri()+"?mode=ro",uri=True)
    try:
        db.execute("BEGIN")
        records=[{"id":ident,"created":created,"actor":actor,"kind":kind,"payload":json.loads(raw)}
                 for ident,created,actor,kind,raw in db.execute(
                    "SELECT id,created,actor,kind,payload FROM events WHERE kind IN ("+','.join('?' for _ in KINDS)+") ORDER BY id",KINDS)]
        pointer=json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()[0])
        digest,payload=db.execute("SELECT digest,payload FROM checkpoints WHERE id=?",(pointer["id"],)).fetchone()
    finally:db.close()
    assert len(payload)<=16_000_000 and hashlib.sha256(payload).hexdigest()==digest
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert sum(item.file_size for item in archive.infolist())<=16_000_000
    with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
        meta=json.loads(arrays["metadata"].tobytes())
    evidence_by_id={row["evidence"]["id"]:row for row in meta["replay"]}
    decisions=[r for r in records if r["kind"]=="autonomous_action"]
    proofs={r["payload"]["id"]:r for r in records if r["kind"]=="machine_verification"}
    updates=[r for r in records if r["kind"]=="local_learning_update" and r["payload"].get("accepted") is True]
    rejected=[r for r in records if r["kind"]=="local_learning_update" and not r["payload"].get("accepted")]
    replay_updates={r["payload"]["learning"]["updates"] for r in records if r["kind"]=="sleep_replay_quantum" and r["payload"]["learning"].get("accepted") is True}
    assert len({r["payload"]["updates"] for r in updates})==len(updates)==meta["updates"]
    new_updates=[r for r in updates if r["payload"]["updates"] not in replay_updates]
    attributions=[]
    for row in new_updates:
        item=row["payload"];source=item["evidence"];proof=proofs.get(source["id"])
        candidates=[] if proof is None else [d for d in decisions if d["id"]<proof["id"]
            and d["payload"]["goal_id"]==proof["payload"]["goal_id"]]
        decision=max(candidates,key=lambda d:d["id"]) if candidates else None
        saved=evidence_by_id.get(source["id"])
        if proof and decision and saved:
            target=actions[saved["target"]] if saved["head"]=="action" else str(saved["target"])
            assert proof["id"]<row["id"] and decision["payload"]["action"]==target
            attribution={"update_counter":item["updates"],"source_id":source["id"],
                "proof_event":proof["id"],"decision_event":decision["id"],"learning_event":row["id"],
                "head":saved["head"],"target":saved["target"],"target_meaning":target,
                "reward":saved["reward"],"answer_from_independent_calculator":proof["payload"]["answer"],
                "selector":decision["payload"]["selector"],"proposed_action":decision["payload"]["proposed_action"],
                "executed_action":decision["payload"]["action"],"allowed":decision["payload"]["allowed"],
                "unknown":False}
        else:
            attribution={"update_counter":item["updates"],"source_id":source["id"],"unknown":True}
        attributions.append(attribution)
    (run/"event-ledger.json").write_text(encode(records),encoding="utf-8")
    (run/"new-update-attribution.json").write_text(encode(attributions),encoding="utf-8")
    # Only retained toy replay metadata is published, never the full checkpoint.
    (run/"replay-source-metadata.json").write_text(encode([{k:v for k,v in row.items() if k!="sequence"} for row in meta["replay"]]),encoding="utf-8")
    counts=Counter(r["kind"] for r in records)
    report={"run_id":run.name,"parent":parent.name,"verdict":"EXISTING_LEARNING_TARGETS_ATTRIBUTED_NOT_SKILL_ACQUISITION_PASS",
        "accepted_updates":len(updates),"rejected_updates":len(rejected),"hidden_updates":meta["hidden_updates"],
        "new_experience_updates":len(new_updates),"recorded_replay_updates":len(replay_updates),
        "unique_training_source_ids":len({r["payload"]["evidence"]["id"] for r in updates}),
        "new_target_meanings":dict(Counter(r.get("target_meaning","unknown") for r in attributions)),
        "new_selectors":dict(Counter(r.get("selector","unknown") for r in attributions)),
        "new_single_action_masks":sum(len(r.get("allowed",[]))==1 for r in attributions),
        "new_logged_neural_proposals_with_multiple_allowed":sum(r.get("selector")=="neural_policy" and len(r.get("allowed",[]))>1 for r in attributions),
        "unknown_new_update_attributions":sum(r["unknown"] for r in attributions),
        "reward_sign_counts":dict(Counter("positive" if r["payload"]["reward"]>0 else "negative" if r["payload"]["reward"]<0 else "zero" for r in updates)),
        "event_counts":dict(counts),"action_space":list(actions),"relevance_class_count":2,
        "VLM_weight_training_or_new_neural_calls":0,"active_checkpoint_sha256":digest,
        "unchanged_parent_result_sha256":hashlib.sha256(parent_result).hexdigest(),
        "scope":"Completed owned arithmetic/video engineeringfixture; no measured skill quality or toolerror learning"}
    assert (parent/"result.json").read_bytes()==parent_result and time.monotonic()-start<=30
    with (run/"result.json").open("x",encoding="utf-8") as output:output.write(encode(report))
    live=Store(ROOT/"local/hybrid-v1")
    try:live.event("learning_target_audit.completed",report,"orchestrator")
    finally:live.close()
    print(encode(report))


if __name__=="__main__":main()
