"""Read-only raw response/source analysis; no model judge or implicit new labels."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.tool_response_qualification import classify
from scarx.hybrid.store import Store, encode


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-tool-response-qualification-p0-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    parent = ROOT / "runs/20261114T002144Z-recovery-executor-p0-b29ed8"
    intake_path = ROOT / "runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json"
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    upstream = Path(intake["local_snapshot"])
    raw_result = (parent / "result.json").read_bytes()
    original = json.loads(raw_result)
    hashes = {}
    for path in (Path(__file__), ROOT / "src/scarx/evaluation/tool_response_qualification.py",
                 ROOT / "experiments/hybrid-tool-response-qualification-p0-v1.md"):
        raw, rel = path.read_bytes(), path.relative_to(ROOT)
        destination = run / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        hashes[rel.as_posix()] = digest(raw)
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "parent_run": parent.name,
        "parent_result_sha256": digest(raw_result), "upstream_revision": intake["revision"],
        "intake_sha256": digest(intake_path.read_bytes()),
        "training_or_model_inference": False,
    }), encoding="utf-8")
    for filename in ("LICENSE", "NOTICE"):
        (run / filename).write_bytes((upstream / filename).read_bytes())
    trace = Store(ROOT / "local/hybrid-v1")
    report = {"run_id": name, "rows": [], "conversations": [], "failures": [],
              "dataset_qualified_for_learning": False, "new_labels_created": False}
    started = time.monotonic()
    try:
        assert not original["failures"]
        ids, user_text_groups = defaultdict(list), defaultdict(list)
        for record in original["conversations"]:
            path = upstream / record["path"]
            relative = path.relative_to(upstream).as_posix()
            raw = path.read_bytes()
            assert digest(raw) == intake["file_sha256"][relative]
            authored = json.loads(raw)
            ids[authored["conversation_id"]].append(relative)
            user_text = "\n".join(t["text"].strip().casefold()
                for t in authored["conversation"] if t["role"].casefold() == "user")
            user_text_groups[re.sub(r"\s+", " ", user_text)].append(relative)
            calls = []
            previous_user = None
            for index, turn in enumerate(authored["conversation"]):
                if turn["role"].casefold() == "user":
                    previous_user = turn["text"]
                for call in turn.get("apis", []):
                    calls.append((index, previous_user, call))
            assert len(calls) == len(record["calls"])
            divergent = []
            for index, ((turn, question, expected), observation) in enumerate(zip(calls, record["calls"])):
                assert expected == observation["expected"], "Parent raw annotation changed"
                actual = observation["actual"]
                assert actual["exception"] == expected["exception"], "Exception mismatch"
                result = classify(expected["request"]["api_name"], expected["response"], actual["response"])
                row = {"path": relative, "call_index": index, "turn_index": turn,
                       "preceding_user_text": question, "api": expected["request"]["api_name"],
                       "classification": result, "is_user_goal_verified": False,
                       "actual": actual, "expected": expected,
                       "label_authority": "Unqualified annotations; not a new feedback label"}
                report["rows"].append(row)
                if result["category"] != "exact":
                    divergent.append(index)
                    trace.event("tool_response_qualification.measured_difference",
                        {"run_id": name, **row}, "orchestrator")
            report["conversations"].append({
                "path": relative, "conversation_id": authored["conversation_id"],
                "calls": len(calls), "divergent_call_indices": divergent,
                "raw_trace_quarantine": bool(divergent),
                "intent_annotation_verified": False})
        report["classification_counts"] = dict(Counter(
            r["classification"]["category"] for r in report["rows"]))
        report["duplicate_conversation_ids"] = [v for v in ids.values() if len(v) > 1]
        report["identical_normalized_user_dialogue_groups"] = [
            v for v in user_text_groups.values() if len(v) > 1]
        assert len(report["rows"]) == 266 and len(report["conversations"]) == 78
        assert sum(n for c, n in report["classification_counts"].items() if c != "exact") == 37
        assert not report["duplicate_conversation_ids"]
        for rel, expected in hashes.items():
            assert digest((ROOT / rel).read_bytes()) == expected
        report["verdict"] = "DESCRIPTIVE_DIFFERENCES_CLASSIFIED_NOT_LEARNING_LABELS"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "QUALIFICATION_P0_FAILED"
    finally:
        report["seconds"] = time.monotonic() - started
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("tool_response_qualification.completed", {
            k:v for k,v in report.items() if k not in ("rows", "conversations")}, "orchestrator")
        trace.close()
        print(encode({k:report.get(k) for k in ("run_id", "verdict", "classification_counts", "failures", "seconds")}))
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
