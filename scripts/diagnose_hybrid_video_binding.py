"""Actual native diagnostic contrasts on stored licensed engineering evidence."""

import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.shared_native import SharedNative
from scarx.evaluation.video_binding_diagnostic import messages, explicit_rank, name_observed
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.store import Store, encode


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-video-binding-diagnostic-v1-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    parent_path = ROOT / "runs/20261114T020701Z-asr-video-integration-p0-a98122/result.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    engineering = json.loads((ROOT / "local/hybrid-v1/samples/engineering-manifest.json").read_text(encoding="utf-8"))
    transcript = engineering["independent_transcript"]
    sentences = [s.strip() + "." for s in transcript.split(".") if s.strip()]
    assert len(sentences) == 2
    questions = [
        {"id": "first", "question": parent["arms"]["base"]["answers"][0]["question"],
         "expected": 15, "entity": "Arly Velasquez"},
        {"id": "second", "question": parent["arms"]["base"]["answers"][1]["question"],
         "expected": 9, "entity": "Adam Hall"},
    ]
    sources = [
        Path(__file__), ROOT / "src/scarx/evaluation/video_binding_diagnostic.py",
        ROOT / "src/scarx/evaluation/shared_native.py", ROOT / "src/scarx/hybrid/learner.py",
        ROOT / "experiments/hybrid-video-binding-diagnostic-v1.md",
    ]
    hashes = {}
    for path in sources:
        raw, rel = path.read_bytes(), path.relative_to(ROOT)
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[rel.as_posix()] = sha(raw)
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "parent_SHA256": sha(parent_path.read_bytes()),
        "questions": questions, "human_reference": transcript,
        "known_case_not_newTEST": True, "training_or_action_execution": False,
    }), encoding="utf-8")
    trace = Store(ROOT / "local/hybrid-v1")
    report = {"run_id": name, "records": [], "failures": [], "unique_native_calls": 0,
              "live_model_or_weights_modified": False, "training": False}
    started = time.monotonic()
    cache = {}
    try:
        native = SharedNative(ROOT)
        for question in questions:
            conditions = []
            for arm in ("base", "small"):
                evidence = [{
                    "text": r["text"], "start": i * 10.0, "end": min(21, (i + 1) * 10.0),
                    "status": "observed",
                } for i,r in enumerate(parent["arms"][arm]["ASR_records"])
                    if r["text"].strip() and r["text"] != "[BLANK_AUDIO]"]
                assert len(evidence) == 2
                conditions.extend([
                    {"source": arm, "selector": "chronological", "evidence": evidence},
                    {"source": arm, "selector": "reverse", "evidence": evidence[::-1]},
                ])
                for seed in (260113, 260137, 260171):
                    model = LocalALIF(seed, mode="frozen").to_device("gpu")
                    ranked = sorted(evidence,
                        key=lambda e: model.relevance(question["question"], e), reverse=True)
                    conditions.append({"source": arm, "selector": "untrained_SNN",
                                       "seed": seed, "evidence": ranked})
                    assert model.updates == 0
                    del model
            human = [{"text": sentence, "start": i * 10.0, "end": min(21, (i+1)*10.0),
                      "status": "human_reference"} for i,sentence in enumerate(sentences)]
            conditions.extend([
                {"source": "ORACLE_human", "selector": "chronological", "evidence": human},
                {"source": "ORACLE_human", "selector": "reverse", "evidence": human[::-1]},
                {"source": "ORACLE_human", "selector": "single_target_sentence",
                 "evidence": [human[0 if question["expected"] == 15 else 1]]},
                {"source": "none", "selector": "no_evidence", "evidence": []},
            ])
            for condition in conditions:
                inputs = messages(question["question"], condition["evidence"])
                key = sha(encode(inputs).encode())
                if key not in cache:
                    output = native.generate({"messages": inputs, "max_tokens": 128})
                    response = output["choices"][0]["message"].get("content") or ""
                    cache[key] = {"native_raw": output, "answer": response, "messages": inputs}
                    report["unique_native_calls"] += 1
                    (run / ("native-" + key + ".json")).write_text(encode(cache[key]), encoding="utf-8")
                answer = cache[key]["answer"]
                row = {**question, **condition, "input_sha256": key,
                       "actual_answer": answer, "diagnostic_rank": explicit_rank(answer),
                       "strict_integer_correct": answer.strip() == str(question["expected"]),
                       "diagnostic_rank_correct": explicit_rank(answer) == question["expected"],
                       "exact_entity_observed": any(name_observed(question["entity"], e["text"])
                           for e in condition["evidence"]),
                       "answer_support_independently_verified": False,
                       "is_oracle_only": condition["source"].startswith("ORACLE")}
                report["records"].append(row)
                trace.event("video_binding.actual_diagnostic_answer",
                    {"run_id": name, **row, "training_label_created": False}, "orchestrator")
        assert report["unique_native_calls"] <= 100 and time.monotonic() - started < 300
        for relative, expected in hashes.items():
            assert sha((ROOT / relative).read_bytes()) == expected
        report["verdict"] = "KNOWN_CASE_DIAGNOSTIC_ONLY_NO_POLICY_PROMOTION"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "DIAGNOSTIC_INCOMPLETE"
    finally:
        report["seconds"] = time.monotonic() - started
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("video_binding.completed", {
            k:v for k,v in report.items() if k != "records"}, "orchestrator")
        trace.close()
        print(encode({k:report[k] for k in (
            "run_id", "verdict", "unique_native_calls", "failures", "seconds")}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
