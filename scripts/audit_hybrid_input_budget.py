"""Input-only audit, explicitly no learned forward or weight update."""

import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.learner import LocalALIF, features
from scarx.hybrid.store import Store, encode
from scarx.evaluation.hybrid_incident_encoder import observation_sequence


def words(text):
    return re.findall(r"[^\W_]+", text.lower(), re.UNICODE)


def query(count):
    prefix = "Select the event where the lamp color is blue"
    assert count >= len(words(prefix))
    return prefix + " note" * (count - len(words(prefix)))


def capture_relevance(question, evidence):
    class InputOnlyTap:
        sequence = None
        def forward(self, sequence, head):
            assert head == "relevance"
            self.sequence = np.array(sequence, copy=True)
            return (np.zeros(2, dtype=np.float32),)  # Unreported transport placeholder.
    tap = InputOnlyTap()
    LocalALIF.relevance(tap, question, evidence)
    expected = features(question + " [event] " + evidence["text"],
                        (min(len(evidence["text"]) / 4000, 1.0),))
    assert np.array_equal(tap.sequence, expected)
    return tap.sequence


def cases():
    relevance = [{"query_words": count, "query": query(count),
                  "evidence": [{"text": "lamp color blue", "explicit_color": "blue"},
                               {"text": "lamp color pink", "explicit_color": "pink"}],
                  "requested_color": "blue"} for count in (16,64,120,123,124,125,126,127,128,160)]
    observations = []
    for count in (16,64,120,128,160):
        rows = [{"tool": "read_file", "reported_status": "error", "error_type": error,
                 "empty_result": False, "spillover": False, "streak": 3,
                 "probe": None, "returned_epoch": 100.0}
                for error in ("missing_file", "permission_denied")]
        observations.append({"query_words": count, "query": query(count), "observations": rows,
                             "decision_epoch": 101.0})
    return {"relevance": relevance, "incident": observations,
            "FTS": [{"keyword_word_position": n, "query": "note " * (n-1) + "blue"}
                    for n in (1,15,16,17,32,129)]}


def main():
    begun = time.monotonic()
    run_id = time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()) + "-input-budget-audit-p0-v1-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / run_id
    run.mkdir()
    locked = cases()
    raw = encode(locked).encode()
    (run / "locked-cases.json").write_bytes(raw)
    sources = [Path(__file__), ROOT / "src/scarx/hybrid/learner.py", ROOT / "src/scarx/hybrid/store.py",
               ROOT / "src/scarx/evaluation/hybrid_incident_encoder.py",
               ROOT / "experiments/hybrid-input-budget-audit-p0-v1.md"]
    hashes = {}
    for path in sources:
        rel = path.relative_to(ROOT)
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "manifest.json").write_text(encode({"run_id": run_id, "source_sha256": hashes,
        "locked_cases_sha256": hashlib.sha256(raw).hexdigest(), "model_forward_calls": 0,
        "weight_updates": 0, "measures": "input arrays and actual SQLite retrieval only"}), encoding="utf-8")
    arrays = {}
    report = {"run_id": run_id, "failures": [], "relevance": [], "incident": [], "FTS": [],
              "neural_forward_or_update_calls": 0, "VLM_calls": 0,
              "live_input_or_model_or_source_modified": False,
              "input_tap_is_not_neural_inference": True}
    try:
        for index, case in enumerate(locked["relevance"]):
            x, y = [capture_relevance(case["query"], e) for e in case["evidence"]]
            assert len({len(e["text"]) for e in case["evidence"]}) == 1
            labels = [int(e["explicit_color"] == case["requested_color"]) for e in case["evidence"]]
            same = np.array_equal(x, y)
            arrays[f"relevance_{index}_positive"], arrays[f"relevance_{index}_negative"] = x, y
            report["relevance"].append({"query_words":case["query_words"], "identical_inputs":same,
                "opposing_source_labels": labels, "input_distinguishability_bound_only":.5 if same else 1.,
                "input_sha256":[hashlib.sha256(v.tobytes()).hexdigest() for v in (x,y)],
                "not_a_neural_accuracy":True})
        for index, case in enumerate(locked["incident"]):
            x, y = [observation_sequence(case["query"], [row], decision_epoch=case["decision_epoch"])
                    for row in case["observations"]]
            arrays[f"incident_{index}_missing"], arrays[f"incident_{index}_denied"] = x, y
            report["incident"].append({"query_words":case["query_words"],
                                        "reported_error_type_inputs_identical":np.array_equal(x,y)})
        store = Store(ROOT / "local" / run_id)
        try:
            good = store.observe("lamp color blue", source="owned-input-audit", stream="fixture", status="observed")
            bad = store.observe("lamp color pink", source="owned-input-audit", stream="fixture", status="observed")
            for case in locked["FTS"]:
                actual = store.search(case["query"], stream="fixture", limit=12)
                report["FTS"].append({**case, "first_16_query_words":words(case["query"])[:16],
                    "actual_hit_texts":[row["text"] for row in actual],
                    "blue_source_returned":any(row["id"] == good["id"] for row in actual),
                    "pink_source_returned":any(row["id"] == bad["id"] for row in actual)})
        finally:
            store.close()
        parent = ROOT / "runs/20261114T035022Z-predicate-retention-dev-v1-032f30/locked-owned-data.json"
        data = json.loads(parent.read_text(encoding="utf-8"))
        from scarx.evaluation.predicate_retention_dev import pair_sequence
        # Only named fields actually supplied by pair_sequence, not gold label/ID.
        lengths = []
        all_rows = [row for stage in data["train_stages"] for row in stage] + data["probes"] + data["final_DEV"]
        for case in all_rows:
            qtext = "query " + case["api"] + " owner " + case["owner"]
            etext = "record " + case["api"] + " owner " + case["record_owner"]
            if case["api"] == "SearchMessages":
                qtext += " sender " + case["args"]["sender"] + " keywords " + case["args"]["query"]
                etext += " sender " + case["record"]["sender"] + " body " + case["record"]["message"]
            assert pair_sequence(case).shape == (16,64)
            lengths.extend((len(words(qtext)),len(words(etext))))
        report["historical_predicate_scope"] = {"parent_data_sha256":hashlib.sha256(parent.read_bytes()).hexdigest(),
            "text_fields_checked":len(lengths),"max_words":max(lengths),"over_128":sum(n>128 for n in lengths),
            "long_input_audit_does_not_explain_short_predicate_failure":True}
        assert time.monotonic() - begun <= 30
        assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest for rel,digest in hashes.items())
        report["verdict"] = "INPUT_BOUNDARY_DATA_MEASURED_NOT_MODEL_QUALITY_OR_FIX"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "INPUT_BOUNDARY_AUDIT_INCOMPLETE"
    np.savez(run / "captured-input-arrays.npz", **arrays)
    report["seconds"] = time.monotonic() - begun
    with (run / "result.json").open("x",encoding="utf-8") as output:
        output.write(encode(report))
    store = Store(ROOT / "local/hybrid-v1")
    try:
        store.event("input_budget_audit.completed",report,"orchestrator")
    finally:
        store.close()
    print(encode({k:report[k] for k in ("run_id","verdict","failures","seconds")}))


if __name__ == "__main__":
    main()
