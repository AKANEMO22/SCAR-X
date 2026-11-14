"""Independent data-only arithmetic and SQLite checks of captured boundary inputs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def encode_independent(text, numeric):
    terms = re.findall(r"[^\W_]+", text.lower(), re.UNICODE)[:128]
    terms += [" ".join(pair) for pair in zip(terms, terms[1:])]
    counts = Counter()
    for word in terms:
        raw = hashlib.blake2b(word.encode(), digest_size=8).digest()
        counts[int.from_bytes(raw[:4], "little") % 48] += 1 if raw[4] & 1 else -1
    vector = np.zeros(64, dtype=np.float32)
    for coordinate, value in counts.items():
        vector[coordinate] = value
    norm = float(np.sqrt(sum(float(v) ** 2 for v in vector[:48])))
    if norm:
        vector[:48] *= 2 / norm
    vector[48:48+len(numeric)] = np.clip(numeric, -1, 1)
    return np.tile(vector, (8,1))


def main(run):
    run = run.resolve()
    assert run.is_relative_to(ROOT / "runs")
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    manifest, cases, result = map(read, ("manifest.json", "locked-cases.json", "result.json"))
    assert not result["failures"]
    assert hashlib.sha256((run/"locked-cases.json").read_bytes()).hexdigest() == manifest["locked_cases_sha256"]
    for rel, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((run/"source"/rel).read_bytes()).hexdigest() == digest
    relevance_aliases = incident_aliases = 0
    with np.load(run/"captured-input-arrays.npz", allow_pickle=False) as arrays:
        for index, (case, row) in enumerate(zip(cases["relevance"], result["relevance"])):
            for name, evidence in zip(("positive","negative"), case["evidence"]):
                actual = arrays[f"relevance_{index}_{name}"]
                expected = encode_independent(case["query"]+" [event] "+evidence["text"],
                                              [min(len(evidence["text"])/4000,1)])
                assert actual.dtype == np.float32 and actual.shape == (8,64)
                np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
            same = np.array_equal(arrays[f"relevance_{index}_positive"], arrays[f"relevance_{index}_negative"])
            labels = [int(e["explicit_color"] == case["requested_color"]) for e in case["evidence"]]
            assert labels == [1,0] == row["opposing_source_labels"]
            assert same == row["identical_inputs"]
            assert row["input_distinguishability_bound_only"] == (.5 if same else 1)
            relevance_aliases += int(same)
        for index, (case, row) in enumerate(zip(cases["incident"], result["incident"])):
            for name, observation in zip(("missing","denied"), case["observations"]):
                expected = encode_independent(case["query"]+" [observed tool] "+observation["tool"]+" "+observation["error_type"],
                                               [0,1,0,0,1,0,0,0,0,0])
                np.testing.assert_allclose(arrays[f"incident_{index}_{name}"],expected,atol=1e-6,rtol=1e-6)
            same = np.array_equal(arrays[f"incident_{index}_missing"], arrays[f"incident_{index}_denied"])
            assert same == row["reported_error_type_inputs_identical"]
            incident_aliases += int(same)
    database = ROOT/"local"/run.name/"hybrid.sqlite"
    with sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True) as db:
        for case, row in zip(cases["FTS"], result["FTS"]):
            terms = re.findall(r"[^\W_]+", case["query"], re.UNICODE)[:16]
            assert terms == row["first_16_query_words"]
            match = " OR ".join('"'+word+'"' for word in terms)
            texts = [record[0] for record in db.execute("SELECT text FROM evidence_fts WHERE evidence_fts MATCH ? ORDER BY bm25(evidence_fts)",(match,))]
            assert texts == row["actual_hit_texts"]
            assert ("lamp color blue" in texts) == row["blue_source_returned"]
    scope = result["historical_predicate_scope"]
    parent = ROOT/"runs/20261114T035022Z-predicate-retention-dev-v1-032f30/locked-owned-data.json"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == scope["parent_data_sha256"]
    assert result["neural_forward_or_update_calls"] == result["VLM_calls"] == 0
    report = {"run_id": run.name, "verdict": "CAPTURED_INPUT_BOUNDARY_AND_SQLITE_DATA_VERIFIED_ONLY",
              "relevance_opposing_label_identical_pairs": relevance_aliases,
              "incident_error_type_identical_pairs": incident_aliases,
              "FTS_keyword_hits":sum(row["blue_source_returned"] for row in result["FTS"]),
              "independent_hash_feature_arithmetic_and_SQLite_hit_check": True,
              "neural_or_full_assistant_accuracy_measured": False,
              "source_or_weights_modified_or_new_neural_calls":False}
    with (run/"independent-data-audit-v1.json").open("x",encoding="utf-8") as output:
        json.dump(report,output,indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    main(parser.parse_args().run)
