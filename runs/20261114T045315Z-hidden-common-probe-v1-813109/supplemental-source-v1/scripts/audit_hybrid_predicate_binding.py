"""Data-only check of source truth, stored probabilities and contrast metrics."""

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.predicate_retention_dev import pair_sequence
from audit_hybrid_predicate_retention import label


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def summarize(rows):
    return {"pairs": len(rows),
        "both_correct": sum(r["include_prediction"] == 1 and r["exclude_prediction"] == 0 for r in rows) / len(rows),
        "prediction_flip_rate": sum(r["include_prediction"] != r["exclude_prediction"] for r in rows) / len(rows),
        "mean_probability_contrast": sum(r["include_probability"] - r["exclude_probability"] for r in rows) / len(rows),
        "opposite_label_exact_aliases": sum(r["encoded_equal"] for r in rows)}


def compare(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        assert abs(actual[key] - expected[key]) < 1e-12, (key, actual[key], expected[key])


def main(run, output):
    run = run.resolve()
    assert ROOT / "runs" in run.parents
    manifest = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "result.json").read_text())
    assert not result["failures"] and len(result["cells"]) == 12
    assert not result["new_learning"] and not result["candidate_promoted"]
    for rel, digest in manifest["source_sha256"].items():
        assert sha((run / "source" / rel).read_bytes()) == digest
    parent = ROOT / "runs" / manifest["parent_run"]
    original_raw = (parent / "locked-owned-data.json").read_bytes()
    assert sha(original_raw) == manifest["parent_data_sha256"]
    originals = {r["id"]:r for r in json.loads(original_raw)["final_DEV"]}
    raw = (run / "locked-counterfactual-data.json").read_bytes()
    assert sha(raw) == manifest["contrast_data_sha256"]
    data = json.loads(raw)
    pairs = {r["id"]:r for r in data["pairs"]}
    for pair in pairs.values():
        original = originals[pair["parent_id"]]
        a, b = pair["include"], pair["exclude"]
        assert label(a) == 1 and label(b) == 0
        assert a["record"] == b["record"] == original["record"]
        assert a["clock"] == b["clock"] == original["clock"]
        assert a["api"] == b["api"] == original["api"]
        if pair["contrast"] == "owner":
            assert a["owner"] != b["owner"] and a["args"] == b["args"]
        else:
            assert a["owner"] == b["owner"] and a["args"] != b["args"]
        for case, digest in zip((a,b), pair["input_sha256"]):
            assert sha(pair_sequence(case).tobytes()) == digest
    all_rows, forwards = defaultdict(list), 0
    for cell in result["cells"]:
        checkpoint = f"{cell['seed']}-{cell['arm']}/final-full-state.bin"
        assert sha((parent / checkpoint).read_bytes()) == manifest["checkpoint_sha256"][checkpoint] == cell["checkpoint_sha256"]
        predictions = json.loads((run / f"{cell['seed']}-{cell['arm']}-predictions.json").read_text())
        assert len(predictions["normal"]) == len(pairs)
        assert {r["id"] for r in predictions["normal"]} == set(pairs)
        assert {r["id"] for r in predictions["reversed"]} == set(data["reversed_pair_ids"])
        for direction in ("normal", "reversed"):
            for row in predictions[direction]:
                pair = pairs[row["id"]]
                assert row["api"] == pair["api"] and row["contrast"] == pair["contrast"]
                for side in ("include", "exclude"):
                    p = np.array(row[side + "_distribution"])
                    assert p.shape == (2,) and np.isfinite(p).all() and (p >= 0).all()
                    assert abs(p.sum() - 1) < 1e-6
                    assert row[side + "_probability"] == p[1]
                    assert row[side + "_prediction"] == int(np.argmax(p))
                assert row["encoded_equal"] == (pair["input_sha256"][0] == pair["input_sha256"][1])
            forwards += len(predictions[direction]) * 2
        for group in cell["groups"]:
            select = lambda direction: [r for r in predictions[direction]
                if r["contrast"] == group["contrast"] and r["api"] == group["api"]]
            compare(summarize(select("normal")), {k:v for k,v in group.items()
                if k not in ("contrast", "api", "reversed_first16")})
            compare(summarize(select("reversed")), group["reversed_first16"])
        for row in predictions["normal"]:
            all_rows[cell["arm"], row["contrast"]].append(row)
    assert forwards == result["forwards"] == manifest["expected_forwards"] <= 20000
    assert result["seconds"] < manifest["max_seconds"]
    report = {"run_id": run.name, "source_and_parent_checkpoint_hashes_match": True,
        "dual_verified_pairs_rechecked_with_independent_predicate": len(pairs),
        "stored_distribution_metrics_verified": True, "forwards": forwards,
        "aggregate_by_arm_contrast": [{"arm":a, "contrast":b, **summarize(rows)}
                                     for (a,b),rows in all_rows.items()],
        "in_memory_unchanged_claim": "Primary runner assertion/source; this auditor does not rerun GPU inference.",
        "repeated_parent_sources_and_three_seeds_are_not_independent_quality_replicates": True,
        "verdict": "RECORDED_BINDING_DIAGNOSTIC_VERIFIED_NO_PROMOTION"}
    with (output or run / "independent-contrast-audit-v1.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    main(args.run, args.output)
