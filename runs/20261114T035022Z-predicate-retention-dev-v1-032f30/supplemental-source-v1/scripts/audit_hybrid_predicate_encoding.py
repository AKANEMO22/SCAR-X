"""Check exact model-input aliasing independently of source-entity splits."""

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


def main(output=None):
    run = ROOT / "runs/20261114T035022Z-predicate-retention-dev-v1-032f30"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for rel in ("src/scarx/evaluation/predicate_retention_dev.py", "src/scarx/hybrid/learner.py"):
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == manifest["source_sha256"][rel]
    data = json.loads((run / "locked-owned-data.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    split_hashes = {}
    for split, rows in (("train", sum(data["train_stages"], [])),
                        ("probe", data["probes"]), ("final_DEV", data["final_DEV"])):
        split_hashes[split] = set()
        for row in rows:
            digest = hashlib.sha256(pair_sequence(row).tobytes()).hexdigest()
            groups[digest].append({"id": row["id"], "split": split, "target": row["target"]})
            split_hashes[split].add(digest)
    overlaps = [
        {"input_sha256": digest, "cases": rows}
        for digest, rows in groups.items() if len({r["split"] for r in rows}) > 1
    ]
    duplicate_final_ids = {
        r["id"] for g in overlaps if any(c["split"] == "train" for c in g["cases"])
        for r in g["cases"] if r["split"] == "final_DEV"
    }
    sensitivity = []
    for cell in result["cells"]:
        original = cell["final_DEV"]["predictions"]
        filtered = [r for r in original if r["id"] not in duplicate_final_ids]
        sensitivity.append({
            "seed": cell["seed"], "arm": cell["arm"],
            "original_cases": len(original), "sensitivity_cases": len(filtered),
            "original_accuracy": cell["final_DEV"]["accuracy"],
            "exclude_input_alias_sensitivity_accuracy": float(np.mean(
                [r["prediction"] == r["target"] for r in filtered])),
            "original_metric_or_gate_changed": False,
        })
    report = {
        "run_id": run.name,
        "source_entity_disjointness_does_not_imply_encoded_disjointness": True,
        "cross_split_exact_input_aliases": overlaps,
        "train_final_alias_case_count": len(duplicate_final_ids),
        "opposite_label_exact_input_aliases": [
            {"input_sha256": digest, "cases": rows} for digest, rows in groups.items()
            if len({r["target"] for r in rows}) > 1],
        "sensitivity_only_no_rescoring_original": sensitivity,
        "training_heldout_labels_directly_used": False,
        "scope": "Lossy signed hash can map distinct unseen names to prior model inputs; keep full original metrics and expose 2/400 affected cases.",
        "decision_affected": False,
    }
    assert len(duplicate_final_ids) == 2
    assert not report["opposite_label_exact_input_aliases"]
    destination = output or run / "encoding-alias-audit-v1.json"
    with destination.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("cross_split_exact_input_aliases", "sensitivity_only_no_rescoring_original")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    main(parser.parse_args().output)
