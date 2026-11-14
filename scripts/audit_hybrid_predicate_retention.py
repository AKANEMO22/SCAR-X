"""Data-only rederivation of query labels, predictions, curves and joint gates."""

from collections import Counter
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("FindAlarms", "QueryCalendar", "SearchMessages", "GetReminders")


def label(case):
    if case["record_owner"] != case["owner"]:
        return 0
    row, args = case["record"], case["args"]
    if case["api"] == "GetReminders":
        return 1
    if case["api"] == "FindAlarms":
        return int(args["start_range"] <= row["time"] <= args["end_range"])
    if case["api"] == "QueryCalendar":
        return int(row["start_time"] <= args["end_time"] and row["end_time"] >= args["start_time"])
    matches = [word in row["message"].lower() for word in args["query"].lower().split()]
    return int(row["sender"] == args["sender"] and row["timestamp"] <= case["clock"]
               and (all(matches) if args["match_type"] == "all" else any(matches)))


def raw_meta(path, protected):
    data = path.read_bytes()
    if protected:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            outer = json.loads(z.read("metadata.json"))
            data = z.read("local-alif.npz")
            assert hashlib.sha256(data).hexdigest() == outer["base_sha256"]
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return json.loads(z["metadata"].tobytes())


def main(run, output=None):
    run = run.resolve()
    if ROOT / "runs" not in run.parents:
        raise ValueError("Expected owned run")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    raw = (run / "locked-owned-data.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["data_sha256"]
    data = json.loads(raw)
    for rel, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((run / "source" / rel).read_bytes()).hexdigest() == expected
    all_rows = sum(data["train_stages"], []) + data["probes"] + data["final_DEV"]
    assert len(all_rows) == 920 and len({r["id"] for r in all_rows}) == 920
    assert all(type(r["target"]) is int and r["target"] == label(r) for r in all_rows)
    sets = [{owner for r in rows for owner in r["source_state"]["Account"]} for rows in (
        sum(data["train_stages"], []), data["probes"], data["final_DEV"])]
    assert all(not sets[a] & sets[b] for a, b in ((0,1), (0,2), (1,2)))
    expected_probe = {r["id"]:r for r in data["probes"]}
    expected_final = {r["id"]:r for r in data["final_DEV"]}
    cells = []

    def score(predictions, expected):
        assert len(predictions) == len(expected)
        assert {r["id"] for r in predictions} == set(expected)
        for row in predictions:
            assert row["target"] == expected[row["id"]]["target"]
            assert row["api"] == expected[row["id"]]["api"]
            assert row["prediction"] in (0, 1)
            assert 0 <= row["probability_true"] <= 1
        by_family = {api: sum(
            r["prediction"] == r["target"] for r in predictions if r["api"] == api
        ) / sum(r["api"] == api for r in predictions) for api in FAMILIES}
        accuracy = sum(r["prediction"] == r["target"] for r in predictions) / len(predictions)
        return by_family, accuracy

    for cell in result["cells"]:
        if "final_DEV" not in cell:
            continue
        directory = run / f"{cell['seed']}-{cell['arm']}"
        curves = []
        for stage in range(20):
            original = json.loads((directory / f"stage-{stage:02d}-probes.json").read_text())
            by_family, accuracy = score(original["predictions"], expected_probe)
            assert by_family == original["by_family"] == cell["curves"][stage]["by_family"]
            assert accuracy == original["accuracy"] == cell["curves"][stage]["accuracy"]
            curves.append(by_family)
        final, accuracy = score(cell["final_DEV"]["predictions"], expected_final)
        assert final == cell["final_DEV"]["by_family"] and accuracy == cell["final_DEV"]["accuracy"]
        acquisition = {api:curves[i][api] for i,api in enumerate(FAMILIES)}
        forgetting = {api:100 * max(0, max(c[api] for c in curves[i:]) - curves[-1][api])
                      for i,api in enumerate(FAMILIES)}
        mean, worst = float(np.mean(list(forgetting.values()))), max(forgetting.values())
        assert acquisition == cell["acquisition_by_family"]
        assert forgetting == cell["forgetting"]["by_family_pp"]
        assert mean == cell["forgetting"]["mean_pp"] and worst == cell["forgetting"]["worst_pp"]
        joint = all(v >= .65 for v in acquisition.values()) and all(
            v >= .65 for v in final.values()) and mean <= 3 + 1e-9 and worst <= 5 + 1e-9
        assert joint == cell["joint_acquisition_retention_gate"]
        meta = raw_meta(directory / "final-full-state.bin", cell["arm"] == "protected_hidden")
        expected_updates = 0 if cell["arm"] == "frozen" else 640
        expected_hidden = 640 if cell["arm"] in ("current_hidden", "protected_hidden") else 0
        assert meta["updates"] == expected_updates and meta["hidden_updates"] == expected_hidden
        assert len(meta["replay"]) <= 64
        assert all(r["evidence"]["id"].startswith("SQL+Python:train:") for r in meta["replay"])
        # cell counters are captured AFTER one additional cold-continuation
        # diagnostic update. That update is not part of scored predictions.
        assert cell["updates"] == expected_updates + (cell["arm"] != "frozen")
        assert cell["learn_seconds"] <= manifest["max_LEARN_seconds_per_cell"]
        cells.append({"seed": cell["seed"], "arm": cell["arm"], "final_accuracy": accuracy,
                      "final_by_family": final,
                      "acquisition": acquisition, "forgetting_mean_pp": mean,
                      "forgetting_worst_pp": worst, "joint_gate": joint,
                      "scored_checkpoint_updates": meta["updates"],
                      "extra_cold_diagnostic_updates": cell["updates"] - meta["updates"]})
    if not result["failures"]:
        assert len(cells) == 12
        for seed in manifest["seeds"]:
            replay = [c["replay_evidence_schedule"] for c in result["cells"]
                      if c["seed"] == seed and c["arm"] != "frozen"]
            assert len(replay) == 3 and all(r == replay[0] for r in replay)
    # The runner reported a seed-only CI, although registration requested a
    # seed/family paired bootstrap. Retain the original and add the correct
    # declared clustering here; acquisition/gain gates already fail either way.
    hierarchical = None
    if not result["failures"]:
        differences = np.array([
            [next(c for c in cells if c["seed"] == seed and c["arm"] == "protected_hidden")["final_by_family"][api] -
             next(c for c in cells if c["seed"] == seed and c["arm"] == "current_hidden")["final_by_family"][api]
             for api in FAMILIES] for seed in manifest["seeds"]
        ]) if all("final_by_family" in c for c in cells) else None
        if differences is not None:
            rng = np.random.default_rng(26011499)
            values = []
            for _ in range(10000):
                seed_indices = rng.integers(0, 3, 3)
                values.append(np.mean([
                    np.mean(differences[s, rng.integers(0, 4, 4)]) for s in seed_indices]))
            hierarchical = {"mean_pp": float(differences.mean() * 100),
                "paired_seed_family_CI95_pp": (np.percentile(values, [2.5, 97.5]) * 100).tolist(),
                "runner_CI_scope_deviation": "Original runner used seed-only bootstrap; preserved, superseded for clustering interpretation.",
                "judgement_changed": False, "scope": "3seeds/4ownedcontractfamilies, exploratory only"}
    report = {
        "run_id": run.name, "dual_verifier_labels_rederived": len(all_rows),
        "split_owner_overlap": False, "completed_cells": len(cells),
        "original_failures_retained": result["failures"], "cells": cells,
        "scored_states_exclude_cold_continuation_update": True,
        "registered_paired_seed_family_bootstrap": hierarchical,
        "new_learning_or_serving_edits": False,
        "verdict": "RECORDED_SCREEN_METRICS_VERIFIED" if not result["failures"] else "PARTIAL_SCREEN_ONLY",
        "scope": "Same-orchestrator independent rescore; not an outside peer review or broad intelligence proof",
    }
    destination = output or run / "independent-metric-audit-v1.json"
    with destination.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "cells"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.run, args.output)
