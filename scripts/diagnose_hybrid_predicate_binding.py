"""Registered fixed-checkpoint query contrasts; no learning/live model changes."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.predicate_binding_diagnostic import build_pairs, metrics
from scarx.evaluation.predicate_retention_dev import pair_sequence
from scarx.evaluation.magnitude_protection_p0 import MagnitudeProtectionP0
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.store import Store, encode
from scarx.hybrid.resources import ResourceMonitor
from scarx.hybrid.device_budget import gpu_headroom
from diagnose_hybrid_predicate_fit import immutable_state


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parent = ROOT / "runs/20261114T035022Z-predicate-retention-dev-v1-032f30"
    parent_raw = (parent / "result.json").read_bytes()
    parent_result = json.loads(parent_raw)
    assert not parent_result["failures"] and len(parent_result["cells"]) == 12
    data_raw = (parent / "locked-owned-data.json").read_bytes()
    pairs, exclusions = build_pairs(json.loads(data_raw)["final_DEV"])
    group_counts = Counter()
    sequences, reversed_ids = {}, set()
    for pair in pairs:
        a, b = pair_sequence(pair["include"]), pair_sequence(pair["exclude"])
        sequences[pair["id"]] = a, b
        pair["input_sha256"] = [sha(a.tobytes()), sha(b.tobytes())]
        group = pair["contrast"], pair["api"]
        group_counts[group] += 1
        if group_counts[group] <= 16:
            reversed_ids.add(pair["id"])
    expected_forwards = 12 * 2 * (len(pairs) + len(reversed_ids))
    assert expected_forwards <= 20000
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-predicate-binding-diagnostic-v1-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    owned_raw = encode({"pairs": pairs, "pre_inference_exclusions": exclusions,
                        "reversed_pair_ids": sorted(reversed_ids)}).encode()
    (run / "locked-counterfactual-data.json").write_bytes(owned_raw)
    sources = list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        Path(__file__), ROOT / "scripts/diagnose_hybrid_predicate_fit.py",
        ROOT / "src/scarx/neural/core.py",
        ROOT / "src/scarx/evaluation/readout_eligibility.py",
        ROOT / "src/scarx/evaluation/predicate_binding_diagnostic.py",
        ROOT / "src/scarx/evaluation/predicate_retention_dev.py",
        ROOT / "src/scarx/evaluation/tool_query_oracle.py",
        ROOT / "src/scarx/evaluation/magnitude_protection_p0.py",
        ROOT / "experiments/hybrid-predicate-binding-diagnostic-v1.md",
    ]
    hashes = {}
    for path in sources:
        rel, raw = path.relative_to(ROOT), path.read_bytes()
        target = run / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[rel.as_posix()] = sha(raw)
    checkpoint_hashes = {
        f"{c['seed']}-{c['arm']}/final-full-state.bin": sha(
            (parent / f"{c['seed']}-{c['arm']}/final-full-state.bin").read_bytes())
        for c in parent_result["cells"]}
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "parent_run": parent.name,
        "parent_data_sha256": sha(data_raw), "parent_result_sha256": sha(parent_raw),
        "checkpoint_sha256": checkpoint_hashes, "contrast_data_sha256": sha(owned_raw),
        "expected_forwards": expected_forwards, "max_seconds": 420, "max_forwards": 20000,
        "new_learning": False, "new_unseen_confirmation_data": False,
        "neural_device": "gpu", "neural_weights_dtype": "float32",
    }), encoding="utf-8")
    report = {"run_id": name, "cells": [], "failures": [], "new_learning": False,
              "candidate_promoted": False, "quality_or_production_PASS": False,
              "pairs": len(pairs), "reversed_pairs": len(reversed_ids), "forwards": 0,
              "contrast_groups": [{"contrast": a, "api": b, "pairs": n}
                                  for (a,b),n in group_counts.items()],
              "pre_inference_exclusions": len(exclusions)}
    trace = Store(ROOT / "local/hybrid-v1")
    monitor = ResourceMonitor(limit=None)
    monitor.start()
    started = time.monotonic()

    def check():
        assert time.monotonic() - started < 420, "Seven-minute diagnostic budget exceeded"
        assert report["forwards"] <= 20000, "Diagnostic forward budget exceeded"
        assert gpu_headroom()["free_MiB"] >= 512, "Physical GPU reserve below512MiB"
        for rel, digest in hashes.items():
            assert sha((ROOT / rel).read_bytes()) == digest, "Frozen source changed: " + rel

    def score(model, a, b):
        pa = model.forward(a, head="relevance")[0]
        pb = model.forward(b, head="relevance")[0]
        report["forwards"] += 2
        return {"include_probability": float(pa[1]), "exclude_probability": float(pb[1]),
                "include_distribution": pa.tolist(), "exclude_distribution": pb.tolist(),
                "include_prediction": int(np.argmax(pa)), "exclude_prediction": int(np.argmax(pb)),
                "encoded_equal": bool(np.array_equal(a, b))}

    try:
        for original_cell in parent_result["cells"]:
            check()
            seed, arm = original_cell["seed"], original_cell["arm"]
            checkpoint = f"{seed}-{arm}/final-full-state.bin"
            payload = (parent / checkpoint).read_bytes()
            assert sha(payload) == checkpoint_hashes[checkpoint]
            protected = arm == "protected_hidden"
            wrapper = MagnitudeProtectionP0.loads(payload) if protected else None
            model = wrapper.base if wrapper else LocalALIF.loads(payload, device="gpu")[0]
            assert model.device == "gpu" and model.updates == (0 if arm == "frozen" else 640)
            state_before = immutable_state(model, wrapper)
            records, reverse_records = [], []
            for i, pair in enumerate(pairs):
                if i % 64 == 0:
                    check()
                a, b = sequences[pair["id"]]
                record = {"id": pair["id"], "parent_id": pair["parent_id"],
                          "contrast": pair["contrast"], "api": pair["api"], **score(model, a, b)}
                records.append(record)
                if pair["id"] in reversed_ids:
                    reversed_score = score(model, np.concatenate((a[8:], a[:8])), np.concatenate((b[8:], b[:8])))
                    reverse_records.append({**record, **reversed_score,
                        "original_include_probability": record["include_probability"],
                        "original_exclude_probability": record["exclude_probability"]})
            assert immutable_state(model, wrapper) == state_before
            cell = {"seed": seed, "arm": arm, "checkpoint_sha256": sha(payload),
                    "learning_state_unchanged": True, "groups": []}
            for kind, api in group_counts:
                rows = [r for r in records if r["contrast"] == kind and r["api"] == api]
                reverse = [r for r in reverse_records if r["contrast"] == kind and r["api"] == api]
                group = {"contrast": kind, "api": api, **metrics(rows),
                         "reversed_first16": metrics(reverse)}
                cell["groups"].append(group)
                trace.event("predicate_binding.actual_contrast", {
                    "run_id": name, "seed": seed, "arm": arm, **group,
                    "new_learning": False, "is_live_neural_counter": False}, "orchestrator")
            (run / f"{seed}-{arm}-predictions.json").write_text(encode({
                "normal": records, "reversed": reverse_records}), encoding="utf-8")
            report["cells"].append(cell)
            print(encode({"run_id": name, "seed": seed, "arm": arm,
                          "forwards": report["forwards"], "seconds": time.monotonic()-started}), flush=True)
            del model, wrapper
        check()
        assert report["forwards"] == expected_forwards
        report["verdict"] = "FIXED_CHECKPOINT_BINDING_DIAGNOSIS_ONLY_NO_PROMOTION"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "BINDING_DIAGNOSTIC_INCOMPLETE"
    finally:
        report["seconds"] = time.monotonic() - started
        report["resources"] = monitor.close()
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("predicate_binding.completed", {k:v for k,v in report.items() if k != "cells"}, "orchestrator")
        trace.close()
        print(encode({k:report[k] for k in ("run_id", "verdict", "pairs", "forwards", "failures", "seconds")}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
