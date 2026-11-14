"""Registered fresh predicate screen; real ALIF learning, no live weight edits."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.predicate_retention_dev import dataset, pair_sequence, forgetting, FAMILIES
from scarx.evaluation.magnitude_protection_p0 import MagnitudeProtectionP0
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.resources import ResourceMonitor, windows_tree_commit_limit
from scarx.hybrid.device_budget import gpu_headroom
from scarx.hybrid.store import Store, encode

ARMS = ("frozen", "readout_only", "current_hidden", "protected_hidden")
SEEDS = (260113, 260137, 260171)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-predicate-retention-dev-v1-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    data = dataset()
    data_bytes = encode(data).encode()
    (run / "locked-owned-data.json").write_bytes(data_bytes)
    sources = list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        Path(__file__), ROOT / "src/scarx/neural/core.py",
        ROOT / "src/scarx/evaluation/readout_eligibility.py",
        ROOT / "src/scarx/evaluation/predicate_retention_dev.py",
        ROOT / "src/scarx/evaluation/tool_query_oracle.py",
        ROOT / "src/scarx/evaluation/magnitude_protection_p0.py",
        ROOT / "experiments/hybrid-predicate-retention-dev-v1.md",
    ]
    hashes = {}
    for path in sources:
        rel, raw = path.relative_to(ROOT), path.read_bytes()
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[rel.as_posix()] = sha(raw)
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "data_sha256": sha(data_bytes),
        "seeds": SEEDS, "arms": ARMS, "stage_count": 20,
        "new_experiences_per_stage": 16, "replay_attempts_per_stage": 16,
        "neural_device": "gpu", "max_seconds": 1800,
        "max_LEARN_seconds_per_cell": 120, "data_is_final_confirmationTEST": False,
        "P0_aggregate_budget_failure_retained": True,
        "live_model_or_weights_touched": False,
    }), encoding="utf-8")
    sequences = {r["id"]:pair_sequence(r) for rows in data["train_stages"] for r in rows}
    sequences.update({r["id"]:pair_sequence(r) for r in data["probes"] + data["final_DEV"]})
    trace = Store(ROOT / "local/hybrid-v1")
    monitor = ResourceMonitor(limit=None)
    report = {"run_id": name, "cells": [], "failures": [],
              "scientific_hybrid_P2_P3_or_production_pass": False,
              "SQL_contract_control_accuracy": 1.0,
              "live_model_or_weights_changed": False}
    started = time.monotonic()

    def check():
        if time.monotonic() - started > 1800:
            raise TimeoutError("Registered30minute study budget exceeded")
        if gpu_headroom()["free_MiB"] < 512:
            raise MemoryError("Physical GPU reserve below512MiB")
        for rel, expected in hashes.items():
            if sha((ROOT / rel).read_bytes()) != expected:
                raise RuntimeError("Frozen study source changed: " + rel)

    def evaluate(model, rows):
        predictions = []
        for row in rows:
            probabilities = model.forward(sequences[row["id"]], head="relevance")[0]
            prediction = int(np.argmax(probabilities))
            predictions.append({"id": row["id"], "api": row["api"], "target": row["target"],
                                "prediction": prediction, "probability_true": float(probabilities[1])})
        by_family = {family: float(np.mean([
            r["prediction"] == r["target"] for r in predictions if r["api"] == family]))
            for family in FAMILIES}
        return {"by_family": by_family, "accuracy": float(np.mean([
            r["prediction"] == r["target"] for r in predictions])), "predictions": predictions}

    try:
        report["job"] = windows_tree_commit_limit(None)
        monitor.start()
        for seed in SEEDS:
            for arm in ARMS:
                check()
                protected = arm == "protected_hidden"
                method = MagnitudeProtectionP0(seed, device="gpu") if protected else LocalALIF(
                    seed, mode=arm if arm in ("frozen", "readout_only") else "learned").to_device("gpu")
                model = method.base if protected else method
                mode_dir = run / f"{seed}-{arm}"
                mode_dir.mkdir()
                cell = {"seed": seed, "arm": arm, "curves": [], "learn_seconds": 0.0,
                        "replay_evidence_schedule": [], "stage_logs": []}
                report["cells"].append(cell)
                cell_start = time.monotonic()
                trace.event("predicate_retention.cell_started", {
                    "run_id": name, "seed": seed, "arm": arm, "data_source": "owned dual-verifier query contract",
                    "live_neural_counter": False}, "orchestrator")
                for stage, rows in enumerate(data["train_stages"]):
                    check()
                    logs = []
                    for row in rows:
                        tick = time.monotonic()
                        result = method.learn(sequences[row["id"]], row["target"], head="relevance",
                            evidence={"kind": "machine_verifier", "id": "SQL+Python:" + row["id"]})
                        replay = method.consolidate() if arm != "frozen" else method.learn(
                            sequences[row["id"]], row["target"], head="relevance", remember=False,
                            evidence={"kind": "machine_verifier", "id": "SQL+Python:" + row["id"]})
                        cell["learn_seconds"] += time.monotonic() - tick
                        if cell["learn_seconds"] > 120:
                            raise TimeoutError(f"Matched LEARN cap exceeded: {seed}/{arm}")
                        cell["replay_evidence_schedule"].append(replay["evidence"]["id"] if replay else None)
                        logs.append({"id": row["id"], "target": row["target"], "loss": result["loss"],
                                     "effective_weight_delta": result.get("weight_delta"),
                                     "replay_loss": replay["loss"] if replay else None})
                    curve = evaluate(model, data["probes"])
                    (mode_dir / f"stage-{stage:02d}-probes.json").write_text(encode(curve), encoding="utf-8")
                    summary = {"stage": stage, "by_family": curve["by_family"], "accuracy": curve["accuracy"]}
                    cell["curves"].append(summary)
                    cell["stage_logs"].append({"stage": stage, "samples": logs})
                    trace.event("predicate_retention.actual_learning_stage", {
                        "run_id": name, "seed": seed, "arm": arm, **summary,
                        "mean_train_loss": float(np.mean([r["loss"] for r in logs])),
                        "updates": model.updates, "hidden_updates": model.hidden_updates,
                        "learn_seconds": cell["learn_seconds"], "local_rule": "magnitude_prototype" if protected else "current",
                        "is_live_neural_counter": False}, "orchestrator")
                    print(encode({"run_id": name, "seed": seed, "arm": arm, "stage": stage,
                                  "probe_accuracy": curve["accuracy"]}), flush=True)
                final = evaluate(model, data["final_DEV"])
                cell["final_DEV"] = final
                cell["forgetting"] = forgetting(cell["curves"])
                cell["acquisition_by_family"] = {family: cell["curves"][i]["by_family"][family]
                                                for i,family in enumerate(FAMILIES)}
                cell["joint_acquisition_retention_gate"] = bool(
                    all(v >= .65 for v in cell["acquisition_by_family"].values())
                    and all(v >= .65 for v in final["by_family"].values())
                    and cell["forgetting"]["within_registered_limits"])
                payload = method.dumps()
                (mode_dir / "final-full-state.bin").write_bytes(payload)
                restored = MagnitudeProtectionP0.loads(payload) if protected else LocalALIF.loads(
                    payload, device="gpu")[0]
                assert restored.dumps() == payload
                row = data["train_stages"][-1][-1]
                for instance in (method, restored):
                    instance.learn(sequences[row["id"]], row["target"], head="relevance",
                                   evidence={"kind": "machine_verifier", "id": "cold-next-numeric"})
                assert restored.dumps() == method.dumps()
                cell["cold_GPU_next_update_exact"] = True
                cell["updates"], cell["hidden_updates"] = model.updates, model.hidden_updates
                cell["seconds"] = time.monotonic() - cell_start
                (mode_dir / "cell.json").write_text(encode(cell), encoding="utf-8")
                trace.event("predicate_retention.cell_completed", {
                    k:v for k,v in {"run_id": name, **cell}.items()
                    if k not in ("final_DEV", "replay_evidence_schedule", "stage_logs", "curves")}, "orchestrator")
                del restored, method, model
        for seed in SEEDS:
            schedules = [c["replay_evidence_schedule"] for c in report["cells"]
                         if c["seed"] == seed and c["arm"] != "frozen"]
            assert all(s == schedules[0] for s in schedules)
        report["paired_replay_schedules_exact"] = True
        gains = [p["final_DEV"]["accuracy"] - c["final_DEV"]["accuracy"]
                 for seed in SEEDS
                 for p in report["cells"] if p["seed"] == seed and p["arm"] == "protected_hidden"
                 for c in report["cells"] if c["seed"] == seed and c["arm"] == "current_hidden"]
        bootstrap = np.random.default_rng(26011499).choice(gains, (10000, len(gains)), replace=True).mean(axis=1)
        report["protected_minus_current"] = {
            "seed_differences": gains, "mean_pp": float(np.mean(gains) * 100),
            "exploratory_seed_CI95_pp": (np.percentile(bootstrap, [2.5, 97.5]) * 100).tolist(),
            "scope": "Three neural seeds on one owned corpus; no broad significance claim",
        }
        protected_cells = [c for c in report["cells"] if c["arm"] == "protected_hidden"]
        report["candidate_joint_gate_pass"] = all(c["joint_acquisition_retention_gate"] for c in protected_cells)
        report["verdict"] = ("SCREENING_SIGNAL_REQUIRES_MATCHED_MAGNITUDE_AND_HIDDEN_ABLATION"
            if report["candidate_joint_gate_pass"] else "PIVOT_CANDIDATE_ACQUISITION_OR_RETENTION_FAIL")
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "STUDY_INCOMPLETE_NO_GO_CLAIM"
    finally:
        report["seconds"] = time.monotonic() - started
        report["resources"] = monitor.close()
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("predicate_retention.study_completed", {
            k:v for k,v in report.items() if k != "cells"}, "orchestrator")
        trace.close()
        print(encode({k:report.get(k) for k in (
            "run_id", "verdict", "candidate_joint_gate_pass",
            "protected_minus_current", "failures", "seconds")}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
