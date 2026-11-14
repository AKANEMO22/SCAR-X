"""Fixed checkpoints, no training: seen vs unseen fit and clipping observations."""

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
from scarx.evaluation.predicate_retention_dev import pair_sequence, FAMILIES
from scarx.evaluation.magnitude_protection_p0 import MagnitudeProtectionP0
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.store import Store, encode
from scarx.hybrid.device_budget import gpu_headroom


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def immutable_state(model, wrapper):
    names = model.core.weight_names
    return {
        "weight_hashes": {k:sha(model.host(getattr(model.core, k)).tobytes()) for k in names},
        "normalization": [sha(model.host(getattr(model, k)).tobytes()) for k in ("f_mean", "f_var")],
        "importance": sha(model.host(model.importance).tobytes()),
        "extra": [sha(model.host(wrapper.output_importance).tobytes()),
                  sha(model.host(wrapper.bias_importance).tobytes())] if wrapper else [],
        "rng": json.loads(encode(model.rng.bit_generator.state)),
        "signal_rng": json.loads(encode(model.signal_rng.bit_generator.state)),
        "replay": sha(encode(model.replay).encode()), "updates": model.updates,
    }


def main():
    run = ROOT / "runs" / (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-predicate-final-fit-v1-" + uuid.uuid4().hex[:6])
    run.mkdir()
    parent = ROOT / "runs/20261114T035022Z-predicate-retention-dev-v1-032f30"
    data_raw, result_raw = (parent / "locked-owned-data.json").read_bytes(), (parent / "result.json").read_bytes()
    data, parent_result = json.loads(data_raw), json.loads(result_raw)
    assert not parent_result["failures"]
    hashes = {}
    for path in (
        Path(__file__), ROOT / "src/scarx/evaluation/predicate_retention_dev.py",
        ROOT / "src/scarx/evaluation/magnitude_protection_p0.py",
        ROOT / "src/scarx/hybrid/learner.py",
        ROOT / "experiments/hybrid-predicate-final-fit-diagnostic-v1.md",
    ):
        rel, raw = path.relative_to(ROOT), path.read_bytes()
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[rel.as_posix()] = sha(raw)
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "parent_run": parent.name,
        "data_sha256": sha(data_raw), "parent_result_sha256": sha(result_raw),
        "new_learning_or_calibration": False, "max_forwards": 6480, "max_seconds": 300,
    }), encoding="utf-8")
    trace = Store(ROOT / "local/hybrid-v1")
    report = {"run_id": run.name, "cells": [], "failures": [],
              "candidate_promoted": False, "new_learning": False}
    started = time.monotonic()
    try:
        assert gpu_headroom()["free_MiB"] >= 512
        for cell in parent_result["cells"]:
            if cell["arm"] == "frozen":
                continue
            payload = (parent / f"{cell['seed']}-{cell['arm']}/final-full-state.bin").read_bytes()
            wrapper = MagnitudeProtectionP0.loads(payload) if cell["arm"] == "protected_hidden" else None
            model = wrapper.base if wrapper else LocalALIF.loads(payload, device="gpu")[0]
            assert model.updates == 640
            state_before = immutable_state(model, wrapper)
            scored = {"seed": cell["seed"], "arm": cell["arm"],
                      "checkpoint_sha256": sha(payload)}
            for split, rows in (("seen_train", sum(data["train_stages"], [])), ("unseen_DEV", data["final_DEV"])):
                records = []
                for case in rows:
                    probability, features, raw, scale, indices = model.forward(pair_sequence(case), head="relevance")
                    pred = int(np.argmax(probability))
                    records.append({
                        "id": case["id"], "api": case["api"], "target": case["target"],
                        "prediction": pred, "probability_true": float(probability[1]),
                        "raw_feedback_clip_fraction": float(np.mean(np.abs(raw) >= 5)),
                        "active_spikes": model.active_spikes(),
                    })
                if split == "unseen_DEV":
                    assert len(records) == len(cell["final_DEV"]["predictions"])
                    for actual, expected in zip(records, cell["final_DEV"]["predictions"]):
                        assert actual["id"] == expected["id"] and actual["prediction"] == expected["prediction"]
                        assert abs(actual["probability_true"] - expected["probability_true"]) < 1e-6
                scored[split] = {
                    "accuracy": float(np.mean([r["target"] == r["prediction"] for r in records])),
                    "by_family": {api: float(np.mean([
                        r["target"] == r["prediction"] for r in records if r["api"] == api])) for api in FAMILIES},
                    "mean_raw_feedback_clip_fraction": float(np.mean([r["raw_feedback_clip_fraction"] for r in records])),
                    "mean_active_spikes": float(np.mean([r["active_spikes"] for r in records])),
                }
                (run / f"{cell['seed']}-{cell['arm']}-{split}.json").write_text(encode(records), encoding="utf-8")
            assert immutable_state(model, wrapper) == state_before
            scored["weights_statistics_importance_replay_RNG_unchanged"] = True
            report["cells"].append(scored)
            trace.event("predicate_fit.measured", {"run_id": run.name, **scored,
                "is_live_neural_counter": False}, "orchestrator")
            del model, wrapper
            assert time.monotonic() - started < 300
        for rel, digest in hashes.items():
            assert sha((ROOT / rel).read_bytes()) == digest
        report["verdict"] = "FIXED_STATE_FIT_DIAGNOSTIC_ONLY_CANDIDATE_REMAINS_PIVOT"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "FIT_DIAGNOSTIC_INCOMPLETE"
    finally:
        report["seconds"] = time.monotonic() - started
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("predicate_fit.completed", {k:v for k,v in report.items() if k != "cells"}, "orchestrator")
        trace.close()
        print(encode({k:report[k] for k in ("run_id", "verdict", "failures", "seconds")}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
