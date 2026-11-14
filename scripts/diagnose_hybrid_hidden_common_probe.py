"""Fixed ALIF representations with common ridge heads and hidden-reset controls."""

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
from scarx.evaluation.common_probe_diagnostic import fit_probe, predict_probe
from scarx.evaluation.predicate_retention_dev import pair_sequence, FAMILIES
from scarx.evaluation.magnitude_protection_p0 import MagnitudeProtectionP0
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.store import Store, encode
from scarx.hybrid.device_budget import gpu_headroom
from scarx.hybrid.resources import ResourceMonitor
from diagnose_hybrid_predicate_fit import immutable_state


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bootstrap(differences):
    rng = np.random.default_rng(26011499)
    values = []
    for _ in range(10000):
        indices = rng.integers(0, 3, 3)
        values.append(np.mean([differences[s, rng.integers(0, 4, 4)].mean() for s in indices]))
    return {"mean_pp": float(differences.mean() * 100),
            "paired_seed_family_CI95_pp": (np.percentile(values, [2.5, 97.5]) * 100).tolist(),
            "scope": "Posthoc diagnostic, three neural seeds/four owned predicates, same finalDEV"}


def main():
    parent = ROOT / "runs/20261114T035022Z-predicate-retention-dev-v1-032f30"
    data_raw = (parent / "locked-owned-data.json").read_bytes()
    original_raw = (parent / "result.json").read_bytes()
    data, original = json.loads(data_raw), json.loads(original_raw)
    assert not original["failures"] and len(original["cells"]) == 12
    train, final = sum(data["train_stages"], []), data["final_DEV"]
    rows = train + final
    targets = np.array([r["target"] for r in train])
    sequences = [pair_sequence(r) for r in rows]
    run = ROOT / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) +
                            "-hidden-common-probe-v1-" + uuid.uuid4().hex[:6])
    run.mkdir()
    sources = list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        Path(__file__), ROOT / "scripts/diagnose_hybrid_predicate_fit.py",
        ROOT / "src/scarx/neural/core.py",
        ROOT / "src/scarx/evaluation/readout_eligibility.py",
        ROOT / "src/scarx/evaluation/predicate_retention_dev.py",
        ROOT / "src/scarx/evaluation/tool_query_oracle.py",
        ROOT / "src/scarx/evaluation/magnitude_protection_p0.py",
        ROOT / "src/scarx/evaluation/common_probe_diagnostic.py",
        ROOT / "experiments/hybrid-hidden-common-probe-diagnostic-v1.md",
    ]
    hashes = {}
    for path in sources:
        rel, raw = path.relative_to(ROOT), path.read_bytes()
        destination = run / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        hashes[rel.as_posix()] = sha(raw)
    checkpoint_hashes = {f"{c['seed']}-{c['arm']}/final-full-state.bin": sha(
        (parent / f"{c['seed']}-{c['arm']}/final-full-state.bin").read_bytes()) for c in original["cells"]}
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "parent_run": parent.name, "parent_data_sha256": sha(data_raw),
        "parent_result_sha256": sha(original_raw), "checkpoint_sha256": checkpoint_hashes,
        "max_seconds": 420, "max_neural_forwards": 12960, "ridge_lambda": 1.0,
        "standardization_train_only": True, "new_hidden_learning": False,
        "original_neural_weight_dtype": "float32", "diagnostic_readout_math_dtype": "float64",
    }), encoding="utf-8")
    report = {"run_id": run.name, "cells": [], "failures": [], "forwards": 0,
              "live_or_original_parent_weights_changed": False,
              "candidate_promoted": False, "original_acquisition_retention_gates_unchanged": True}
    trace = Store(ROOT / "local/hybrid-v1")
    monitor = ResourceMonitor(limit=None)
    monitor.start()
    started = time.monotonic()
    frozen_features = {}

    def check():
        assert time.monotonic() - started < 420, "Seven-minute diagnostic budget exceeded"
        assert report["forwards"] <= 12960, "Forward budget exceeded"
        assert gpu_headroom()["free_MiB"] >= 512, "GPU reserve below512MiB"
        for rel, digest in hashes.items():
            assert sha((ROOT / rel).read_bytes()) == digest, "Frozen source changed: " + rel

    try:
        for cell in original["cells"]:
            seed, arm = cell["seed"], cell["arm"]
            for reset_hidden in ((False, True) if arm in ("current_hidden", "protected_hidden") else (False,)):
                check()
                checkpoint = f"{seed}-{arm}/final-full-state.bin"
                payload = (parent / checkpoint).read_bytes()
                assert sha(payload) == checkpoint_hashes[checkpoint]
                wrapper = MagnitudeProtectionP0.loads(payload) if arm == "protected_hidden" else None
                model = wrapper.base if wrapper else LocalALIF.loads(payload, device="gpu")[0]
                assert model.device == "gpu"
                if reset_hidden:
                    initial = LocalALIF(seed, mode="frozen")
                    np.testing.assert_array_equal(model.host(model.core.indices), initial.core.indices)
                    model.core.w_in[:] = model.xp.asarray(initial.core.w_in)
                    model.core.w_rec[:] = model.xp.asarray(initial.core.w_rec)
                    del initial
                before = immutable_state(model, wrapper)
                features = []
                for i, sequence in enumerate(sequences):
                    if i % 64 == 0:
                        check()
                    model.forward(sequence, head="relevance")
                    features.append(model.host(model.core.features).copy())
                    report["forwards"] += 1
                assert immutable_state(model, wrapper) == before
                features = np.stack(features)
                assert features.shape == (720, 128) and features.dtype == np.float32
                if arm == "frozen":
                    frozen_features[seed] = features.copy()
                elif arm == "readout_only" or reset_hidden:
                    np.testing.assert_array_equal(features, frozen_features[seed])
                xp = model.xp
                probe = fit_probe(xp, features[:len(train)], targets)
                expected = np.linalg.solve(model.host(probe["gram"]), model.host(probe["right"]))
                np.testing.assert_allclose(model.host(probe["weights"]), expected, atol=1e-7, rtol=1e-6)
                scores, predictions = predict_probe(xp, features, probe)
                scores, predictions = model.host(scores), model.host(predictions)
                key = f"{seed}-{arm}" + ("-reset_hidden" if reset_hidden else "")
                scored = {"seed": seed, "arm": arm, "reset_hidden": reset_hidden,
                          "feature_sha256": sha(features.tobytes()), "solver": probe["solver"],
                          "weights_state_unchanged_during_extraction": True,
                          "CPU_closed_form_weights_match": True}
                for split, part, start in (("train", train, 0), ("final_DEV", final, len(train))):
                    predicted = predictions[start:start+len(part)]
                    by_family = {api: float(np.mean([
                        predicted[i] == r["target"] for i,r in enumerate(part) if r["api"] == api]))
                        for api in FAMILIES}
                    scored[split] = {"accuracy": float(np.mean(predicted == [r["target"] for r in part])),
                                     "by_family": by_family}
                np.savez(run / (key + ".npz"), features=features, scores=scores, predictions=predictions,
                         weights=model.host(probe["weights"]), mean=model.host(probe["mean"]),
                         scale=model.host(probe["scale"]), train_targets=targets)
                report["cells"].append(scored)
                trace.event("hidden_common_probe.actual_score", {"run_id": run.name, **scored,
                    "new_hidden_learning": False, "live_neural_counter": False}, "orchestrator")
                print(encode({"run_id": run.name, **scored}), flush=True)
                del model, wrapper, probe
        check()
        assert report["forwards"] == 12960 and len(report["cells"]) == 18
        for arm in ("current_hidden", "protected_hidden"):
            differences = np.array([[next(c for c in report["cells"]
                if c["seed"] == seed and c["arm"] == arm and not c["reset_hidden"])["final_DEV"]["by_family"][api]
                - next(c for c in report["cells"] if c["seed"] == seed and c["arm"] == "frozen")["final_DEV"]["by_family"][api]
                for api in FAMILIES] for seed in (260113,260137,260171)])
            report[arm + "_minus_frozen_common_probe"] = bootstrap(differences)
        report["verdict"] = "DIAGNOSTIC_COMMON_HEAD_ONLY_ORIGINAL_CANDIDATE_STILL_PIVOT"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "COMMON_PROBE_DIAGNOSTIC_INCOMPLETE"
    finally:
        report["seconds"] = time.monotonic() - started
        report["resources"] = monitor.close()
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("hidden_common_probe.completed", {k:v for k,v in report.items() if k != "cells"}, "orchestrator")
        trace.close()
        print(encode({k:v for k,v in report.items() if k not in ("cells", "resources")}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
