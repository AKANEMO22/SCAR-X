"""Numeric local-mechanism checks, no accuracy dataset or deployment."""

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
from scarx.evaluation.magnitude_protection_p0 import MagnitudeProtectionP0, CONSTANTS
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.store import Store, encode
from scarx.hybrid.device_budget import gpu_headroom


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-magnitude-protection-p0-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    hashes = {}
    for path in (
        Path(__file__), ROOT / "src/scarx/evaluation/magnitude_protection_p0.py",
        ROOT / "src/scarx/hybrid/learner.py", ROOT / "src/scarx/hybrid/cuda_alif.py",
        ROOT / "src/scarx/neural/plasticity.py",
        ROOT / "experiments/hybrid-magnitude-protection-prototype-p0-v1.md",
    ):
        raw, rel = path.read_bytes(), path.relative_to(ROOT)
        out = run / "source" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
        hashes[rel.as_posix()] = sha(raw)
    rng = np.random.default_rng(26011477)
    inputs = rng.normal(0, .2, (4, 8, 64)).astype(np.float32)
    np.save(run / "owned-numeric-inputs.npy", inputs, allow_pickle=False)
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "input_bytes_sha256": sha(inputs.tobytes()),
        "seeds": [260113, 260137, 260171], "steps_per_seed": 32,
        "algorithm_constants": CONSTANTS, "stimuli_scope": "Numeric regression, not language/skill dataset",
        "scientific_quality_training": False, "max_seconds": 120,
    }), encoding="utf-8")
    store = Store(ROOT / "local/hybrid-v1")
    report = {"run_id": name, "seeds": [], "failures": [],
              "live_algorithm_or_weights_changed": False,
              "acquisition_retention_or_hidden_advantage_measured": False}
    started = time.monotonic()
    try:
        assert gpu_headroom()["free_MiB"] >= 512
        for seed in (260113, 260137, 260171):
            model = MagnitudeProtectionP0(seed)
            baseline = LocalALIF(seed).to_device("gpu")
            steps = []
            for step in range(32):
                i = step % 4
                evidence = {"kind": "machine_verifier", "id": f"numeric-regression:{seed}:{step}"}
                result = model.learn(inputs[i], i, evidence=evidence)
                if step == 0:
                    baseline.learn(inputs[i], i, evidence=evidence)
                    difference = max(float(model.base.xp.max(model.base.xp.abs(
                        getattr(model.base.core, key) - getattr(baseline.core, key))))
                        for key in ("w_in", "w_rec", "w_out", "bias"))
                    assert difference <= 3e-7
                steps.append({"step": step, "loss": result["loss"],
                    "effective_weight_delta": result["weight_delta"],
                    "hidden_importance_sum": float(model.base.xp.sum(model.base.importance)),
                    "output_importance_sum": float(model.base.xp.sum(model.output_importance)),
                    "bias_importance_sum": float(model.base.xp.sum(model.bias_importance)),
                })
            assert steps[-1]["hidden_importance_sum"] > 0 and steps[-1]["output_importance_sum"] > 0
            raw = model.dumps()
            (run / f"seed-{seed}-prototype.zip").write_bytes(raw)
            restored = MagnitudeProtectionP0.loads(raw)
            assert restored.dumps() == raw
            for value in (model, restored):
                value.learn(inputs[0], 1, reward=-1,
                    evidence={"kind": "machine_verifier", "id": "numeric-negative-sign-check"})
                value.consolidate()
            assert restored.dumps() == model.dumps()
            extra_arrays = model.output_importance.nbytes + model.bias_importance.nbytes
            row = {"seed": seed, "first_step_original_maxdifference": difference,
                   "steps": steps, "cold_GPU_restore_exact": True,
                   "next_negative_and_replay_exact": True, "extra_importance_bytes": extra_arrays,
                   "checkpoint_bytes": len(raw), "learner_device": model.base.device}
            report["seeds"].append(row)
            store.event("magnitude_protection.numeric_seed", {
                "run_id": name, **row, "is_skill_learning_evidence": False,
                "is_live_neural_counter": False}, "orchestrator")
            del model, restored, baseline
        anchors = MagnitudeProtectionP0(260113)
        initial = anchors.base.core.stable_out.copy()
        for step in range(64):
            anchors.learn(inputs[step % 4], step % 4,
                evidence={"kind": "machine_verifier", "id": f"EMA-numeric:{step}"})
            if step == 62:
                assert anchors.base.xp.array_equal(initial, anchors.base.core.stable_out)
        assert not anchors.base.xp.array_equal(initial, anchors.base.core.stable_out)
        report["EMA_first_occurs_at_step64"] = True
        assert time.monotonic() - started < 120
        for rel, digest in hashes.items():
            assert sha((ROOT / rel).read_bytes()) == digest
        report["verdict"] = "NUMERICAL_PROTECTION_PATH_P0_PASS_NOT_QUALITY"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "PROTECTION_P0_FAILED"
    finally:
        report["seconds"] = time.monotonic() - started
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        store.event("magnitude_protection.completed", {
            k:v for k,v in report.items() if k != "seeds"}, "orchestrator")
        store.close()
        print(encode({k:report.get(k) for k in (
            "run_id", "verdict", "EMA_first_occurs_at_step64", "failures", "seconds")}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
