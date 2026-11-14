"""Actual local CUDA counterfactuals; no algorithm, serving or live-weight edit."""

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
from scarx.hybrid.learner import LocalALIF, features
from scarx.hybrid.store import Store, encode
from scarx.hybrid.device_budget import gpu_headroom

SEEDS = (260113, 260137, 260171)


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-retention-mechanism-p0-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    source_paths = [
        Path(__file__), ROOT / "src/scarx/hybrid/learner.py",
        ROOT / "src/scarx/hybrid/cuda_alif.py", ROOT / "src/scarx/neural/core.py",
        ROOT / "experiments/hybrid-retention-mechanism-audit-p0-v1.md",
    ]
    hashes = {}
    for path in source_paths:
        raw, rel = path.read_bytes(), path.relative_to(ROOT)
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[rel.as_posix()] = hashlib.sha256(raw).hexdigest()
    sequences = [features("Owned observation " + text, (i / 4, i % 2, i == 2))
                 for i, text in enumerate((
                     "file missing source requested",
                     "tool schema time format error",
                     "no authorization cannot execute",
                     "result valid query complete"))]
    for i, sequence in enumerate(sequences):
        np.save(run / f"input-{i}.npy", sequence, allow_pickle=False)
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "seeds": SEEDS, "steps_per_seed": 32,
        "input_sha256": [hashlib.sha256(s.tobytes()).hexdigest() for s in sequences],
        "scope": "Stored-state causal audit, not skill/generalization or a new learner",
    }), encoding="utf-8")
    trace = Store(ROOT / "local/hybrid-v1")
    trace.event("retention_mechanism.started", {"run_id": name, "updates_are_private": True,
        "neural_device": "gpu", "production_learner_touched": False}, "orchestrator")
    report = {"run_id": name, "seeds": [], "failures": [],
              "live_weights_or_algorithms_modified": False}
    started = time.monotonic()
    try:
        assert gpu_headroom()["free_MiB"] >= 512
        for seed in SEEDS:
            a = LocalALIF(seed).to_device("gpu")
            b, _ = LocalALIF.loads(a.dumps(), device="gpu")
            b.importance.fill(1000)
            for suffix in ("in", "rec", "out", "bias"):
                getattr(b.core, "stable_" + suffix)[:] += .25
            stable_before = {n: a.host(getattr(a.core, n)).copy()
                             for n in ("stable_in", "stable_rec", "stable_out", "stable_bias")}
            initial_mean, initial_var = a.f_mean.copy(), a.f_var.copy()
            initial_weights = {n: a.host(getattr(a.core, n)).copy()
                               for n in ("w_in", "w_rec", "w_out", "bias")}
            record = {"seed": seed, "steps": []}
            for step in range(32):
                i = step % len(sequences)
                evidence = {"kind": "machine_verifier", "id": f"owned-counterfactual:{seed}:{step}"}
                ra = a.learn(sequences[i], i, evidence=evidence)
                rb = b.learn(sequences[i], i, evidence=evidence)
                differences = {n: float(a.xp.max(a.xp.abs(
                    getattr(a.core, n) - getattr(b.core, n))))
                    for n in initial_weights}
                assert not any(differences.values())
                record["steps"].append({"step": step, "loss_default": ra["loss"],
                    "loss_importance1000_stable_shift": rb["loss"],
                    "trainable_weight_maxdifference": differences})
            record["default_importance_nonzero"] = int(a.xp.count_nonzero(a.importance))
            record["default_stable_arrays_unchanged"] = all(np.array_equal(
                a.host(getattr(a.core, n)), raw) for n, raw in stable_before.items())
            record["synaptic_weight_delta"] = {n: float(np.linalg.norm(
                a.host(getattr(a.core, n)) - raw)) for n, raw in initial_weights.items()}
            trained, _ = LocalALIF.loads(a.dumps(), device="gpu")
            original_stats, _ = LocalALIF.loads(a.dumps(), device="gpu")
            original_stats.f_mean[:] = initial_mean
            original_stats.f_var[:] = initial_var
            outputs = []
            for i, sequence in enumerate(sequences):
                p = trained.forward(sequence)[0]
                q = original_stats.forward(sequence)[0]
                outputs.append({"input": i, "with_online_statistics": p.tolist(),
                    "with_initial_statistics_same_weights": q.tolist(),
                    "max_probability_delta": float(np.max(np.abs(p - q))),
                    "rank_changed": int(np.argmax(p)) != int(np.argmax(q))})
            record["same_synapses_statistics_counterfactual"] = outputs
            record["mean_shift_norm"] = float(a.xp.linalg.norm(a.f_mean - initial_mean))
            record["variance_shift_norm"] = float(a.xp.linalg.norm(a.f_var - initial_var))
            assert not record["default_importance_nonzero"] and record["default_stable_arrays_unchanged"]
            (run / f"seed-{seed}-default.npz").write_bytes(a.dumps())
            (run / f"seed-{seed}-altered-protection.npz").write_bytes(b.dumps())
            restored, _ = LocalALIF.loads(a.dumps(), device="gpu")
            assert restored.dumps() == a.dumps()
            record["cold_GPU_state_exact"] = True
            report["seeds"].append(record)
            trace.event("retention_mechanism.measured_seed", {"run_id": name, **record,
                "is_live_neural_counter": False}, "orchestrator")
            del a, b, trained, original_stats, restored
        assert time.monotonic() - started < 120
        for rel, digest in hashes.items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
        report["verdict"] = "STORED_IMPORTANCE_STABLE_BUFFERS_INERT_IN_CURRENT_HYBRID_UPDATE"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "P0_FAILED_DO_NOT_INFER"
    finally:
        report["seconds"] = time.monotonic() - started
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("retention_mechanism.completed", {
            k:v for k,v in report.items() if k != "seeds"}, "orchestrator")
        trace.close()
        print(encode({k:report[k] for k in ("run_id", "failures", "verdict", "seconds")}))
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
