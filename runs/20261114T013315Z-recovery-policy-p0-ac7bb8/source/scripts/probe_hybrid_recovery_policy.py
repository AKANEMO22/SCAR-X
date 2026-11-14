"""GPU-only untrained shadow proposals on actual observed faults; no execution."""

import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.recovery_policy import RecoveryPolicy
from scarx.hybrid.store import Store, encode


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-recovery-policy-p0-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    parent = ROOT / "runs/20261113T232337Z-incident-information-p0-2d5a24"
    actual_raw = (parent / "actual-file-observations.json").read_bytes()
    observations = json.loads(actual_raw)
    source_paths = [
        Path(__file__), ROOT / "src/scarx/evaluation/recovery_policy.py",
        ROOT / "src/scarx/evaluation/hybrid_incident_encoder.py",
        ROOT / "src/scarx/hybrid/learner.py", ROOT / "src/scarx/hybrid/cuda_alif.py",
        ROOT / "experiments/hybrid-recovery-policy-wrapper-p0-v1.md",
    ]
    hashes = {}
    for path in source_paths:
        rel = path.relative_to(ROOT)
        raw = path.read_bytes()
        destination = run / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        hashes[rel.as_posix()] = hashlib.sha256(raw).hexdigest()
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "observations_sha256": hashlib.sha256(actual_raw).hexdigest(),
        "parent_actual_observations": parent.name, "seeds": [260113, 260137, 260171],
        "max_seconds": 120, "training_or_action_execution": False,
    }), encoding="utf-8")
    trace = Store(ROOT / "local/hybrid-v1")
    report = {"run_id": name, "seeds": [], "failures": [], "training": False,
              "action_execution": False, "production_weights_or_router_changed": False}
    started = time.monotonic()
    try:
        for seed in (260113, 260137, 260171):
            policy = RecoveryPolicy(seed=seed)
            row = {"seed": seed, "proposals": []}
            for case in observations["cases"]:
                history = []
                for index in range(3):
                    history.append({
                        "tool": "read_file", "reported_status": "unknown", "error_type": "",
                        "empty_result": False, "spillover": False, "streak": index + 1,
                        "returned_epoch": case["probe_returned_epoch"] if index == 2
                            else case["missing_returned_epochs"][index],
                        "probe": {k: case["probe"][k] for k in (
                            "kind", "read_ok", "reported_not_found", "original_path_differs")}
                            if index == 2 else None,
                    })
                before = {
                    n: hashlib.sha256(policy.model.host(getattr(policy.model.core, n)).tobytes()).hexdigest()
                    for n in ("w_in", "w_rec", "w_out", "bias")}
                proposal = policy.propose("Inspect the actual file failure", history,
                    decision_epoch=case["probe_returned_epoch"] + 1)
                payload = policy.dumps()
                clone = RecoveryPolicy.loads(payload)
                assert clone.dumps() == payload
                replayed = clone.propose("Inspect the actual file failure", history,
                    decision_epoch=case["probe_returned_epoch"] + 1)
                assert replayed == proposal
                assert before == {
                    n: hashlib.sha256(policy.model.host(getattr(policy.model.core, n)).tobytes()).hexdigest()
                    for n in before}
                assert policy.model.updates == clone.model.updates == 0
                row["proposals"].append({
                    "case_id": case["id"], "proposal": proposal,
                    "exact_cold_GPU_proposal": True, "weights_unchanged": True,
                    "input_observations": history,
                    "no_goal_or_cause_label_passed_to_policy": True})
                del clone
            (run / f"seed-{seed}-untrained-recovery.zip").write_bytes(policy.dumps())
            report["seeds"].append(row)
            trace.event("recovery_policy.shadow_proposals", {
                "run_id": name, **row, "is_live_neural_counter": False,
                "policy_qualified": False}, "orchestrator")
            del policy
        for rel, digest in hashes.items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
        assert time.monotonic() - started < 120
        report["verdict"] = "RESEARCH_SHADOW_WRAPPER_P0_PASS_NO_LEARNING_CLAIM"
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "WRAPPER_P0_FAILED"
    finally:
        report["seconds"] = time.monotonic() - started
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("recovery_policy.wrapper_completed", {
            k:v for k,v in report.items() if k != "seeds"}, "orchestrator")
        trace.close()
        print(encode({k:report[k] for k in ("run_id", "verdict", "failures", "seconds")}))
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
