"""Data-only forensic verification; no CUDA/learner update or live state access."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    problems, rows = [], []
    for rel, expected in manifest["source_sha256"].items():
        if hashlib.sha256((run / "source" / rel).read_bytes()).hexdigest() != expected:
            problems.append("Frozen source mismatch: " + rel)
    if result["failures"] or len(result["seeds"]) != len(manifest["seeds"]):
        problems.append("Original numerical audit incomplete")
    for seed, record in zip(manifest["seeds"], result["seeds"]):
        with np.load(run / f"seed-{seed}-default.npz", allow_pickle=False) as a, \
                np.load(run / f"seed-{seed}-altered-protection.npz", allow_pickle=False) as b:
            if seed != record["seed"] or len(record["steps"]) != 32:
                problems.append("Wrong seed/update count")
            trainables_equal = all(np.array_equal(a["core." + k], b["core." + k])
                                   for k in ("w_in", "w_rec", "w_out", "bias"))
            trace_equal = all(not any(r["trainable_weight_maxdifference"].values())
                              and r["loss_default"] == r["loss_importance1000_stable_shift"]
                              for r in record["steps"])
            altered_correctly = (
                np.count_nonzero(a["importance"]) == 0
                and np.all(b["importance"] == 1000)
                and all(np.allclose(b["core.stable_" + suffix] -
                                    a["core.stable_" + suffix], .25, atol=1e-7)
                        for suffix in ("in", "rec", "out", "bias")))
            if not trainables_equal or not trace_equal or not altered_correctly:
                problems.append("Counterfactual archive disagrees: " + str(seed))
            max_delta, changed = 0.0, 0
            for output in record["same_synapses_statistics_counterfactual"]:
                p = np.array(output["with_online_statistics"], dtype=np.float32)
                q = np.array(output["with_initial_statistics_same_weights"], dtype=np.float32)
                delta = float(np.max(np.abs(p - q)))
                if delta != output["max_probability_delta"] or (
                    int(np.argmax(p)) != int(np.argmax(q))) != output["rank_changed"]:
                    problems.append("Statistic-output metric mismatch")
                max_delta = max(max_delta, delta)
                changed += output["rank_changed"]
            meta = json.loads(a["metadata"].tobytes())
            if meta["updates"] != 32 or meta["hidden_updates"] != 32:
                problems.append("Checkpoint update count mismatch")
            rows.append({"seed": seed, "trainables_identical": trainables_equal,
                "paired_trace_identical": trace_equal,
                "stored_protection_intervention_verified": altered_correctly,
                "max_statistics_probability_delta": max_delta,
                "argmax_changes_on4_inputs": changed})
    return {"run_id": run.name, "integrity_problems": problems, "seeds": rows,
            "verdict": "INERT_PROTECTION_COUNTERFACTUAL_CONFIRMED" if not problems else "AUDIT_FAIL",
            "scope": "Current Hybrid update path, not historical PlasticCore/metaplasticity research or broad retention",
            "changes_training_or_serving": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = verify(args.run)
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(bool(result["integrity_problems"]))
