"""Independent data-only checks of raw file faults, frozen sources and inputs."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(run):
    def read(name):
        return json.loads((run / name).read_text(encoding="utf-8"))
    result, actual, manifest = read("result.json"), read("actual-file-observations.json"), read("manifest.json")
    problems = []
    for rel, expected in manifest["source_sha256"].items():
        if hashlib.sha256((run / "source" / rel).read_bytes()).hexdigest() != expected:
            problems.append("Frozen source bytes: " + rel)
    if actual["errors"] or result["failures"]:
        problems.append("Original receipt contains failures")
    if len(actual["cases"]) != 3 or len(result["records"]) != 3:
        problems.append("Incomplete finite bank")
    groups = {key: defaultdict(list) for key in ("current", "proposed")}
    for case, record in zip(actual["cases"], result["records"]):
        if case["id"] != record["case_id"]:
            problems.append("Changed case ordering")
        missing = [json.loads(r) for r in case["missing_outputs"]]
        if len(missing) != 3 or not all(r.get("not_found") is True for r in missing):
            problems.append("Expected actual three missing-file results")
        if not all(r == missing[0] for r in missing):
            problems.append("Nonidentical initial results")
        probe = json.loads(case["probe_outputs"][0])
        if probe.get("not_found") is True:
            label = "request_missing_source"
        elif "not a regular file" in str(probe.get("error", "")):
            label = "clarify_file_versus_directory"
        elif probe.get("content") and not probe.get("error"):
            label = "inspect_readable_bound_source"
        else:
            problems.append("Unannotatable actual probe outcome")
            continue
        if label != record["next_step_annotation"]:
            problems.append("Annotation disagrees with actual file outcome")
        if case["probe"].get("semantic_resolution_verified") is not False:
            problems.append("Self-verified semantic label")
        if "missing_returned_epochs" in case:
            epochs = case["missing_returned_epochs"] + [case["probe_returned_epoch"]]
            if epochs != sorted(epochs) or not all(t > 0 for t in epochs):
                problems.append("Backwards actual tool-return chronology")
        for key, shape in (("current", (8, 64)), ("proposed", (24, 64))):
            values = np.load(run / (case["id"] + "-" + key + ".npy"), allow_pickle=False)
            if values.dtype != np.float32 or values.shape != shape or not np.isfinite(values).all():
                problems.append("Invalid encoder matrix")
            digest = hashlib.sha256(values.tobytes()).hexdigest()
            if digest != record[key + "_input_sha256"]:
                problems.append("Saved matrix hash mismatch")
            groups[key][digest].append(label)
    bound = result["information_bound"]
    independently_computed = {}
    for key, group in groups.items():
        count = sum(len(v) for v in group.values())
        ceiling = sum(max(Counter(v).values()) for v in group.values()) / count
        independently_computed[key] = {"input_groups": len(group), "annotation_ceiling": ceiling}
        if len(group) != bound[key + "_input_groups"] or ceiling != bound[key + "_deterministic_label_ceiling"]:
            problems.append("Information-group metric mismatch")
    partial, full = (json.loads(actual["bounded_read"][k]) for k in ("partial", "full"))
    if partial.get("truncated") is not True or full.get("truncated") is not False:
        problems.append("Bounded read was not distinguished from full read")
    for key in ("zero_synapse_updates", "arrays_unchanged", "data_only_cold_restore_exact"):
        if result.get(key) is not True:
            problems.append("Persistence/update boundary failed: " + key)
    return {"run_id": run.name, "integrity_problems": problems,
            "independently_computed": independently_computed,
            "verdict": "FINITE_INPUT_COLLISION_VERIFIED" if not problems else "AUDIT_FAIL",
            "scope": "Three annotated probe states, not policy accuracy, optimal actions, representation learning or broad recovery",
            "retrained_or_changed_serving": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = verify(a.run)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if a.output:
        with a.output.open("x", encoding="utf-8") as f:
            f.write(payload + "\n")
    print(payload)
    raise SystemExit(bool(report["integrity_problems"]))
