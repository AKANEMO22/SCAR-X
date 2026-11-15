"""P1 re-audit: raw-data analysis only, no model imports or neural replay."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def main():
    start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    original, output = args.original.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(original):
        raise ValueError("New audit output outside immutable original run required")
    result = json.loads((original / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    custody = json.loads((original / "custody.json").read_text(encoding="utf-8"))
    cases = {x["id"]: x for x in json.loads((original / "inputs.json").read_text(encoding="utf-8"))}
    rows = list(csv.DictReader((original / "raw.csv").open(encoding="utf-8", newline="")))
    tapes = json.loads((original / "traces.json").read_text(encoding="utf-8"))
    errors = []
    def need(ok, why):
        if not ok:
            errors.append(why)
    for name, sha in result["evidence_hashes"].items():
        need(hashlib.sha256((original / name).read_bytes()).hexdigest() == sha, "evidence hash " + name)
    for name, item in custody["files"].items():
        need(hashlib.sha256((original / name).read_bytes()).hexdigest() == item["sha256"], "custody hash " + name)
    for name, sha in manifest["source_hashes"].items():
        need(hashlib.sha256((original / "source" / name).read_bytes()).hexdigest() == sha, "source hash " + name)
    need(len(rows) == 144 and len(tapes) == 144, "expected144records")
    lookup = {(str(t["seed"]), t["role"], t["case_id"]): t for t in tapes}
    values = {0: [], 1: []}
    correct = {}; p_ablation = []; max_u_error = 0.
    ro_feature_vectors = []
    for r in rows:
        key = (r["seed"], r["role"], r["case_id"])
        t = lookup[key]; c = cases[r["case_id"]]
        pulses = [i for i, v in enumerate(c["sequence"]) if v[0] != 0]
        label = int(pulses[1] - pulses[0] <= 2)
        need(label == int(r["label"]) == c["label"], "labels decoded from observed input")
        need(pulses[1] == 8 and len(pulses) == 2, "fixed causal endpoints")
        need(all(all(v == 0 for v in x[1:]) for x in c["sequence"]), "no targetchannel")
        role = r["role"]; w = .3 if role in ("student_plus", "student_minus") else .4
        expected_u = w * c["sequence"][8][0] * (1 + .85 ** c["gap"])
        max_u_error = max(max_u_error, abs(expected_u - float(r["u_second"])))
        need(abs(expected_u - float(r["u_second"])) < 2e-6, "nonzero weight temporal effect")
        need(r["weights_before"] == r["weights_after"] and int(r["updates"]) == 0, "no learning")
        sign = int(r["sign"])
        score = sign * 3 * float(t["features"][t["active"]]) - 1.5 * .08 * .92
        p1 = 1 / (1 + math.exp(-score))
        need(abs(p1 - float(r["p1"])) < 2e-7 and abs(1 - p1 - float(r["p0"])) < 2e-7,
             "two-class softmax p1/p0")
        correct.setdefault(role, [0, 0]);correct[role][0] += int(r["prediction"]) == label;correct[role][1] += 1
        if role == "teacher":
            values[label].append(Decimal(r["p1"]))
            need(all(v == 0 for v in t["features"][:64]), "teacher Sfeature zero")
            need(t["active"] == 64 and all(v == 0 for i, v in enumerate(t["features"]) if i != 64), "head support lies in P")
            without_P_p1 = 1 / (1 + math.exp(1.5 * .08 * .92))
            p_ablation.append(int(without_P_p1 > .5) == label)
        elif role in ("student_plus", "student_minus"):
            ro_feature_vectors.append(tuple(t["features"]))
    class_stats = {}
    for y, vals in values.items():
        class_stats[str(y)] = {"n_records": len(vals), "unique_p1_full_precision": [str(v) for v in sorted(set(vals))],
                               "population_variance_decimal": str(sum((v - sum(vals)/len(vals))**2 for v in vals)/len(vals))}
        need(len(set(vals)) == 1, "teacher terminal classconstant")
    need(len(set(ro_feature_vectors)) == 1, "readout-only exact featurecollision")
    need(max_u_error < 2e-6, "voltage formula")
    need(all(v == [36, 36] for k, v in correct.items() if k in ("teacher", "student_witness")), "constructed competence")
    need(sum(p_ablation) / len(p_ablation) == .5, "P ablation from saved features")
    need(result["forward_calls"] == 144 and result["learning_updates"] == 0, "record counts")
    worker_lines = (original / "worker.stdout.txt").read_text(encoding="utf-8-sig").strip().splitlines()
    need(json.loads(worker_lines[-1]) == result, "stdout/result match")
    need(result["aggregate_conservative_peak_bounds"]["peak_rss"] == result["worker_memory"]["peak_rss"] +
         json.loads((original/"audit.json").read_text())["memory"]["peak_rss"], "RSS aggregation")
    report = {"verdict": "P1_NO_GO_CONFIRMED_BY_SAVED_DATA" if not errors else "AUDIT_ISSUES",
              "failures": errors, "new_neural_forwards": 0, "new_learning_updates": 0,
              "old_run_files_changed": False, "unique_input_cases": len(cases), "raw_records": len(rows),
              "class_stats": class_stats, "correct_by_role": correct,
              "teacher_P_ablated_saved_readout_accuracy": sum(p_ablation)/len(p_ablation),
              "initial_readout_unique_feature_vectors": len(set(ro_feature_vectors)),
              "max_input_voltage_formula_error": max_u_error,
              "source_level_findings": ["Teacher/witness were declared hand constructions, not trained checkpoints",
                  "Sparse topology/bounds checked at source level, not independently reconstructed from serialized weights",
                  "G6 was assigned False by logic in runner and asserted by old audit; not a measured neural gate",
                  "Three topology seeds do not provide independent quality trials because useful weights and R=0 are fixed"],
              "resource_audit_scope": "Historical measured values agree across archived stdout/JSON and source instrumentation; cannot independently remeasure old wall/RSS without rerun, which was not performed",
              "same_update_conditions": ["same weights/input/normalizer/head/eligibility/optimizer/RNG/order/masks",
                  "replace T(x)=c_y by same label-derived vector c_y", "same arithmetic and clipping => same physical update by induction"],
              "CPU_wall_seconds": time.perf_counter()-start,
              "interpretation": "P needed for constructed teacher decision, but terminal output adds no sample-specific teaching content beyond the class table; no learned transfer/restart/lifelong result"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
