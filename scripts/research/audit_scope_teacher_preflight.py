"""Independent stdlib scalar audit of saved preflight; no model imports/fit."""
import argparse
import csv
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import json
import math
from pathlib import Path
import time


def memory():
    class PMC(ct.Structure):
        _fields_ = [("cb", wt.DWORD), ("faults", wt.DWORD)] + [
            (name, ct.c_size_t) for name in (
                "peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
                "nonpaged", "commit", "peak_commit", "private",
            )
        ]
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    psapi = ct.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wt.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ct.POINTER(PMC), wt.DWORD]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    info = PMC()
    info.cb = ct.sizeof(info)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ct.byref(info), info.cb):
        raise OSError(ct.get_last_error(), "memory query")
    return {key: int(getattr(info, key)) for key in ("peak_rss", "rss", "commit", "peak_commit", "private")}


def main():
    start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    run = ap.parse_args().run.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    cases = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    checks = json.loads((run / "checks.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((run / "raw.csv").open(encoding="utf-8", newline="")))
    tapes = json.loads((run / "traces.json").read_text(encoding="utf-8"))
    failures = []
    def require(value, reason):
        if not value:
            failures.append(reason)
    for name, sha in manifest["source_hashes"].items():
        require(hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == sha,
                "source archive hash: " + name)
    by_id = {}
    for case in cases:
        seq = case["sequence"]
        require(len(seq) == 10 and all(len(r) == 64 for r in seq), "input shape")
        pulses = [i for i, row in enumerate(seq) if row[0] != 0]
        require(len(pulses) == 2 and pulses[1] == 8, "two causal pulses, fixed endpoint")
        require(all(all(v == 0 for v in row[1:]) for row in seq), "unexpected input channels")
        gap = pulses[1] - pulses[0]
        require(gap == case["gap"] and case["label"] == int(gap <= 2), "label independently decoded from input times")
        require(all(abs(seq[t][0] - case["amplitude"]) < 1e-6 for t in pulses), "pulse amplitude")
        by_id[case["id"]] = case
    require(len(rows) == 144 and len(tapes) == 144, "expected 3 seeds x4 constructed configurations x12cases")
    lookup = {(r["seed"], r["role"], r["case_id"]): r for r in rows}
    require(len(lookup) == 144, "duplicate or missing prediction entries")
    max_error = 0.
    constant_targets = {0: set(), 1: set()}
    spike_label_identity = True
    frozen_collision = True
    teacher_correct = witness_correct = 0
    for trace in tapes:
        key = (str(trace["seed"]), trace["role"], trace["case_id"])
        row = lookup[key]
        case = by_id[trace["case_id"]]
        role = trace["role"]
        weight = .3 if role in ("student_plus", "student_minus") else .4
        sign = -1 if role == "student_minus" else 1
        v = a = f = 0.
        zs = []
        for index, input_row in enumerate(case["sequence"]):
            theta = .65 + .08 * a
            u = .85 * v + weight * input_row[0]
            z = float(u >= theta)
            v = u - z * theta
            a = .95 * a + z
            f = .92 * f + .08 * z
            zs.append(z)
            expected = {"u": u, "theta": theta, "spike": z, "v": v, "a": a, "f": f}
            for name, value in expected.items():
                error = abs(value - trace["tape"][index][name])
                max_error = max(max_error, error)
                require(error < 2e-6, "scalar ALIF mismatch: " + str(key) + "/" + name)
        ell1 = sign * 3 * f - 1.5 * .08 * .92
        p1 = 1 / (1 + math.exp(-ell1))
        pred = int(p1 > .5)
        require(abs(float(row["p1"]) - p1) < 2e-7 and abs(float(row["p0"]) - (1 - p1)) < 2e-7,
                "independent probability")
        require(int(row["prediction"]) == pred, "prediction argmax")
        require(row["weights_before"] == row["weights_after"] and int(row["updates"]) == 0,
                "weight invariant/no learning")
        require(all(value == 0 for i, value in enumerate(trace["features"]) if i != trace["active"]),
                "unexpected active feature")
        if role == "teacher":
            teacher_correct += pred == case["label"]
            constant_targets[case["label"]].add(float(row["p1"]))
            spike_label_identity &= zs == [float(case["label"]) if t == 8 else 0. for t in range(10)]
        elif role == "student_witness":
            witness_correct += pred == case["label"]
        else:
            frozen_collision &= all(value == 0 for value in trace["features"])
    require(teacher_correct == 36 and witness_correct == 36, "constructed teacher/student competence")
    require(frozen_collision and checks["readout_only_balanced_ceiling"] == .5, "readout-only collision ceiling")
    require(spike_label_identity, "teacher P spike tape equals a known class-conditional constant tape")
    require(all(len(v) == 1 for v in constant_targets.values()), "teacher terminal probabilities label-table identity")
    require(checks["gates"]["G5_teacher_identifiable_beyond_class_targets"] is False,
            "preflight must reject nonidentifying terminal teacher signal")
    require(checks["gates"]["G6_new_credit_path_not_P0_sign_repetition"] is False,
            "no novel credit path asserted")
    result = {"pass": not failures, "failures": failures,
              "scope": "Independent scalar equations/hash/labels audit, same author; not independent quality replication",
              "teacher_correct": teacher_correct, "constructed_witness_correct": witness_correct,
              "teacher_class_constants": {str(k): sorted(v) for k, v in constant_targets.items()},
              "teacher_spike_tape_class_identity": spike_label_identity,
              "frozen_feature_collision": frozen_collision, "max_scalar_forward_error": max_error,
              "learning_calls": 0, "model_imports": False, "wall_seconds": time.perf_counter() - start,
              "memory": memory(), "verdict_reconstructed": "PREFLIGHT_NO_GO"}
    (run / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
