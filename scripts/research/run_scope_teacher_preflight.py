"""No-training CPU preflight. This script cannot launch a P1 learner."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import csv
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from scarx.runtime.limits import apply_memory_limit
LIMIT = 2_000_000_000
CAP = apply_memory_limit(LIMIT)
import numpy as np
from scarx.neural.core import Config, SpikingCore

SEEDS = (261151, 261157, 261163)
PROTOCOL = "experiments/scope-p1-teacher-preflight-20261115.md"
REFERENCE = (
    PROTOCOL, "scripts/research/run_scope_teacher_preflight.py",
    "scripts/research/audit_scope_teacher_preflight.py",
    "src/scarx/neural/core.py", "src/scarx/hybrid/learner.py",
    "src/scarx/evaluation/readout_eligibility.py", "src/scarx/runtime/limits.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
        raise OSError(ct.get_last_error(), "GetProcessMemoryInfo")
    return {name: int(getattr(info, name)) for name in ("peak_rss", "rss", "commit", "peak_commit", "private")}


def budget():
    if time.perf_counter() - START >= 115:
        raise TimeoutError("120s aggregate budget: leave audit/serialization reserve")
    m = memory()
    if max(m["peak_rss"], m["peak_commit"]) >= LIMIT:
        raise MemoryError("Preflight exceeded RAM/commit budget")


def initialized(seed, role):
    teacher = role == "teacher"
    h, active = (128, 64) if teacher else (64, 0)
    model = SpikingCore(Config(inputs=64, hidden=h, outputs=2, fan_in=8,
                               plastic_rows=12, seed=seed))
    if teacher:
        rng = np.random.default_rng(seed + 7000)
        for row in range(64):
            model.indices[row] = rng.choice(64, 8, replace=False)
        for row in range(64, 128):
            model.indices[row] = np.concatenate((rng.choice(64, 4, replace=False),
                                                64 + rng.choice(64, 4, replace=False)))
    # Declared constructed initialization, never learned or fitted.
    model.w_in.fill(0)
    model.w_rec.fill(0)
    model.w_out.fill(0)
    model.bias.fill(0)
    weight = .3 if role in ("student_plus", "student_minus") else .4
    sign = -1 if role == "student_minus" else 1
    model.w_in[active, 0] = weight
    model.w_out[1, active] = sign * 3
    model.bias[1] = np.float32(-1.5 * (1 - .92) * .92)
    for name in ("in", "rec", "out", "bias"):
        current = model.bias if name == "bias" else getattr(model, "w_" + name)
        getattr(model, "stable_" + name)[:] = current
    return model, active, sign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("Owned new runs directory required")
    run.mkdir(parents=True, exist_ok=False)
    sources = {name: digest((ROOT / name).read_bytes()) for name in REFERENCE}
    for name in REFERENCE:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    manifest = {
        "protocol": PROTOCOL, "source_hashes": sources, "seeds": list(SEEDS),
        "roles": ["teacher", "student_plus", "student_minus", "student_witness"],
        "kind": "constructed_forward_preflight_only", "learning_calls": 0,
        "wall_limit_seconds": 120, "memory_limit_bytes": LIMIT,
        "memory_cap": CAP, "device": "CPU", "old_TEST_access": False,
    }
    write_json(run / "manifest.json", manifest)
    cases = []
    for gap in (1, 2, 4, 5):
        for amplitude in (.96, 1., 1.04):
            seq = np.zeros((10, 64), np.float32)
            seq[8 - gap, 0] = seq[8, 0] = amplitude
            cases.append({"id": f"gap{gap}-amp{amplitude}", "gap": gap,
                          "amplitude": amplitude, "label": int(gap <= 2),
                          "split": "reserved_train_like" if gap in (1, 5) and amplitude == 1.
                          else "preflight_novel_coordinates", "sequence": seq.tolist()})
    write_json(run / "inputs.json", cases)
    rows, tapes = [], []
    for seed in SEEDS:
        for role in manifest["roles"]:
            model, active, sign = initialized(seed, role)
            before = model.fingerprint()
            for case in cases:
                budget()
                model.reset_state()
                tape = []
                for row in np.asarray(case["sequence"], np.float32):
                    threshold = model.config.threshold + model.config.adaptation_strength * model.adaptation[active]
                    u = (model.config.leak * model.voltage[active]
                         + model.w_in[active] @ row
                         + np.sum(model.w_rec[active] * model.spikes[model.indices[active]]))
                    model.step(row, track=False)
                    tape.append({"u": float(u), "theta": float(threshold),
                                 "spike": float(model.spikes[active]),
                                 "v": float(model.voltage[active]),
                                 "a": float(model.adaptation[active]),
                                 "f": float(model.features[active])})
                p = model.probabilities()
                if model.fingerprint() != before or model.updates != 0:
                    raise AssertionError("Preflight changed weights or learning counter")
                rows.append({"seed": seed, "case_id": case["id"], "role": role,
                             "sign": sign, "gap": case["gap"], "amplitude": case["amplitude"],
                             "label": case["label"], "prediction": int(p.argmax()),
                             "p0": float(p[0]), "p1": float(p[1]),
                             "feature_active": float(model.features[active]), "u_second": tape[8]["u"],
                             "weights_before": before, "weights_after": model.fingerprint(), "updates": model.updates})
                tapes.append({"seed": seed, "role": role, "case_id": case["id"],
                              "active": active, "tape": tape, "features": model.features.tolist()})
    with (run / "raw.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(run / "traces.json", tapes)
    teacher = [r for r in rows if r["role"] == "teacher"]
    witness = [r for r in rows if r["role"] == "student_witness"]
    initial = [r for r in rows if r["role"].startswith("student_") and r["role"] != "student_witness"]
    const = {str(y): sorted({r["p1"] for r in teacher if r["label"] == y}) for y in (0, 1)}
    feature_collision = all(all(v == 0 for v in t["features"])
                            for t in tapes if t["role"] in ("student_plus", "student_minus"))
    u_spans = [max(r["u_second"] for r in initial if r["seed"] == seed and r["amplitude"] == a)
               - min(r["u_second"] for r in initial if r["seed"] == seed and r["amplitude"] == a)
               for seed in SEEDS for a in (.96, 1.04)]
    gates = {
        "G1_input_and_temporal_effect": min(u_spans) > 0,
        "G2_constructed_teacher_competent": all(r["prediction"] == r["label"] for r in teacher),
        "G3_constructed_student_representable": all(r["prediction"] == r["label"] for r in witness),
        "G4_readout_only_collision_ceiling": feature_collision,
        "G5_teacher_identifiable_beyond_class_targets": not all(len(v) == 1 for v in const.values()),
        "G6_new_credit_path_not_P0_sign_repetition": False,
    }
    checks = {"gates": gates, "teacher_class_probability_table": const,
              "minimum_initial_voltage_span": min(u_spans),
              "readout_only_balanced_ceiling": .5 if feature_collision else None,
              "teacher_correct": sum(r["prediction"] == r["label"] for r in teacher),
              "teacher_cases": len(teacher), "witness_correct": sum(r["prediction"] == r["label"] for r in witness),
              "witness_cases": len(witness), "forward_calls": len(rows), "core_ticks": len(rows) * 10,
              "learning_calls": 0, "quality_confirmation": False,
              "G6_reason": "Only terminal targets changed; output feedback/eligibility path is unchanged. Binary teacher class targets can be generated from verified labels without P."}
    write_json(run / "checks.json", checks)
    budget()
    remaining = 120 - (time.perf_counter() - START)
    child = subprocess.run([sys.executable, str(ROOT / "scripts/research/audit_scope_teacher_preflight.py"),
                            "--run", str(run)], capture_output=True, text=True,
                           encoding="utf-8", timeout=max(.1, remaining - .5))
    (run / "audit.stdout.txt").write_text(child.stdout, encoding="utf-8")
    (run / "audit.stderr.txt").write_text(child.stderr, encoding="utf-8")
    audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    peaks = memory()
    aggregate = {key: peaks[key] + audit["memory"][key] for key in ("peak_rss", "peak_commit")}
    elapsed = time.perf_counter() - START
    resource_ok = max(aggregate.values()) < LIMIT and elapsed <= 120
    result = {"verdict": "PREFLIGHT_NO_GO" if not all(gates.values()) else "PREFLIGHT_PASS_NOT_P1_AUTHORIZATION",
              "reason": "Terminal teacher P is class-conditional label smoothing; not a discriminating P1 for causal teacher benefit or bad-sign credit rescue.",
              "gates": gates, "forward_calls": len(rows), "learning_updates": 0,
              "P1_registered": False, "P1_started": False, "teacher_was_constructed_not_trained": True,
              "seeds_are_technical_topology_checks_not_quality_replications": True,
              "audit_exit_code": child.returncode, "audit_pass": audit["pass"],
              "worker_memory": peaks, "aggregate_conservative_peak_bounds": aggregate,
              "memory_cap": CAP, "wall_seconds_including_audit": elapsed,
              "resource_pass": resource_ok, "CPU_only": not any(k.startswith(("torch", "cupy")) for k in sys.modules),
              "old_skills_retention": "NOT_MEASURED", "learned_student_restart": "NOT_MEASURED",
              "scientific_CI_or_20stage_PASS": False}
    result["evidence_hashes"] = {p.name: digest(p.read_bytes()) for p in run.iterdir() if p.is_file()}
    write_json(run / "result.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if resource_ok and child.returncode == 0 and audit["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
