"""Independent saved-data STOP certificate; stdlib only, no generation or model."""
import time
START = time.perf_counter()
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from collections import Counter
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys

ORIGINAL_SHA = "6cadbbe2113f94c88b44ecc87678adb5770bb552"
ORIGINAL_RUN = "20261115T193825Z-rc-operator-pilot-v1-a76d1c"
SNAPSHOTS = ("docs/research/REUSABLE_OPERATOR_DESIGN_20261116.md",
             "experiments/rc-operator-pilot-v1.md",
             "scripts/research/rc_operator_preflight.py", "scripts/research/rc_operator_world.py")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
    os.replace(temporary, path)


class Budget:
    def __init__(self):
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in ("peak_rss", "rss", "peak_paged", "paged",
                "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")]
        if os.name != "nt":
            raise RuntimeError("Windows receipt implementation required")
        self.counters = Counters
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.process = kernel.GetCurrentProcess()
        self.get_memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        self.get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        self.last = {}
        self.check()

    def check(self):
        info = self.counters()
        info.cb = ctypes.sizeof(info)
        if not self.get_memory(self.process, ctypes.byref(info), info.cb):
            raise OSError(ctypes.get_last_error())
        self.last = {"elapsed_seconds": time.perf_counter() - START,
                     "OS_peak_RSS_bytes": int(info.peak_rss), "current_RSS_bytes": int(info.rss),
                     "private_commit_bytes": int(info.private), "child_processes_created": 0,
                     "scope": "owned process equals owned tree; no subprocesses",
                     "limits": {"seconds": 10, "RSS_bytes": 100_000_000}}
        if self.last["elapsed_seconds"] > 10 or info.peak_rss >= 100_000_000:
            raise RuntimeError("Registered audit resource limit exceeded")
        return self.last


def state(value):
    result = tuple(value)
    if len(result) != 5 or any(type(v) is not int or v not in (0, 1, 2, 3) for v in result):
        raise ValueError("Invalid saved categorical state")
    return result


def shifted_orbit(value):
    return frozenset(tuple((v + offset) % 4 for v in value) for offset in (0, 1, 2, 3))


def rank(label, value):
    return hashlib.sha256(f"rcop-v1|261180|{label}|{value}".encode()).hexdigest()


def digits(value):
    return "".join(map(str, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authoring-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.authoring_commit):
        raise ValueError("Full locked authoring SHA required")
    root = Path(__file__).resolve().parents[2]
    source = root / "runs" / ORIGINAL_RUN
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    terminal = {"status": "AUDIT_FAILED", "model_calls_created": 0, "fits_created": 0,
                "dataset_generation_or_resampling": False, "original_model_outcome": "NOT_MEASURED"}
    budget = None
    try:
        budget = Budget()
        names = ("raw-data.json", "result.json", "manifest.json", "control-predictions.json")
        input_hashes = {name: sha(source / name) for name in names}
        input_hashes.update({"source/" + p: sha(source / "source" / p) for p in SNAPSHOTS})
        config = {"original_run": ORIGINAL_RUN, "original_source_sha": ORIGINAL_SHA,
                  "locked_audit_commit_asserted_by_launcher": args.authoring_commit,
                  "script_sha256": sha(Path(__file__)),
                  "protocol_sha256": sha(root / "experiments/rc-operator-stop-audit-v1.md"),
                  "input_hashes": input_hashes, "failing_state": [0, 0, 0, 2, 3],
                  "fixed_limits": {"seconds": 10, "RSS_bytes": 100_000_000},
                  "python": sys.version, "started_UTC": datetime.now(timezone.utc).isoformat()}
        config["config_sha256"] = hashlib.sha256(canonical(config)).hexdigest()
        write(output / "registration.json", config)
        raw = json.loads((source / "raw-data.json").read_text(encoding="utf-8"))
        original = json.loads((source / "result.json").read_text(encoding="utf-8"))
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        assert original["source_sha"] == manifest["source_sha"] == ORIGINAL_SHA
        assert original["raw_data_sha256"] == input_hashes["raw-data.json"]
        assert original["control_predictions_sha256"] == input_hashes["control-predictions.json"]
        assert set(manifest["source_hashes"]) == set(SNAPSHOTS)
        assert all(input_hashes["source/" + p] == h for p, h in manifest["source_hashes"].items())
        assert all(hashlib.sha256(canonical(raw[k])).hexdigest() == h for k, h in original["dataset_component_hashes"].items())
        assert set(raw) == {"partitions", "train"}, "D0 stopped before completing primitive DEV/sequence/control stages"
        partitions = {key: {state(s) for s in values} for key, values in raw["partitions"].items()}
        assert all(len(values) == len(raw["partitions"][key]) for key, values in partitions.items())
        train_set, iid_set, shift_set = (partitions[k] for k in ("train_states", "iid_dev_states", "shift_states"))
        assert (len(train_set), len(iid_set), len(shift_set)) == (512, 448, 64)
        assert not (train_set & iid_set or train_set & shift_set or iid_set & shift_set)
        universe = train_set | iid_set | shift_set
        assert len(universe) == 1024  # All tuples are domain-valid; cardinality exhausts the domain.
        orbit_keys = set()
        for s in universe:
            budget.check()
            orbit = shifted_orbit(s)
            assert len(orbit) == 4 and orbit <= universe
            orbit_keys.add(min(orbit))
        ordinary = {s for s in universe if max(Counter(s).values()) <= 3}
        four = {s for s in universe if max(Counter(s).values()) == 4}
        five = {s for s in universe if max(Counter(s).values()) == 5}
        assert (len(orbit_keys), len(ordinary), len(four), len(five)) == (256, 960, 60, 4)
        assert shift_set == four | five and partitions["all_five_unmeasured"] == five
        ordinary_keys = sorted({min(shifted_orbit(s)) for s in ordinary}, key=lambda s: rank("partition", digits(s)))
        assert len(ordinary_keys) == 240
        expected_train = set().union(*(shifted_orbit(k) for k in ordinary_keys[:128]))
        assert train_set == expected_train and iid_set == ordinary - train_set
        iid_keys = sorted(ordinary_keys[128:], key=lambda s: rank("iid-eval", digits(s)))[:16]
        four_keys = {min(shifted_orbit(s)) for s in four}
        shift_keys = sorted(four_keys, key=lambda s: rank("shift-eval", digits(s)))[:8]
        assert len(four_keys) == 15 and len({min(shifted_orbit(s)) for s in five}) == 1
        assert partitions["iid_eval_states"] == set().union(*(shifted_orbit(k) for k in iid_keys))
        assert partitions["shift_eval_states"] == set().union(*(shifted_orbit(k) for k in shift_keys))
        assert (len(partitions["iid_eval_states"]), len(partitions["shift_eval_states"])) == (64, 32)
        commands = [("MOVE", i, -1, location) for i in range(5) for location in range(4)]
        commands += [("SWAP", i, j, -1) for i, j in combinations(range(5), 2)]
        command_set = set(commands)
        train, quotas, coverage, prestates, ids = raw["train"], Counter(), Counter(), Counter(), set()
        assert len(train) == 512
        input_rows, target_rows = [], []
        for record in train:
            budget.check()
            before, target, command = state(record["state"]), state(record["target"]), tuple(record["command"])
            assert command in command_set
            op, i, j, location = command
            expected = tuple(location if slot == i else value for slot, value in enumerate(before)) if op == "MOVE" else tuple(
                before[j] if slot == i else before[i] if slot == j else value for slot, value in enumerate(before))
            assert target == expected and before in train_set and target in train_set
            payload = f"{digits(before)}|{':'.join(map(str, command))}|{digits(target)}"
            assert record["id"] == rank("record-id", payload) and record["id"] not in ids
            ids.add(record["id"])
            assert record["changed"] == (before != target)
            assert record["stratum"] == "TRAIN" and record["noop_diagnostic"] is False
            quotas[f"{op}:{'changed' if target != before else 'noop'}"] += 1
            coverage[":".join(map(str, command))] += 1
            prestates[before] += 1
            input_rows.append(before)
            target_rows.append(target)
        assert quotas == {"MOVE:changed": 224, "MOVE:noop": 32, "SWAP:changed": 224, "SWAP:noop": 32}
        assert len(prestates) >= 256 and max(prestates.values()) <= 2
        assert all(coverage[":".join(map(str, c))] >= (8 if c[0] == "MOVE" else 16) for c in commands)
        receipt = original["train_receipt"]
        assert quotas == receipt["quota_counts"] and coverage == receipt["command_counts"]
        assert len(prestates) == receipt["distinct_prestates"] and max(prestates.values()) == receipt["max_records_per_prestate"]
        assert len(train) == receipt["selected_records"]
        balance = {}
        for key, rows in (("state", input_rows), ("target", target_rows)):
            counts = [[sum(row[slot] == location for row in rows) for location in range(4)] for slot in range(5)]
            fractions = [[value / 512 for value in row] for row in counts]
            assert counts == original["balance"][key]["counts"]
            assert fractions == original["balance"][key]["fractions"]
            assert all(.2 <= value <= .3 for row in fractions for value in row)
            assert original["balance"][key]["within_registered_20_to_30_percent"] is True
            balance[key] = {"counts": counts, "min_fraction": min(map(min, fractions)), "max_fraction": max(map(max, fractions))}
        failing = (0, 0, 0, 2, 3)
        assert failing in partitions["iid_eval_states"] and failing not in train_set
        swap_rows = []
        for i, j in combinations(range(5), 2):
            target = tuple(failing[j] if slot == i else failing[i] if slot == j else value for slot, value in enumerate(failing))
            changed, in_train = target != failing, target in train_set
            swap_rows.append({"pair": [i, j], "target": target, "changed": changed,
                              "target_in_TRAIN": in_train, "valid_changed_heldout": changed and not in_train})
        valid = sum(row["valid_changed_heldout"] for row in swap_rows)
        assert valid < 2
        assert original["status"] == "STOP_DATA" and original["stage"] == "primitive_dev"
        assert original["reason"] == "DEV valid-command shortage: IID/(0, 0, 0, 2, 3)/SWAP"
        assert all(original[key] == 0 for key in ("model_calls", "learning_updates", "neural_fits"))
        audit = {"audit_pass": True, "classification": "CERTIFIED_DATA_PREREQUISITE_STOP",
                 "model_capability_outcome": "NOT_MEASURED", "dataset_generator_rerun": False,
                 "independence": "same Codex team, separately implemented tuple/count audit; not external peer review",
                 "source_run": ORIGINAL_RUN, "original_source_sha": ORIGINAL_SHA,
                 "saved_TRAIN_labels_checked": 512, "distinct_TRAIN_prestates": len(prestates),
                 "quotas": dict(quotas), "balance": balance,
                 "partitions": {"all_states": 1024, "orbits": 256, "ordinary_states": 960,
                                "ordinary_orbits": 240, "exactly_four_states": 60, "exactly_four_orbits": 15,
                                "all_five_states": 4, "all_five_orbits": 1, "TRAIN_orbits": 128, "IID_DEV_orbits": 112},
                 "failing_state": failing, "SWAP_rows": swap_rows, "valid_changed_heldout_SWAP_count": valid,
                 "required_changed_SWAP_count": 2,
                 "recorded_neural_counts_zero": True, "recorded_counts_are_not_machine_wide_process_audit": True}
        write(output / "audit.json", audit)
        terminal["status"] = "COMPLETED_AUDIT_PASS"
        terminal["certified_original_outcome"] = "STOP_DATA"
        terminal["valid_changed_heldout_SWAP_count"] = valid
    except Exception as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
    finally:
        if budget is not None:
            try:
                budget.check()
            except Exception as error:
                terminal["status"], terminal["error"] = "RESOURCE_OR_RECEIPT_FAILED", repr(error)
            write(output / "resource.json", budget.last)
        write(output / "output-hashes.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
        terminal["finished_UTC"] = datetime.now(timezone.utc).isoformat()
        write(output / "status.json", terminal)
        print(json.dumps(terminal), flush=True)
    return 0 if terminal["status"] == "COMPLETED_AUDIT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
