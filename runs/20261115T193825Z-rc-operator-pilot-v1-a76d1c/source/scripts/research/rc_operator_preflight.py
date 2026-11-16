"""Execute RC-OPERATOR v1 D0 once after root commits reviewed source/protocol.

CPU1 / 20s / peakRSS300MB. Pure data/counts/codec/oracle/retrieval; no neural
model, hidden fit, optimizer, autograd, CUDA or live-service calls.
"""
import os
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from collections import Counter, defaultdict
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import time

START_WALL = time.perf_counter()
START_CPU = time.process_time()
ROOT = Path(__file__).resolve().parents[2]
import rc_operator_world as world

SOURCE_PATHS = (
    "scripts/research/rc_operator_world.py", "scripts/research/rc_operator_preflight.py",
    "experiments/rc-operator-pilot-v1.md", "docs/research/REUSABLE_OPERATOR_DESIGN_20261116.md",
)
REPORT_PATH = ROOT / "docs/research/RC_OPERATOR_PREFLIGHT_20261116.md"
ACTIVE_BUDGET = None


class ResourceStop(RuntimeError):
    pass


def write_json(path, value):
    raw = world.canonical_json(value) + b"\n"
    temp = path.with_name(path.name + ".partial")
    with temp.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return hashlib.sha256(raw).hexdigest()


def git(*arguments):
    # Provenance subprocesses are owned, sampled and charged to D0 as well.
    process = subprocess.Popen(["git", *arguments], cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    started = time.perf_counter()
    try:
        while True:
            ACTIVE_BUDGET.sample_child(int(process._handle))
            ACTIVE_BUDGET.check()
            if process.poll() is not None:
                break
            if time.perf_counter() - started > 3:
                raise ResourceStop("Owned git provenance subprocess timeout")
            time.sleep(.001)
        output, error = process.communicate()
        if process.returncode:
            raise RuntimeError("Git provenance failure: " + error.strip())
        return output.strip()
    finally:
        if process.poll() is None:
            process.kill()  # Only this owned short-lived provenance child.
            process.communicate()


class Budget:
    def __init__(self):
        if os.name != "nt":
            raise ResourceStop("This locked receipt implementation requires Windows")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.process = kernel.GetCurrentProcess()
        kernel.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(self.process, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise ResourceStop("Cannot inspect owned-process CPU affinity")
        self.affinity_before = int(process_mask.value)
        self.affinity = self.affinity_before & -self.affinity_before
        if not self.affinity or not kernel.SetProcessAffinityMask(self.process, self.affinity):
            raise ResourceStop("Cannot enforce one logical CPU on owned D0 process")

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        self.counters_type = Counters
        self.get_memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        self.get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        self.peak_private = 0
        self.child_peak_rss = 0
        self.child_peak_private = 0
        self.child_memory_samples = 0
        self.samples = 0
        self.last = {}
        self.check()

    def sample_child(self, handle):
        counters = self.counters_type()
        counters.cb = ctypes.sizeof(counters)
        if self.get_memory(handle, ctypes.byref(counters), counters.cb):
            self.child_peak_rss = max(self.child_peak_rss, int(counters.PeakWorkingSetSize))
            self.child_peak_private = max(self.child_peak_private, int(counters.PrivateUsage))
            self.child_memory_samples += 1

    def check(self):
        counters = self.counters_type()
        counters.cb = ctypes.sizeof(counters)
        if not self.get_memory(self.process, ctypes.byref(counters), counters.cb):
            raise ResourceStop("Owned D0 process memory receipt unavailable")
        self.peak_private = max(self.peak_private, int(counters.PrivateUsage))
        self.samples += 1
        self.last = {"elapsed_active_wall_seconds": time.perf_counter() - START_WALL,
                     "cpu_seconds": time.process_time() - START_CPU,
                     "peak_rss_bytes": int(counters.PeakWorkingSetSize),
                     "peak_tree_rss_conservative_bound_bytes": int(counters.PeakWorkingSetSize) + self.child_peak_rss,
                     "provenance_child_peak_rss_bytes": self.child_peak_rss,
                     "provenance_child_memory_samples": self.child_memory_samples,
                     "current_rss_bytes": int(counters.WorkingSetSize),
                     "sampled_peak_private_commit_bytes": self.peak_private,
                     "sampled_peak_tree_private_commit_bound_bytes": self.peak_private + self.child_peak_private,
                     "peak_pagefile_usage_bytes": int(counters.PeakPagefileUsage),
                     "sample_count": self.samples, "logical_cpu_count_allowed": 1,
                     "affinity_mask_before": self.affinity_before, "affinity_mask_used": self.affinity,
                     "caps": {"wall_seconds": 20, "peak_rss_bytes": 300_000_000}}
        if self.last["elapsed_active_wall_seconds"] > 20:
            raise ResourceStop("D0 20-second cap reached")
        if self.last["peak_tree_rss_conservative_bound_bytes"] > 300_000_000:
            raise ResourceStop("D0 300MB peakRSS cap reached")
        return self.last


def independent_oracle(state, command):
    op, first, second, destination = command
    if op == "MOVE":
        return tuple(destination if index == first else value for index, value in enumerate(state))
    return tuple(state[second] if index == first else state[first] if index == second else value
                 for index, value in enumerate(state))


def contracts(check):
    seen, count = set(), 0
    for index, state in enumerate(product(range(4), repeat=5)):
        state = tuple(state)
        if index % 32 == 0:
            check()
        encoded = world.encode_state(state)
        if world.decode_state(encoded) != state or world.identity_feedback(encoded) != encoded:
            raise ValueError("State roundtrip or neutral feedback failure")
        for command in world.COMMANDS:
            vector = world.encode_input(state, command)
            target = world.oracle(state, command)
            independent = independent_oracle(state, command)
            if target != independent or len(vector) != 64 or vector[:20] != encoded or any(vector[36:]):
                raise ValueError("Oracle/codec contract violation")
            # The codec receives no target. Changing evaluator labels cannot
            # enter its signature. Each actual serialized input must be unique.
            key = bytes(int(v) for v in vector)
            if key in seen:
                raise ValueError("Input alias across distinct state/command pairs")
            seen.add(key)
            if command[0] == "MOVE" and world.oracle(target, command) != target:
                raise ValueError("MOVE idempotence failure")
            if command[0] == "SWAP" and world.oracle(target, command) != state:
                raise ValueError("SWAP involution failure")
            count += 1
    return {"state_roundtrips": 1024, "unique_inputs": len(seen),
            "oracle_independent_comparisons": count, "expected_inputs": 30720,
            "identity_feedback_has_no_command_argument": True, "passed": count == 30720}


def balance(training):
    receipt = {}
    for name in ("state", "target"):
        rows = [r[name] for r in training]
        counts = [[sum(s[o] == l for s in rows) for l in range(4)] for o in range(5)]
        fractions = [[value / len(rows) for value in group] for group in counts]
        receipt[name] = {"counts": counts, "fractions": fractions,
                         "within_registered_20_to_30_percent": all(.2 <= v <= .3 for group in fractions for v in group)}
    return receipt


def histograms(data):
    groups, commands = defaultdict(list), defaultdict(Counter)
    training = data.get("train", data.get("failure_evidence", {}).get("partial_train", []))
    for row in training:
        groups["TRAIN:input"].append(row["state"])
        groups["TRAIN:target"].append(row["target"])
        commands["TRAIN"][world.command_id(row["command"])] += 1
    for row in data.get("primitive_dev", []):
        key = f"DEV:{row['stratum']}:{row['command'][0]}:{'changed' if row['changed'] else 'noop'}"
        groups[key + ":input"].append(row["state"])
        groups[key + ":target"].append(row["target"])
        commands[key][world.command_id(row["command"])] += 1
    for row in data.get("sequences", data.get("failure_evidence", {}).get("sequences_partial", [])):
        key = f"SEQ:{row['stratum']}:{row['family']}"
        groups[key + ":input"].extend(row["oracle_path"][:-1])
        groups[key + ":target"].extend(row["oracle_path"][1:])
        commands[key].update(world.command_id(c) for c in row["commands"])
    return {"state_histograms": {key: {"n": len(rows), "distinct": len(set(rows)),
             "fullstate_counts": dict(sorted(Counter(world.state_id(s) for s in rows).items())),
             "per_object_location_counts": [[sum(s[o] == l for s in rows) for l in range(4)] for o in range(5)]}
             for key, rows in sorted(groups.items())},
            "command_counts_by_stratum_family": {key: dict(sorted(counts.items())) for key, counts in sorted(commands.items())}}


def assert_dataset(data):
    partitions, train = data["partitions"], data["train"]
    train_inputs = {r["state"] for r in train}
    train_targets = {r["target"] for r in train}
    train_partition = set(partitions["train_states"])
    dev_states = {r["state"] for r in data["primitive_dev"]}
    dev_states.update(r["target"] for r in data["primitive_dev"])
    dev_states.update(s for r in data["sequences"] for s in r["oracle_path"])
    intersections = {"train_input_vs_dev_pre_prefix": sorted(train_inputs & dev_states),
                     "train_target_vs_dev_pre_prefix": sorted(train_targets & dev_states),
                     "train_partition_vs_dev_pre_prefix": sorted(train_partition & dev_states)}
    if any(intersections.values()):
        raise ValueError("Strict TRAIN/DEV split leaked")
    if any(r["state"] not in train_partition or r["target"] not in train_partition for r in train):
        raise ValueError("TRAIN input/target outside TRAIN partition")
    for r in train + data["primitive_dev"]:
        if independent_oracle(r["state"], r["command"]) != r["target"]:
            raise ValueError("Saved transition label independent audit failed")
    twins = defaultdict(dict)
    for item in data["sequences"]:
        expected = [item["initial_state"]]
        for command in item["commands"]:
            expected.append(independent_oracle(expected[-1], command))
        if tuple(expected) != item["oracle_path"] or not world.path_valid(expected, train_partition):
            raise ValueError("Saved sequence independent audit failed")
        if item["family"] in ("AB", "BA"):
            twins[item["initial_index"]][item["family"]] = item
    for pair in twins.values():
        if set(pair) != {"AB", "BA"} or pair["AB"]["commands"] != pair["BA"]["commands"][::-1]:
            raise ValueError("AB/BA twin command mismatch")
        if pair["AB"]["oracle_path"][-1] == pair["BA"]["oracle_path"][-1]:
            raise ValueError("Order twin is commuting")
    return {"intersections": intersections, "order_twins": len(twins),
            "train_prestate_count": len(train_inputs), "train_target_state_count": len(train_targets),
            "true_dev_pre_prefix_state_count": len(dev_states), "all_prefixes_unseen_in_train": True}


def score_controls(data, check):
    controls = world.EpisodicControls(data["train"])
    primitive_raw, sequence_raw, summary, time_receipt = [], [], {}, {}
    for name in (*controls.names, "symbolic_oracle"):
        control_started = time.perf_counter()
        counts = defaultdict(lambda: [0, 0])
        for index, row in enumerate(data["primitive_dev"]):
            if index % 32 == 0:
                check()
            prediction = (independent_oracle(row["state"], row["command"]) if name == "symbolic_oracle"
                          else controls.predict(name, row["state"], row["command"]))
            key = f"{row['stratum']}:{row['command'][0]}:{'changed' if row['changed'] else 'noop'}"
            counts[key][0] += prediction == row["target"]
            counts[key][1] += 1
            primitive_raw.append({"control": name, "id": row["id"], "prediction": prediction})
        for index, row in enumerate(data["sequences"]):
            if index % 16 == 0:
                check()
            state, predicted, prefix_correct = row["initial_state"], [], True
            for step, command in enumerate(row["commands"], 1):
                state = (independent_oracle(state, command) if name == "symbolic_oracle"
                         else controls.predict(name, state, command))
                predicted.append(state)
                prefix_correct = prefix_correct and state == row["oracle_path"][step]
            for metric, correct in (("path", prefix_correct), ("final", state == row["oracle_path"][-1])):
                key = f"{row['stratum']}:{row['family']}:{metric}"
                counts[key][0] += correct
                counts[key][1] += 1
            sequence_raw.append({"control": name, "id": row["id"], "predicted_path": predicted})
        summary[name] = {key: {"correct": value[0], "n": value[1], "accuracy": value[0] / value[1]}
                         for key, value in sorted(counts.items())}
        time_receipt[name] = time.perf_counter() - control_started
    compact = bytes(v % 256 for row in data["train"] for v in
                    (*row["state"], *row["target"], int(row["command"][0] == "SWAP"), *row["command"][1:]))
    cost = {"episodic_records": len(data["train"]), "compact_record_bytes": len(compact),
            "compact_record_sha256": hashlib.sha256(compact).hexdigest(),
            "compact_layout": "uint8: pre5,target5,op1,first1,second1,destination1; unused255",
            "training_json_bytes": len(world.canonical_json(data["train"])),
            "query_counts": dict(controls.query_counts), "Hamming_distance_evaluations": dict(controls.distance_evaluations),
            "measured_control_wall_seconds_including_prediction_records": time_receipt,
            "optimized_neural_parameters": 0,
            "storage_limitation": "compact payload excludes Python objects/index/runtime; process memory reported separately",
            "whole_tuple_1NN_cannot_emit_unseen_TRAIN_target_tuple": True}
    return summary, {"primitive_predictions": primitive_raw, "sequence_predictions": sequence_raw}, cost


def adequacy(summary):
    cheap = ("identity", "last_destination_constant", "operator_only_majority",
             "argument_only_majority", "state_only_identity")
    cheap_keys = ("IID:MOVE:changed", "IID:SWAP:changed", "IID:AB:final", "IID:BA:final")
    cheap_maxima = {key: max(summary[name][key]["accuracy"] for name in cheap) for key in cheap_keys}
    cheap_pass = all(value < .5 for value in cheap_maxima.values())
    knn = summary["knn3"]
    memory_suffices = (knn["IID:MOVE:changed"]["accuracy"] >= .8
                       and knn["IID:SWAP:changed"]["accuracy"] >= .8
                       and knn["IID:AB:path"]["accuracy"] >= .6
                       and knn["IID:BA:path"]["accuracy"] >= .6)
    oracle_pass = all(row["accuracy"] == 1 for row in summary["symbolic_oracle"].values())
    failures = []
    if not cheap_pass:
        failures.append("cheap refusal exactstate reaches registered 50% ceiling: " + json.dumps(cheap_maxima, sort_keys=True))
    if memory_suffices:
        failures.append("kNN3 meets both primitive80% and AB/BA path60% registered sufficiency gates")
    if not oracle_pass:
        failures.append("symbolic oracle is not exact100% in every evaluated cell")
    return {"cheap_maxima": cheap_maxima, "cheap_refusal_gap_pass": cheap_pass,
            "knn3_memory_suffices": memory_suffices, "symbolic_oracle_all_exact": oracle_pass,
            "failure_reasons": failures,
            "status": "PASS_D0_ONLY" if cheap_pass and not memory_suffices and oracle_pass else "STOP_ADEQUACY"}


def save_report(run_dir, result):
    relative = run_dir.relative_to(ROOT).as_posix()
    lines = ["# RC operator D0 preflight — 16/11/2026", "", f"**{result['status']}**.", "",
             f"Source SHA `{result['source_sha']}`; [run](../../{relative}/result.json).",
             "D0 data-only: no neural/profile/fit, 0 model calls, 0 updates. No main/live/Hermes changes.", "",
             f"Reason: {result.get('reason', 'All registered D0 data/codec/adequacy checks passed.' if result['status'] == 'PASS_D0_ONLY' else 'See recorded status and gate receipts.')}", "",
             "```json", json.dumps({key: result.get(key) for key in
                                    ("stage", "train_receipt", "balance", "split_audit", "adequacy", "resources")},
                                   ensure_ascii=False, indent=2), "```", "",
             "Strict split/quotas/seed were not changed. D0 FAIL closes this family without rescue sweeps.",
             "D0 PASS requires another root source/dependency/backend review before any neural profiling/training.",
             "Audit level: same-team independent tuple implementation and saved-data recomputation, not external peer review.",
             "H128 representability, hidden contribution, H3 generalization, H5 and continual retention remain NOT_MEASURED.", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    global ACTIVE_BUDGET
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-d0", action="store_true", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    args = parser.parse_args()
    ACTIVE_BUDGET = Budget()
    source_sha = git("rev-parse", "HEAD")
    if source_sha != args.expected_source_sha:
        raise SystemExit("Source SHA does not match root-reviewed commit; no D0 started")
    for path in SOURCE_PATHS:
        git("ls-files", "--error-unmatch", path)
    if git("diff", "HEAD", "--", *SOURCE_PATHS):
        raise SystemExit("Reviewed source/protocol modified since commit; no D0 started")
    protocol = (ROOT / SOURCE_PATHS[2]).read_text(encoding="utf-8")
    if "ROOT_REVIEWED_D0_ONLY" not in protocol:
        raise SystemExit("D0-only review status missing; no D0 started")
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-rc-operator-pilot-v1-" + secrets.token_hex(3)
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(exist_ok=False)
    result = {"status": "IN_PROGRESS", "stage": "resource_start", "source_sha": source_sha,
              "run_id": run_name, "model_calls": 0, "learning_updates": 0, "neural_fits": 0,
              "data_seed": world.DATA_SEED, "protocol": SOURCE_PATHS[2]}
    data, raw_controls, budget = {}, {}, ACTIVE_BUDGET
    try:
        manifest = {**result, "stage": "SOURCE_LOCKED_BEFORE_DATA", "python": sys.version,
                    "numpy": world.np.__version__, "platform": platform.platform(), "pid": os.getpid(),
                    "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS},
                    "ponytail": "unavailable in enabled inventory", "backend": "CPU_DATA_ONLY",
                    "caps": {"seconds": 20, "RSS_bytes": 300_000_000, "CPU": 1}}
        write_json(run_dir / "manifest.json", manifest)
        for path in SOURCE_PATHS:
            target = run_dir / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / path).read_bytes())
        result["stage"] = "exhaustive_codec_oracle_contract"
        result["contracts"] = contracts(budget.check)
        result["stage"] = "partition"
        data["partitions"] = world.make_partitions(budget.check)
        result["stage"] = "greedy_train_selection"
        data["train"], result["train_receipt"] = world.make_train(data["partitions"], budget.check)
        result["stage"] = "train_balance"
        result["balance"] = balance(data["train"])
        if not all(row["within_registered_20_to_30_percent"] for row in result["balance"].values()):
            raise world.DataStop("TRAIN input/target balance outside registered [20%,30%]")
        result["stage"] = "primitive_dev"
        data["primitive_dev"] = world.make_primitive_dev(data["partitions"], budget.check)
        result["stage"] = "heldout_sequences"
        data["sequences"], result["sequence_attempts"] = world.make_sequences(data["partitions"], budget.check)
        result["stage"] = "strict_split_and_independent_label_audit"
        result["split_audit"] = assert_dataset(data)
        result["stage"] = "cheap_control_evaluation"
        result["control_summary"], raw_controls, result["memory_control_costs"] = score_controls(data, budget.check)
        result["adequacy"] = adequacy(result["control_summary"])
        result["status"] = result["adequacy"]["status"]
        if result["adequacy"]["failure_reasons"]:
            result["reason"] = "; ".join(result["adequacy"]["failure_reasons"])
        result["stage"] = "D0_COMPLETE"
    except world.DataStop as error:
        result["status"], result["reason"] = "STOP_DATA", str(error)
        if error.evidence:
            data["failure_evidence"] = error.evidence
            if "train_receipt" in error.evidence:
                result["train_receipt"] = error.evidence["train_receipt"]
    except ResourceStop as error:
        result["status"], result["reason"] = "STOP_RESOURCE", str(error)
    except Exception as error:
        result["status"], result["reason"] = "INVALID", f"{type(error).__name__}: {error}"
    finally:
        result["histograms"] = histograms(data)
        result["raw_data_sha256"] = write_json(run_dir / "raw-data.json", data)
        result["control_predictions_sha256"] = write_json(run_dir / "control-predictions.json", raw_controls)
        result["dataset_component_hashes"] = {key: world.digest(value) for key, value in data.items()}
        if budget is not None:
            try:
                budget.check()
            except ResourceStop as error:
                result["status"], result["reason"] = "STOP_RESOURCE", str(error)
            result["resources"] = budget.last
        result["disk_bytes_before_final_report"] = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
        save_report(run_dir, result)
        result["elapsed_wall_through_report_seconds"] = time.perf_counter() - START_WALL
        result["cpu_seconds_through_report"] = time.process_time() - START_CPU
        if result["elapsed_wall_through_report_seconds"] > 20:
            result["status"], result["reason"] = "STOP_RESOURCE", "D0 including report exceeded 20s"
            save_report(run_dir, result)
        write_json(run_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "stage": result["stage"],
                      "run_id": run_name, "reason": result.get("reason"),
                      "resources": result.get("resources"), "model_calls": 0, "updates": 0}, ensure_ascii=False))
    return 0 if result["status"] == "PASS_D0_ONLY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
