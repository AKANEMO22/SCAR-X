"""Sustained owned ASR residency, accurate active sampling; never changes live."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.asr_candidate_resources import qualify_active_samples
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.device_budget import gpu_headroom
from scarx.hybrid.resources import ResourceMonitor, windows_tree_commit_limit
from scarx.hybrid.store import Store, encode


def filehash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while part := f.read(2 << 20):
            h.update(part)
    return h.hexdigest()


def main():
    run = ROOT / "runs" / (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-asr-resource-followup-v1-" + uuid.uuid4().hex[:6])
    run.mkdir()
    source = ROOT / "local/hybrid-v1"
    parent = ROOT / "runs/20261114T015546Z-asr-candidate-benchmark-v1-68e3c7"
    manifest = json.loads((parent / "manifest.json").read_text())
    data_path = ROOT / "runs" / manifest["data_run"] / "manifest.json"
    assert filehash(data_path) == manifest["data_sha256"]
    data = json.loads(data_path.read_text(encoding="utf-8"))
    models = {k:Path(v) for k,v in manifest["model_paths"].items()}
    for arm, path in models.items():
        assert filehash(path) == manifest["model_sha256"][arm]
    exe = source / "runtime/whisper.cpp-b5454-cuda12.4/Release/whisper-cli.exe"
    assert filehash(exe) == manifest["executable_sha256"]
    for sample in data["samples"]:
        assert filehash(Path(sample["audio"])) == sample["audio_sha256"]
    hashes = {}
    for path in (Path(__file__), ROOT / "src/scarx/evaluation/asr_candidate_resources.py",
                 ROOT / "experiments/hybrid-whisper-small-resource-followup-v1.md"):
        rel = path.relative_to(ROOT)
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[rel.as_posix()] = filehash(path)
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "parent_benchmark": parent.name,
        "parent_manifest_sha256": filehash(parent / "manifest.json"),
        "new_quality_samples": False, "modified_live_model": False,
        "input_files_per_arm": 32, "sampling_seconds": .25,
    }), encoding="utf-8")
    trace = Store(source)
    native = SharedNative(ROOT)
    monitor = ResourceMonitor(limit=None, extra_pids={native.owner.pid})
    process = None
    report = {"run_id": run.name, "arms": {}, "failures": [],
              "live_ASR_changed": False, "new_independent_quality_samples": 0}
    started = time.monotonic()
    try:
        report["job"] = windows_tree_commit_limit(None)
        monitor.start()
        for arm in ("base", "small"):
            native.check()
            assert gpu_headroom()["free_MiB"] >= 1800
            baseline = monitor.sample()
            assert baseline["gpu_shared"]["available"] and baseline["gpu_dedicated"]["available"]
            command = [str(exe), "-m", str(models[arm]), "-l", "auto",
                       "-t", "2", "-dev", "0"] + [s["audio"] for s in data["samples"]]
            samples = []
            logpath = run / (arm + "-batch.log")
            arm_start = time.monotonic()
            with logpath.open("wb") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                while process.poll() is None:
                    native.check()
                    resources = monitor.sample()
                    row = {"epoch": time.time(), **gpu_headroom(),
                        "owned_ASR_active": process.poll() is None,
                        "shared_bytes": resources["gpu_shared"]["bytes"],
                        "dedicated_bytes": resources["gpu_dedicated"]["bytes"],
                        "shared_available": resources["gpu_shared"]["available"],
                        "dedicated_available": resources["gpu_dedicated"]["available"],
                        "PDH_collection_monotonic": monitor.gpu.last,
                        "owned_ASR_pid": process.pid}
                    samples.append(row)
                    if row["free_MiB"] < 512:
                        raise MemoryError("Physical VRAM reserve breached")
                    if time.monotonic() - started > 300:
                        raise TimeoutError("Registered resource budget exceeded")
                    time.sleep(.25)
                if process.returncode:
                    raise RuntimeError("Batch ASR failed; see " + logpath.name)
            process = None
            text = logpath.read_text(encoding="utf-8", errors="replace")
            assert text.count("main: processing '") == 32, "Did not execute all32 audio inputs"
            assert "using CUDA0 backend" in text and "ftype         = 0" in text
            metrics = qualify_active_samples(samples, baseline["gpu_shared"]["bytes"])
            metrics["seconds"] = time.monotonic() - arm_start
            metrics["input_files_processed"] = 32
            metrics["raw_samples"] = samples
            report["arms"][arm] = metrics
            trace.event("ASR_candidate.resource_probe_arm", {
                "run_id": run.name, "model": arm,
                **{k:v for k,v in metrics.items() if k != "raw_samples"}}, "orchestrator")
        report["resource_followup_pass"] = all(a["resource_gate_pass"] for a in report["arms"].values())
        for rel, expected in hashes.items():
            assert filehash(ROOT / rel) == expected
    except Exception:
        report["failures"].append(traceback.format_exc())
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait(10)
        report["resources_at_close"] = monitor.close()
        report["seconds"] = time.monotonic() - started
        report["verdict"] = "ACTIVE_RESOURCE_PROFILE_PASS_CANDIDATE_ONLY" if (
            not report["failures"] and report.get("resource_followup_pass")) else "RESOURCE_PROFILE_NOT_QUALIFIED"
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("ASR_candidate.resource_probe_completed", {
            k:v for k,v in report.items() if k != "arms"}, "orchestrator")
        trace.close()
        print(encode({"run_id": run.name, "verdict": report["verdict"],
            "arms": {a:{k:v for k,v in m.items() if k != "raw_samples"}
                     for a,m in report["arms"].items()},
            "failures": report["failures"], "seconds": report["seconds"]}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
