"""Paired frozen native ASR on locked speech; no live model or learning edits."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.asr_candidate_metrics import word_error, paired_language
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.device_budget import gpu_headroom
from scarx.hybrid.resources import ResourceMonitor, windows_tree_commit_limit
from scarx.hybrid.store import Store, encode


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(2 << 20):
            h.update(block)
    return h.hexdigest()


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-asr-candidate-benchmark-v1-" + uuid.uuid4().hex[:6]
    run, private = ROOT / "runs" / name, ROOT / "local" / name
    run.mkdir()
    private.mkdir()
    source = ROOT / "local/hybrid-v1"
    data_run = Path((ROOT / "local/current-asr-dev-data-run.txt").read_text())
    data_file = data_run / "manifest.json"
    data = json.loads(data_file.read_text(encoding="utf-8"))
    candidate = source / "candidates/whisper-small"
    receipt = json.loads((candidate / "native-storage-receipt.json").read_text())
    if data["failures"] or len(data["samples"]) != 32 or receipt["quality_qualified"]:
        raise ValueError("Expected locked32sampleDEV and unpromoted candidate")
    sources = [
        Path(__file__), ROOT / "src/scarx/evaluation/asr_candidate_metrics.py",
        ROOT / "src/scarx/evaluation/shared_native.py",
        ROOT / "experiments/hybrid-whisper-small-native-v1.md",
    ]
    hashes = {}
    for path in sources:
        raw, rel = path.read_bytes(), path.relative_to(ROOT)
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[rel.as_posix()] = hashlib.sha256(raw).hexdigest()
    models = {"base": source / "models/native/whisper-base-f32.bin",
              "small": candidate / "whisper-small-f32.bin"}
    assert sha(models["small"]) == receipt["export"]["sha256"]
    integrity = json.loads((source / "native-integrity.json").read_text())
    assert sha(models["base"]) == integrity["exports"]["whisper-base-f32.bin"]["sha256"]
    exe = source / "runtime/whisper.cpp-b5454-cuda12.4/Release/whisper-cli.exe"
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "data_run": data_run.name, "data_sha256": sha(data_file),
        "model_sha256": {k:sha(v) for k,v in models.items()},
        "model_paths": {k:str(v) for k,v in models.items()},
        "executable_sha256": sha(exe), "sample_count": 32,
        "model_order": "base,small for even index; small,base for odd index",
        "native_VLM_or_live_ALIF_modified": False,
    }), encoding="utf-8")
    trace = Store(source)
    native = SharedNative(ROOT)
    monitor = ResourceMonitor(limit=None, extra_pids={native.owner.pid})
    report = {"run_id": name, "pairs": [], "failures": [],
              "live_ASR_changed": False, "candidate_promoted": False,
              "original_precision": "FP32", "native_device": "CUDA0"}
    started = time.monotonic()
    process = None
    sampling_stop = threading.Event()
    samples = []

    def sample_gpu():
        while not sampling_stop.wait(.25):
            try:
                samples.append({"epoch": time.time(), **gpu_headroom()})
            except Exception as exc:
                samples.append({"epoch": time.time(), "sampling_error": str(exc)})

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    try:
        report["job"] = windows_tree_commit_limit(None)
        monitor.start()
        baseline = monitor.sample()
        report["before_resources"] = baseline
        assert gpu_headroom()["free_MiB"] >= 1800, "Insufficient freeVRAM for candidate preflight"
        sampler.start()
        trace.event("ASR_candidate.benchmark_started",
            {"run_id": name, "sample_pairs": 32, "native_device": "CUDA0"}, "orchestrator")
        for index, sample in enumerate(data["samples"]):
            audio = Path(sample["audio"])
            assert sha(audio) == sample["audio_sha256"]
            pair = {"id": sample["id"], "language": sample["language"],
                    "reference": sample["reference"], "duration_seconds": sample["duration_seconds"]}
            for arm in (("base", "small") if index % 2 == 0 else ("small", "base")):
                if time.monotonic() - started > 1200:
                    raise TimeoutError("Registered20minuteASR budget exhausted")
                native.check()
                deadline = time.monotonic() + 30
                while native.status().get("cognition", {}).get("phase") == "serving":
                    if time.monotonic() > deadline:
                        raise TimeoutError("User work retains priority; no new candidate request")
                    time.sleep(.2)
                if gpu_headroom()["free_MiB"] < 1800:
                    raise MemoryError("Not enough headroom before owned ASR launch")
                output = private / (sample["id"] + "-" + arm)
                logpath = run / (sample["id"] + "-" + arm + ".log")
                command = [str(exe), "-m", str(models[arm]), "-f", str(audio),
                           "-l", "auto", "-t", "2", "-dev", "0", "-ojf", "-of", str(output)]
                call_start = time.monotonic()
                with logpath.open("wb") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    while process.poll() is None:
                        recent = samples[-1] if samples else None
                        if recent and recent.get("free_MiB", 99999) < 512:
                            raise MemoryError("Observed physical reserve below512MiB")
                        if time.monotonic() - call_start > 120:
                            raise TimeoutError("ASR call exceeded120seconds")
                        time.sleep(.1)
                    if process.returncode:
                        raise RuntimeError("ASR subprocess failed; see " + logpath.name)
                elapsed = time.monotonic() - call_start
                process = None
                logtext = logpath.read_text(encoding="utf-8", errors="replace")
                if "using CUDA0 backend" not in logtext or "ftype         = 0" not in logtext:
                    raise ValueError("Missing actualCUDA/originalFP32 evidence")
                result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
                transcript = " ".join(row.get("text", "").strip()
                                      for row in result.get("transcription", [])).strip()
                (run / (sample["id"] + "-" + arm + ".json")).write_text(encode(result), encoding="utf-8")
                pair[arm] = {
                    **word_error(transcript, sample["reference"]), "transcript": transcript,
                    "seconds": elapsed, "RTF": elapsed / sample["duration_seconds"],
                    "detected_language": result.get("result", {}).get("language"),
                    "native_CUDA_FP32_confirmed": True,
                }
                trace.event("ASR_candidate.measured_transcript", {
                    "run_id": name, "id": sample["id"], "language": sample["language"],
                    "reference": sample["reference"], "model": arm, **pair[arm],
                    "self_learning_label": False}, "orchestrator")
                monitor.sample()
            report["pairs"].append(pair)
        report["by_language"] = {lang: paired_language([
            r for r in report["pairs"] if r["language"] == lang]) for lang in ("vi", "en")}
        timings = {arm: {"median_seconds": float(np.median([r[arm]["seconds"] for r in report["pairs"]])),
                        "p95_RTF": float(np.percentile([r[arm]["RTF"] for r in report["pairs"]], 95))}
                   for arm in models}
        report["timing"] = timings
        vi, en = report["by_language"]["vi"], report["by_language"]["en"]
        report["locked_quality_gate"] = (
            vi["relative_error_reduction"] >= .10
            and vi["paired_utterance_delta_CI95"][1] < 0
            and en["delta_small_minus_base"] <= .02)
        report["locked_speed_gate"] = (
            timings["small"]["p95_RTF"] <= 1
            and timings["small"]["median_seconds"] <= 2 * timings["base"]["median_seconds"])
        for rel, expected in hashes.items():
            assert sha(ROOT / rel) == expected
    except Exception:
        report["failures"].append(traceback.format_exc())
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        sampling_stop.set()
        if sampler.is_alive():
            sampler.join(5)
        report["resources"] = monitor.close()
        report["GPU_sampling_errors"] = [r for r in samples if "sampling_error" in r]
        report["sampled_min_free_MiB"] = min([r["free_MiB"] for r in samples if "free_MiB" in r] or [0])
        report["shared_GPU_growth_bytes"] = monitor.last["gpu_shared"]["bytes"] - report.get(
            "before_resources", {}).get("gpu_shared", {}).get("bytes", 0)
        report["locked_resource_gate"] = None
        report["resource_status"] = (
            "Per-chunk quality timing run does not prove sampled peak sharedGPU. "
            "Require separate active-residency resource probe and independent audit.")
        report["qualified_DEV_candidate"] = False
        report["verdict"] = (
            "QUALITY_SPEED_DEV_ONLY_RESOURCE_PROFILE_PENDING"
            if not report["failures"] and report.get("locked_quality_gate")
            and report.get("locked_speed_gate") and len(report["pairs"]) == 32
            else "RETAIN_BASE_CANDIDATE_NOT_QUALIFIED")
        report["seconds"] = time.monotonic() - started
        (run / "GPU-samples.json").write_text(encode(samples), encoding="utf-8")
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("ASR_candidate.benchmark_completed",
            {k:v for k,v in report.items() if k != "pairs"}, "orchestrator")
        trace.close()
        print(encode({k:report.get(k) for k in (
            "run_id", "verdict", "by_language", "timing", "sampled_min_free_MiB",
            "failures", "seconds")}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
