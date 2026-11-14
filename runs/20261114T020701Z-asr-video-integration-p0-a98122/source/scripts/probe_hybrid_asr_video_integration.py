"""Real private VideoMemory/ALIF/QA with explicit frozen ASR candidate identity."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from hybrid_shared_endurance import SharedSensory
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.config import HybridConfig
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.media import VideoMemory
from scarx.hybrid.resources import ResourceMonitor, windows_tree_commit_limit
from scarx.hybrid.device_budget import gpu_headroom
from scarx.hybrid.store import Store, encode


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(2 << 20):
            h.update(block)
    return h.hexdigest()


class ExplicitASR(SharedSensory):
    def __init__(self, native, directory, store, monitor, config, model_path):
        super().__init__(native, directory, store, monitor, config)
        self.model_path = model_path
        self.directory, self.store = directory, store
        self.owned_asr = None
        self.records = []

    def transcribe(self, wav):
        self.native.check()
        if gpu_headroom()["free_MiB"] < 1800:
            raise MemoryError("Candidate ASR headroom preflight failed")
        source = ROOT / "local/hybrid-v1"
        output = self.directory / ("asr-" + uuid.uuid4().hex)
        log_path = output.with_suffix(".log")
        command = [str(source / "runtime/whisper.cpp-b5454-cuda12.4/Release/whisper-cli.exe"),
                   "-m", str(self.model_path), "-f", str(wav), "-l", "auto",
                   "-t", "2", "-dev", "0", "-ojf", "-of", str(output)]
        started = time.monotonic()
        with log_path.open("wb") as log:
            self.owned_asr = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                while self.owned_asr.poll() is None:
                    if gpu_headroom()["free_MiB"] < 512:
                        raise MemoryError("Observed physical VRAM reserve breached")
                    if time.monotonic() - started > 120:
                        raise TimeoutError("Owned ASR did not finish")
                    time.sleep(.1)
                if self.owned_asr.returncode:
                    raise RuntimeError("Owned candidate ASR failed; inspect " + log_path.name)
            finally:
                if self.owned_asr.poll() is None:
                    self.owned_asr.terminate()
                    self.owned_asr.wait(10)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "using CUDA0 backend" in text and "ftype         = 0" in text
        result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        transcript = " ".join(r.get("text", "").strip() for r in result["transcription"]).strip()
        record = {"text": transcript, "seconds": time.monotonic() - started,
                  "model_path": str(self.model_path), "CUDA_FP32_confirmed": True,
                  "raw_result_path": str(output.with_suffix(".json"))}
        self.records.append(record)
        self.store.event("asr_result", record, "asr")
        return transcript


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-asr-video-integration-p0-" + uuid.uuid4().hex[:6]
    run, data = ROOT / "runs" / name, ROOT / "local" / name
    run.mkdir()
    data.mkdir()
    source = ROOT / "local/hybrid-v1"
    clip = source / "samples/engineering-vi-video.mkv"
    questions = [
        {"id": "speech_first_rank", "question": "Theo lời nói trong video, Arly Velasquez xếp thứ mấy? Chỉ trả lời số thứ hạng.", "expected": "15"},
        {"id": "speech_second_rank", "question": "Theo lời nói trong video, Adam Hall xếp thứ mấy? Chỉ trả lời số thứ hạng.", "expected": "9"},
        {"id": "visual_last_color", "question": "Hình tròn bên phải ở cuối video có màu gì? Chỉ trả lời một từ tiếng Anh.", "expected": "yellow"},
    ]
    (run / "locked-questions.json").write_text(encode(questions), encoding="utf-8")
    paths = list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        Path(__file__), ROOT / "scripts/hybrid_shared_endurance.py",
        ROOT / "src/scarx/evaluation/shared_native.py",
        ROOT / "experiments/hybrid-asr-video-integration-p0-v1.md",
    ]
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        out = run / "source" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(path.read_bytes())
        hashes[rel.as_posix()] = digest(path)
    models = {"base": source / "models/native/whisper-base-f32.bin",
              "small": source / "candidates/whisper-small/whisper-small-f32.bin"}
    benchmark_manifest = json.loads((ROOT / "runs/20261114T015546Z-asr-candidate-benchmark-v1-68e3c7/manifest.json").read_text())
    for arm, path in models.items():
        assert digest(path) == benchmark_manifest["model_sha256"][arm]
    (run / "manifest.json").write_text(encode({
        "source_sha256": hashes, "clip_sha256": digest(clip),
        "model_sha256": benchmark_manifest["model_sha256"],
        "known_engineering_clip_not_newTEST": True,
        "learning": False, "live_ASR_changed": False,
    }), encoding="utf-8")
    trace = Store(source)
    native = SharedNative(ROOT)
    monitor = ResourceMonitor(limit=None, extra_pids={native.owner.pid})
    report = {"run_id": name, "arms": {}, "failures": [],
              "SNN_hidden_advantage_claim": False, "live_ASR_changed": False}
    started = time.monotonic()
    try:
        report["job"] = windows_tree_commit_limit(None)
        monitor.start()
        for arm in ("base", "small"):
            directory = data / arm
            directory.mkdir()
            store = Store(directory)
            revision = ("e37978b90ca9030d5170a5c07aadb050351a65bb" if arm == "base"
                        else "973afd24965f72e36ca33b3055d56a652f456b4d")
            config = HybridConfig(execution_mode="gpu", asr_revision=revision)
            backend = ExplicitASR(native, directory, store, monitor, config, models[arm])
            engine = HybridEngine(store, backend, monitor, config)
            engine.model.to_device("gpu")
            engine.state["learning_enabled"] = False
            video = VideoMemory(engine)
            arm_start = time.monotonic()
            try:
                initial = {k:hashlib.sha256(engine.model.host(v).tobytes()).hexdigest()
                           for k,v in engine.model.arrays().items() if "w_" in k}
                observed = video.observe(clip)
                deadline = time.monotonic() + 120
                while True:
                    state = store.get("stream", observed["id"])
                    if state["status"] in ("complete", "failed", "stopped"):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("Actual video integration did not finish")
                    time.sleep(.2)
                assert state["status"] == "complete", state.get("error")
                assert not state["asr_errors"] and not state["skipped_intervals"]
                answers = []
                for question in questions:
                    answer = engine.ask(question["question"], stream=state["id"])
                    scored = answer["answer"].strip().casefold() == question["expected"]
                    row = {**question, "actual": answer, "strict_contract_passed": scored}
                    answers.append(row)
                    trace.event("ASR_candidate.integrated_video_answer", {
                        "run_id": name, "ASR_model": arm, **row,
                        "known_clip_not_formal_videoP1": True}, "orchestrator")
                assert engine.model.updates == engine.model.hidden_updates == 0
                assert initial == {k:hashlib.sha256(engine.model.host(v).tobytes()).hexdigest()
                                   for k,v in engine.model.arrays().items() if "w_" in k}
                pin = engine.save(pinned=True)
                raw = store.snapshot(pin)
                engine.load(pin)
                assert engine.model.dumps(engine.state) == raw
                report["arms"][arm] = {
                    "ASR_identity": {"repo": "openai/whisper-" + arm, "revision": revision},
                    "stream": state, "ASR_records": backend.records, "answers": answers,
                    "score": sum(r["strict_contract_passed"] for r in answers),
                    "local_updates": 0, "hidden_updates": 0,
                    "private_save_load_exact": True, "seconds": time.monotonic() - arm_start}
            finally:
                video.close()
                engine.close()
                backend.close()
                store.close()
        for rel, digest_ in hashes.items():
            assert digest(ROOT / rel) == digest_
        assert time.monotonic() - started < 300
    except Exception:
        report["failures"].append(traceback.format_exc())
    finally:
        report["resources"] = monitor.close()
        report["seconds"] = time.monotonic() - started
        report["verdict"] = "KNOWN_CLIP_INTEGRATION_ONLY_NO_PROMOTION" if not report["failures"] else "INTEGRATION_P0_FAILED"
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("ASR_candidate.integration_completed", {
            k:v for k,v in report.items() if k != "arms"}, "orchestrator")
        trace.close()
        print(encode({"run_id": name, "scores": {a:r["score"] for a,r in report["arms"].items()},
            "verdict": report["verdict"], "failures": report["failures"],
            "seconds": report["seconds"]}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
