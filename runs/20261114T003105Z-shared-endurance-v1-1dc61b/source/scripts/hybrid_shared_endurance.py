"""Private real mixed pipeline, sharing one live frozen VLM; no live weight edits."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from types import SimpleNamespace
import uuid

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.backends import NativeBackends
from scarx.hybrid.config import HybridConfig
from scarx.hybrid.device_budget import gpu_headroom, audio_requires_release
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.media import VideoMemory
from scarx.hybrid.resources import ResourceMonitor, windows_tree_commit_limit
from scarx.hybrid.store import Store, encode


class SharedSensory:
    def __init__(self, native, data, store, monitor, config):
        self.native = native
        self.audio = NativeBackends(data, store, monitor, config)
        self.cancel_idle = threading.Event()
        self.vlm = SimpleNamespace(poll=self.poll)

    def poll(self):
        try:
            self.native.check()
            return None
        except (RuntimeError, OSError):
            return 1

    def wake(self):
        self.native.check()

    def generate(self, messages, max_tokens=128, background=False, response_format=None,
                 **kwargs):
        if background and (self.cancel_idle.is_set() or
                self.native.status().get("cognition", {}).get("phase") == "serving"):
            raise InterruptedError("Production/foreground work has priority")
        body = {"messages": messages, "max_tokens": min(max_tokens, 128)}
        if response_format is not None:
            body["response_format"] = response_format
        raw = self.native.generate(body)
        return {"text": raw["choices"][0]["message"].get("content") or "",
                "usage": raw.get("usage", {}), "raw": raw}

    def describe(self, image, background=False):
        value = base64.b64encode(Path(image).read_bytes()).decode()
        return self.generate([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + value}},
            {"type": "text", "text": "Describe visible shapes, colors and text briefly."},
        ]}], background=background)

    def transcribe(self, wav):
        if self.cancel_idle.is_set() or self.native.status().get("cognition", {}).get("phase") == "serving":
            raise InterruptedError("Production/foreground audio has priority")
        if audio_requires_release(gpu_headroom()):
            raise InterruptedError("Not enough shared GPU headroom; never unload live VLM")
        return self.audio.transcribe(wav)

    def close(self):
        self.audio.close()  # Own ASR only; no reference to live VLM in this object.


def main(seconds):
    if seconds not in (90, 28800):
        raise ValueError("Registered smoke90s or endurance28800s only")
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-shared-endurance-v1-" + uuid.uuid4().hex[:6]
    run, data = ROOT / "runs" / name, ROOT / "local" / name
    run.mkdir()
    data.mkdir()
    source = ROOT / "local/hybrid-v1"
    for folder in ("runtime", "models/native"):
        shutil.copytree(source / folder, data / folder, copy_function=os.link)
    sample = source / "samples/engineering-vi-video.mkv"
    paths = list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        ROOT / "src/scarx/neural/core.py", ROOT / "src/scarx/evaluation/readout_eligibility.py",
        ROOT / "src/scarx/evaluation/shared_native.py", ROOT / "src/scarx/learning/verifiers.py",
        ROOT / "experiments/hybrid-shared-native-endurance-v1.md", Path(__file__),
    ]
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        raw = path.read_bytes()
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[rel.as_posix()] = hashlib.sha256(raw).hexdigest()
    manifest = {
        "run_id": name, "pid": os.getpid(), "duration_seconds": seconds,
        "source_sha256": hashes, "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "private_data": str(data), "video_interval": 60 if seconds == 90 else 300,
        "stage_interval": 15 if seconds == 90 else 1440,
        "checkpoint_interval": 30 if seconds == 90 else 1800,
        "scope": "Private engineering fixture endurance; no formal learning/quality claim",
    }
    (run / "manifest.json").write_text(encode(manifest), encoding="utf-8")
    (ROOT / "local/current-shared-endurance-run.txt").write_text(str(run))
    trace = Store(source)
    private = Store(data)
    report = {"run_id": name, "duration_requested": seconds, "failures": [],
              "video_cycles": [], "checkpoints": [], "learning_goals": [],
              "live_model_replaced_or_restarted": False}
    engine = video = backend = monitor = None
    started = time.monotonic()
    print(encode({"run_id": name, "pid": os.getpid(), "duration_seconds": seconds}), flush=True)
    try:
        native = SharedNative(ROOT)
        status = native.status()
        monitor = ResourceMonitor(limit=None, extra_pids={
            native.owner.pid, *status["resources"].get("included_client_pids", [])})
        report["lifecycle_job"] = windows_tree_commit_limit(None)
        monitor.start()
        config = HybridConfig(execution_mode="gpu", searxng_enabled=False)
        backend = SharedSensory(native, data, private, monitor, config)
        engine = HybridEngine(private, backend, monitor, config)
        with private.lock, private.db:
            private.db.execute("DELETE FROM objects WHERE kind='public_topic'")
        engine.model.to_device("gpu")
        video = VideoMemory(engine)
        engine.capture_active = lambda: any(t.is_alive() for t in video.threads)
        engine.start()
        report["native_owner_pid"] = native.owner.pid
        report["native_worker_pid"] = native.worker.pid
        report["live_updates_at_start"] = status["neural"]["updates"]
        trace.event("shared_endurance.started", {**manifest,
            "native_worker_pid": native.worker.pid,
            "live_neural_weights_untouched": True}, "orchestrator")
        active = None
        next_video = next_stage = next_checkpoint = next_heartbeat = 0.0
        cursor, stage = 0, 0
        while time.monotonic() - started < seconds:
            elapsed = time.monotonic() - started
            native.check()
            if active:
                state = private.get("stream", active)
                if state["status"] in ("complete", "failed", "stopped"):
                    if state["status"] != "complete":
                        raise RuntimeError("Video cycle failed: " + str(state.get("error")))
                    answer = engine.ask("What color and shape appear last in this video?", stream=active)
                    record = {"stream": state, "actual_answer": answer,
                              "answer_quality_scored": False}
                    report["video_cycles"].append(record)
                    trace.event("shared_endurance.measured_video", {
                        "run_id": name, **record, "lab_not_live_neural_updates": True}, "orchestrator")
                    active = None
            if elapsed >= next_video and active is None:
                active = video.observe(sample)["id"]
                next_video = elapsed + manifest["video_interval"]
            if elapsed >= next_stage and stage < 20:
                a, b = 12 + stage, 3 + stage
                goal = engine.goal(f"{a}+{b}", reason="Private endurance arithmetic verifier transaction", origin="user")
                report["learning_goals"].append(goal["id"])
                stage += 1
                next_stage = elapsed + manifest["stage_interval"]
            if elapsed >= next_checkpoint and active is None:
                with engine.lock:
                    enabled = engine.state["learning_enabled"]
                    engine.state["learning_enabled"] = False
                until = time.monotonic() + 20
                while engine.cognition.work_active.is_set():
                    if time.monotonic() > until:
                        raise TimeoutError("Private quantum did not quiesce")
                    time.sleep(.1)
                with engine.lock:
                    pin = engine.save(pinned=True)
                    payload = private.snapshot(pin)
                    clone, controller = LocalALIF.loads(payload, device="gpu")
                    assert clone.dumps(controller) == payload
                    engine.load(pin)
                    assert engine.model.dumps(engine.state) == payload
                    del clone
                    engine.state["learning_enabled"] = enabled
                    engine.save()
                report["checkpoints"].append({"elapsed_seconds": elapsed, "id": pin,
                    "bytes": len(payload), "cold_GPU_clone_exact": True,
                    "full_private_rollback_exact": True})
                next_checkpoint = elapsed + manifest["checkpoint_interval"]
            if elapsed >= next_heartbeat:
                resource = monitor.sample()
                headroom = gpu_headroom()
                if headroom["free_MiB"] < 512:
                    raise MemoryError("Observed physical VRAM reserve below512MiB")
                with engine.lock:
                    assert len(engine.model.replay) <= 64 and len(engine.state["probes"]) <= 16
                    assert len(engine.state["goals"]) <= 64 and len(engine.state["review"]["pending"]) <= 32
                    neural = {"updates": engine.model.updates,
                              "hidden_updates": engine.model.hidden_updates,
                              "device": engine.model.device, "replay_items": len(engine.model.replay)}
                for rel, digest in hashes.items():
                    if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != digest:
                        raise RuntimeError("Frozen endurance source changed: " + rel)
                events = private.events(after=cursor, limit=200)
                for event in events:
                    cursor = event["id"]
                    if event["kind"] in ("local_learning_update", "cognitive_quantum_failed",
                                         "video_asr_failed", "autonomous_action"):
                        trace.event("shared_endurance.actual_lab_event",
                            {"run_id": name, "lab_actor": event["actor"], **event,
                             "is_live_neural_counter": False}, "orchestrator")
                beat = {"run_id": name, "pid": os.getpid(), "elapsed_seconds": elapsed,
                        "video_cycles": len(report["video_cycles"]),
                        "checkpoints": len(report["checkpoints"]), "neural": neural,
                        "cognitive_phase": engine.cognition.phase,
                        "active_video": active, "resources": resource,
                        "GPU_headroom": headroom, "state": "running"}
                (run / "heartbeat.json").write_text(encode(beat), encoding="utf-8")
                trace.event("shared_endurance.heartbeat", beat, "orchestrator")
                print(encode({k: beat[k] for k in (
                    "run_id", "elapsed_seconds", "video_cycles", "checkpoints", "neural")}), flush=True)
                next_heartbeat = elapsed + 30
            time.sleep(.25)
        report["elapsed_seconds"] = time.monotonic() - started
        report["final_neural"] = {"updates": engine.model.updates,
            "hidden_updates": engine.model.hidden_updates, "device": engine.model.device}
        report["gate_pass"] = (report["elapsed_seconds"] >= seconds
            and len(report["video_cycles"]) >= (1 if seconds == 90 else 80)
            and len(report["checkpoints"]) >= (1 if seconds == 90 else 8)
            and engine.model.updates > 0 and engine.model.device == "gpu")
        report["verdict"] = ("SMOKE_ENGINEERING_PASS" if seconds == 90 else "EIGHT_HOUR_ENGINEERING_PASS") if report["gate_pass"] else "INSUFFICIENT_COVERAGE"
    except BaseException:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "ENDURANCE_INTERRUPTED_OR_FAILED"
    finally:
        if video:
            video.close()
        if engine:
            engine.close()
        if backend:
            backend.close()
        if monitor:
            report["resources"] = monitor.sample()
            monitor.close()
        report["elapsed_seconds"] = time.monotonic() - started
        report["formal_P1_P2_P3_or_lifelong_pass"] = False
        with (run / "result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        trace.event("shared_endurance.completed", {
            k: v for k, v in report.items() if k not in ("video_cycles",)}, "orchestrator")
        private.close()
        trace.close()
        print(encode({"run_id": name, "verdict": report["verdict"],
                      "failures": report["failures"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return not report["failures"] and report.get("gate_pass", False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=int, choices=(90, 28800), required=True)
    raise SystemExit(0 if main(p.parse_args().seconds) else 1)
