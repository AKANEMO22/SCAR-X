"""Small-ASR private mixed engineering pipeline; same frozen live VLM, no live learning edits."""

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
import psutil

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.shared_native import SharedNative
from scarx.evaluation.endurance_gates import qualifies
from scarx.hybrid.backends import NativeBackends
from scarx.hybrid.config import HybridConfig
from scarx.hybrid.device_budget import gpu_headroom, audio_requires_release
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.media import VideoMemory
from scarx.hybrid.resources import ResourceMonitor, windows_tree_commit_limit
from scarx.hybrid.store import Store, encode
from scarx.runtime.asr_profiles import PROFILES
from scarx.evaluation.mixed_call_budget import charge
from scarx.evaluation.hermes_resource_roots import verified_gateway_pid


class SharedSensory:
    def __init__(self, native, data, store, monitor, config, phase):
        self.native = native
        self.audio = NativeBackends(data, store, monitor, config)
        self.phase = phase
        class AudioMonitor:
            def check(inner):
                if native.status().get("cognition", {}).get("phase") == "serving":
                    self.audio.cancel_idle.set()
                return monitor.check()
            def sample(inner):
                return monitor.sample()
        self.audio.monitor = AudioMonitor()
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
        charge(ROOT / "local/small-mixed-endurance-v1-budget.json", self.phase, "VLM")
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
        charge(ROOT / "local/small-mixed-endurance-v1-budget.json", self.phase, "ASR")
        return self.audio.transcribe(wav)

    def close(self):
        self.audio.close()  # Own ASR only; no reference to live VLM in this object.


def main(seconds, preflight=None):
    if seconds not in (240, 28800):
        raise ValueError("Registered small-profile preflight240s or endurance28800s only")
    if seconds == 28800 and preflight is None:
        raise ValueError("8h requires explicit passed240s preflight with identical frozen source")
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-small-mixed-endurance-v1-" + uuid.uuid4().hex[:6]
    run, data = ROOT / "runs" / name, ROOT / "local" / name
    run.mkdir()
    data.mkdir()
    source = ROOT / "local/hybrid-v1"
    for folder in ("runtime", "models/native", "candidates/whisper-small"):
        shutil.copytree(source / folder, data / folder, copy_function=os.link)
    (data / "native-integrity.json").write_bytes((source / "native-integrity.json").read_bytes())
    sample = source / "samples/engineering-vi-video.mkv"
    paths = list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        ROOT / "src/scarx/neural/core.py", ROOT / "src/scarx/evaluation/readout_eligibility.py",
        ROOT / "src/scarx/evaluation/shared_native.py", ROOT / "src/scarx/learning/verifiers.py",
        ROOT / "src/scarx/evaluation/endurance_gates.py",
        ROOT / "experiments/hybrid-small-mixed-endurance-v1.md", Path(__file__),
        ROOT / "src/scarx/runtime/asr_profiles.py", ROOT / "src/scarx/evaluation/mixed_call_budget.py",
        ROOT / "src/scarx/evaluation/hermes_resource_roots.py", ROOT / "scripts/monitor_hybrid_endurance.py",
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
        "private_data": str(data), "video_interval": {90: 60, 240: 120, 28800: 300}[seconds],
        "stage_interval": {90: 15, 240: 60, 28800: 1440}[seconds],
        "checkpoint_interval": {90: 30, 240: 60, 28800: 1800}[seconds],
        "scope": "Private small-ASR engineering fixture endurance; no formal learning/quality claim",
        "runner": "scripts/hybrid_small_mixed_endurance.py",
        "process_created_epoch": psutil.Process().create_time(),
        "selected_ASR": PROFILES["openai/whisper-small"].identity(),
        "model_export_sha256": PROFILES["openai/whisper-small"].native_sha256,
    }
    if seconds == 28800:
        preflight = Path(preflight).resolve()
        if ROOT / "runs" not in preflight.parents:
            raise ValueError("External preflight path")
        prior_manifest = json.loads((preflight / "manifest.json").read_text(encoding="utf-8"))
        prior = json.loads((preflight / "result.json").read_text(encoding="utf-8"))
        if (prior_manifest["duration_seconds"] != 240
                or prior_manifest["source_sha256"] != hashes
                or prior["failures"] or not prior.get("gate_pass")
                or not qualifies(
                    requested=240, elapsed=prior["elapsed_seconds"],
                    cycles=len(prior["video_cycles"]), checkpoints=len(prior["checkpoints"]),
                    updates=prior["final_neural"]["updates"],
                    replay_seen=prior["final_neural"]["replay_seen"],
                    device=prior["final_neural"]["device"],
                    critical_failures=prior["critical_quantum_failures"])):
            raise ValueError("Missing/stale/failed240s preflight")
        manifest["preflight_run_id"] = preflight.name
        manifest["preflight_result_sha256"] = hashlib.sha256((preflight / "result.json").read_bytes()).hexdigest()
    (run / "manifest.json").write_text(encode(manifest), encoding="utf-8")
    (run / "process-identity-v1.json").write_text(encode({"pid": os.getpid(), "process_created_epoch": psutil.Process().create_time()}), encoding="utf-8")
    (ROOT / "local/current-small-mixed-endurance-run.txt").write_text(str(run))
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
            native.owner.pid}, include_hermes=True)
        report["lifecycle_job"] = windows_tree_commit_limit(None)
        monitor.start()
        if status.get("ASR_identity", {}).get("name") != "openai/whisper-small":
            raise ValueError("Live selected-ASR profile must match this small-profile run")
        config = HybridConfig(execution_mode="gpu", searxng_enabled=False,
            asr_model_name="openai/whisper-small", asr_revision=PROFILES["openai/whisper-small"].revision)
        backend = SharedSensory(native, data, private, monitor, config, "long" if seconds == 28800 else "preflight")
        engine = HybridEngine(private, backend, monitor, config)
        with private.lock, private.db:
            private.db.execute("DELETE FROM objects WHERE kind='public_topic'")
        engine.model.to_device("gpu")
        execute = engine.cognition.execute

        def measured_execute(work, now):
            start_wall, start_cpu = time.monotonic(), time.process_time()
            cpu = native.worker.cpu_times()
            start_native = cpu.user + cpu.system
            value = execute(work, now)  # Unmodified scheduler/learning rules.
            if work != "resting":
                cpu = native.worker.cpu_times()
                private.event("cognitive_quantum_cpu_profile", {
                    "work": work, "wall_seconds": time.monotonic() - start_wall,
                    "python_process_CPU_seconds": time.process_time() - start_cpu,
                    "shared_native_CPU_seconds": cpu.user + cpu.system - start_native,
                    "next_delay_seconds": max(0, engine.state["cognition"]["next_quantum"] - time.time()),
                    "scope": "Persistent-process CPU subsets; excludes exited workers/other clients",
                    "scheduler_changed": False,
                }, "orchestrator")
            return value

        engine.cognition.execute = measured_execute
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
                    if state["asr_errors"] or state["skipped_intervals"]:
                        raise RuntimeError("Completed stream has ASR errors or skipped intervals")
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
                hermes_pid = verified_gateway_pid(Path(os.environ["LOCALAPPDATA"]) / "hermes")
                if hermes_pid not in resource["included_client_pids"]:
                    raise RuntimeError("Actual current Hermes gateway missing from dynamic monitor roots")
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
                        "GPU_headroom": headroom, "state": "running",
                        "selected_ASR": config.asr_model_name, "actual_Hermes_gateway_pid": hermes_pid}
                (run / "heartbeat.json").write_text(encode(beat), encoding="utf-8")
                trace.event("shared_endurance.heartbeat", beat, "orchestrator")
                print(encode({k: beat[k] for k in (
                    "run_id", "elapsed_seconds", "video_cycles", "checkpoints", "neural")}), flush=True)
                next_heartbeat = elapsed + 30
            time.sleep(.25)
        report["elapsed_seconds"] = time.monotonic() - started
        report["final_neural"] = {"updates": engine.model.updates,
            "hidden_updates": engine.model.hidden_updates, "device": engine.model.device,
            "replay_seen": engine.model.replay_seen}
        with private.lock:
            critical = private.db.execute(
                "SELECT count(*) FROM events WHERE kind='cognitive_quantum_failed'"
            ).fetchone()[0]
        report["critical_quantum_failures"] = critical
        with private.lock:
            ASR_results = [json.loads(row[0]) for row in private.db.execute("SELECT payload FROM events WHERE kind='asr_result' ORDER BY id")]
        report["actual_small_ASR_results"] = len(ASR_results)
        if not ASR_results or any(row.get("ASR_identity") != PROFILES["openai/whisper-small"].identity()
                or row.get("actual_neural_device") != "CUDA0"
                or row.get("execution_device_log_verified") is not True
                or row.get("precision_log_verified") is not True for row in ASR_results):
            raise ValueError("Completed ASR calls do not establish selected small/CUDA/FP32 execution")
        report["gate_pass"] = qualifies(
            requested=seconds, elapsed=report["elapsed_seconds"],
            cycles=len(report["video_cycles"]), checkpoints=len(report["checkpoints"]),
            updates=engine.model.updates, replay_seen=engine.model.replay_seen,
            device=engine.model.device, critical_failures=critical, active_capture=active is not None)
        if seconds == 28800 and engine.model.replay_seen <= 0:
            report["gate_pass"] = False
        report["verdict"] = ("EIGHT_HOUR_ENGINEERING_PASS" if seconds == 28800 else "SMOKE_ENGINEERING_PASS") if report["gate_pass"] else "INSUFFICIENT_COVERAGE"
    except BaseException:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "ENDURANCE_INTERRUPTED_OR_FAILED"
    finally:
        if video:
            video.close()
            if any(thread.is_alive() for thread in video.threads):
                report["failures"].append("Private video thread still live after close")
        if engine:
            engine.close()
            if engine.thread and engine.thread.is_alive():
                report["failures"].append("Private cognitive worker still active after close")
                report["gate_pass"] = False
                report["verdict"] = "ENDURANCE_CLEANUP_INCOMPLETE"
        if backend:
            backend.close()
        if monitor:
            report["resources"] = monitor.sample()
            monitor.close()
        report["elapsed_seconds"] = time.monotonic() - started
        report["formal_P1_P2_P3_or_lifelong_pass"] = False
        budget_path = ROOT / "local/small-mixed-endurance-v1-budget.json"
        report["aggregate_component_call_budget"] = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else None
        if report["failures"]:
            report["gate_pass"] = False
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
    p.add_argument("--seconds", type=int, choices=(240, 28800), required=True)
    p.add_argument("--preflight", type=Path)
    args = p.parse_args()
    raise SystemExit(0 if main(args.seconds, args.preflight) else 1)
