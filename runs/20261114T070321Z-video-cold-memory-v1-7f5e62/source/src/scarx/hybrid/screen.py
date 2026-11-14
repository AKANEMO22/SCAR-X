"""Explicit window grants; bounded sampled frames and optional system loopback."""

import ctypes as c
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from PIL import Image


def window_sources():
    if os.name != "nt":
        return []
    from ctypes import wintypes as w

    user = c.WinDLL("user32", use_last_error=True)
    items = []
    callback = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
    user.GetWindowTextLengthW.argtypes = [w.HWND]
    user.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, c.c_int]
    user.IsWindowVisible.argtypes = [w.HWND]

    @callback
    def visit(hwnd, _):
        length = user.GetWindowTextLengthW(hwnd)
        if length and user.IsWindowVisible(hwnd):
            text = c.create_unicode_buffer(length + 1)
            user.GetWindowTextW(hwnd, text, length + 1)
            items.append({"id": int(hwnd), "title": text.value})
        return True

    user.EnumWindows.argtypes = [callback, w.LPARAM]
    user.EnumWindows(visit, 0)
    return items


class ScreenMemory:
    def __init__(self, engine, prune=None):
        self.engine = engine
        self.store = engine.store
        self.directory = self.store.directory
        self.jobs = {}
        self.prune = prune
        self.last_prune = 0.0
        self.heartbeats = {}
        self.threads = []

    def start(self, hwnd, system_audio=False, session_id=None):
        if type(hwnd) is not int or type(system_audio) is not bool:
            raise ValueError("Invalid screen selection")
        source = next((s for s in window_sources() if s["id"] == hwnd), None)
        if source is None:
            raise ValueError("Selected window no longer exists")
        if any(p.poll() is None for p, _ in self.jobs.values()):
            raise ValueError("Stop the active screen share before starting another")
        identifier = uuid.uuid4().hex
        owned = self.directory / "archive" / identifier
        owned.mkdir(parents=True)
        root = Path(__file__).resolve().parents[3]
        python = root / "local/hybrid-v1/runtime-env/Scripts/python.exe"
        if not python.exists():
            raise RuntimeError(
                "Install the optional capture dependencies in runtime-env first"
            )
        state = {
            "id": identifier,
            "kind": "screen",
            "window": source,
            "session_id": session_id,
            "system_audio_requested": system_audio,
            "microphone_used": False,
            "timebase": "seconds since sharing started, not original player position",
            "status": "starting",
            "capture_frontier": 0.0,
            "vision_frontier": 0.0,
            "asr_frontier": 0.0,
            "latest_frame": None,
            "audio_available": False,
            "error": None,
        }
        command = [
            str(python),
            str(root / "scripts/hybrid_screen_worker.py"),
            "--hwnd",
            str(hwnd),
            "--directory",
            str(owned),
        ]
        if system_audio:
            command.append("--system-audio")
        log = (owned / "worker.log").open("ab")
        try:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            log.close()
        self.jobs[identifier] = (process, owned)
        heartbeat_stop = threading.Event()
        self.heartbeats[identifier] = heartbeat_stop

        def heartbeat():
            while not heartbeat_stop.is_set() and process.poll() is None:
                (owned / "controller-alive").touch()
                heartbeat_stop.wait(3)

        threading.Thread(target=heartbeat, daemon=True).start()
        self.store.put("stream", identifier, state)
        self.store.event(
            "screen_share_started",
            {
                "stream_id": identifier,
                "source": source,
                "system_audio_requested": system_audio,
                "microphone_used": False,
                "grant_origin": "explicit normal-composer window selection",
            },
            "user",
        )
        thread = threading.Thread(
            target=self._ingest, args=(state, process, owned), daemon=True
        )
        self.threads.append(thread)
        thread.start()
        return state

    def _ingest(self, state, process, owned):
        journal = owned / "capture.jsonl"
        offset = 0
        last_vision = -10.0
        previous_audio = None
        try:
            while True:
                (owned / "controller-alive").touch()
                self.engine.monitor.check()
                rows = []
                if journal.exists():
                    with journal.open(encoding="utf-8") as handle:
                        handle.seek(offset)
                        for _ in range(64):
                            line = handle.readline()
                            if not line or not line.endswith("\n"):
                                break
                            rows.append(json.loads(line))
                            offset = handle.tell()
                frames = [r for r in rows if r["kind"] == "frame"]
                if frames:
                    frame = frames[-1]
                    state["capture_frontier"] = frame["time"]
                    state["latest_frame"] = str(owned / frame["path"])
                    state["status"] = "sharing"
                for row in rows:
                    if row["kind"] == "audio_started":
                        state["audio_available"] = True
                    elif row["kind"] == "audio_unavailable":
                        state["audio_available"] = False
                        state["audio_error"] = row["error"]
                    elif row["kind"] == "audio":
                        audio = (owned / row["path"]).resolve()
                        if not audio.is_relative_to(owned.resolve()):
                            raise ValueError("Invalid capture source")
                        output = owned / "asr-input.wav"
                        inputs = []
                        filters = None
                        if previous_audio and previous_audio.exists():
                            inputs = ["-i", str(previous_audio), "-i", str(audio)]
                            filters = [
                                "-filter_complex",
                                "[0:a][1:a]concat=n=2:v=0:a=1[a]",
                                "-map",
                                "[a]",
                            ]
                        else:
                            inputs = ["-i", str(audio)]
                            filters = []
                        subprocess.run(
                            ["ffmpeg", "-v", "error", "-nostdin", "-y"]
                            + inputs
                            + filters
                            + [
                                "-ar",
                                "16000",
                                "-ac",
                                "1",
                                "-c:a",
                                "pcm_s16le",
                                str(output),
                            ],
                            capture_output=True,
                            check=True,
                            timeout=15,
                        )
                        if not self.engine.foreground.is_set():
                            text = self.engine.backend.transcribe(output)
                            if text and text != "[BLANK_AUDIO]":
                                self.engine.observe(
                                    text,
                                    stream=state["id"],
                                    start=max(
                                        0.0,
                                        row["start"] - (10 if previous_audio else 0),
                                    ),
                                    end=row["end"],
                                    source=str(audio),
                                    modality="audio",
                                )
                            state["asr_frontier"] = row["end"]
                        previous_audio = audio
                if (
                    frames
                    and frame["time"] - last_vision >= 8
                    and not self.engine.foreground.is_set()
                ):
                    path = owned / "inference-frame.jpg"
                    with Image.open(state["latest_frame"]) as image:
                        image.thumbnail((512, 512))
                        image.convert("RGB").save(path, quality=85)
                    try:
                        result = self.engine.backend.describe(path, background=True)
                        self.engine.observe(
                            result["text"],
                            stream=state["id"],
                            start=frame["time"],
                            end=frame["time"],
                            source=state["latest_frame"],
                            modality="sampled_frame",
                        )
                        state["vision_frontier"] = frame["time"]
                        last_vision = frame["time"]
                    except InterruptedError:
                        pass
                self.store.put("stream", state["id"], state)
                if self.prune and time.monotonic() - self.last_prune >= 10:
                    self.last_prune = time.monotonic()
                    self.prune(
                        exclude=(
                            {Path(state["latest_frame"])}
                            if state["latest_frame"]
                            else set()
                        )
                    )
                if process.poll() is not None and not rows:
                    break
                time.sleep(0.5)
            state["status"] = "stopped" if process.returncode == 0 else "failed"
            if process.returncode:
                state["error"] = "Capture worker failed; see its local worker.log"
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = str(exc)
            if process.poll() is None:
                process.terminate()
        finally:
            self.heartbeats[state["id"]].set()
            self.store.put("stream", state["id"], state)
            self.store.event("screen_share_finished", state, "runtime")

    def stop(self, identifier):
        if identifier not in self.jobs:
            raise ValueError("No active screen share with this identity")
        process, owned = self.jobs[identifier]
        (owned / "stop").touch()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=3)
        self.store.event(
            "screen_share_stop_requested", {"stream_id": identifier}, "user"
        )
        return {"stopped": True}

    def snapshot(self, identifier, evidence=None):
        state = self.store.get("stream", identifier)
        if not state or state.get("kind") != "screen" or not state.get("latest_frame"):
            return None
        root = (self.directory / "archive" / identifier).resolve()
        historical = next(
            (e for e in evidence or () if e.get("modality") == "sampled_frame"), None
        )
        if historical:
            source = Path(historical["source"]).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                self.store.event(
                    "historical_screen_frame_unavailable",
                    {
                        "stream_id": identifier,
                        "evidence_id": historical["id"],
                        "time": historical["start"],
                        "reason": "retained source missing; no latest-frame substitution",
                    },
                )
                return None
            self.store.event(
                "source_screen_reinspected",
                {
                    "stream_id": identifier,
                    "evidence_id": historical["id"],
                    "time": historical["start"],
                    "source": str(source),
                },
            )
        else:
            source = None
        if source is None:
            pointer = root / "latest.json"
            latest = (
                json.loads(pointer.read_text())
                if pointer.exists()
                else {"path": Path(state["latest_frame"]).name}
            )
            source = (root / latest["path"]).resolve()
        if not source.is_relative_to(root) or not source.exists():
            return None
        target = root / "question-frame.jpg"
        with Image.open(source) as image:
            image.thumbnail((512, 512))
            image.convert("RGB").save(target, quality=85)
        return target

    def close(self):
        for identifier in list(self.jobs):
            if self.jobs[identifier][0].poll() is None:
                self.stop(identifier)
        for thread in self.threads:
            thread.join(timeout=15)
