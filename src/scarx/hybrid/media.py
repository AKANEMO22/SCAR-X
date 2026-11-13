"""Causal 1x capture to owned segments, separate ASR/vision frontiers and TTL."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid


class VideoMemory:
    def __init__(self, engine):
        self.engine = engine
        self.store = engine.store
        self.directory = self.store.directory
        self.archive = (self.directory / "archive").resolve()
        self.archive.mkdir(exist_ok=True)
        self.threads = []
        self.processes = []
        self.stop_event = threading.Event()
        self.last_prune = 0.0

    def observe(self, path, public_topic_id=None):
        path = Path(path).resolve()
        if not path.is_file() or path.suffix.lower() not in (
            ".mp4",
            ".mkv",
            ".webm",
            ".mov",
            ".avi",
        ):
            raise ValueError("Expected a readable local video file")
        if public_topic_id and not self.store.get("public_topic", public_topic_id):
            raise ValueError("Unknown public topic")
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=True,
            timeout=15,
        )
        info = json.loads(probe.stdout)
        if not any(s.get("codec_type") == "video" for s in info["streams"]):
            raise ValueError("Input has no video stream")
        identifier = uuid.uuid4().hex
        owned = self.archive / identifier
        owned.mkdir()
        state = {
            "id": identifier,
            "original": str(path),
            "original_owned": False,
            "created": time.time(),
            "duration": float(info.get("format", {}).get("duration", 0)),
            "status": "capturing",
            "capture_frontier": 0.0,
            "asr_frontier": 0.0,
            "vision_frontier": 0.0,
            "analyzed_intervals": [],
            "skipped_intervals": [],
            "public_topic_id": public_topic_id,
            "error": None,
        }
        self.store.put("stream", identifier, state)
        thread = threading.Thread(target=self._run, args=(state, owned), daemon=True)
        self.threads.append(thread)
        thread.start()
        return state

    def _run(self, state, owned):
        listing = owned / "segments.csv"
        log = (owned / "capture.log").open("wb")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-re",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            state["original"],
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(self.engine.config.segment_seconds),
            "-reset_timestamps",
            "1",
            "-segment_list",
            str(listing),
            "-segment_list_type",
            "csv",
            str(owned / "%06d.mkv"),
        ]
        process = None
        seen = set()
        try:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.processes.append(process)
            self.store.event(
                "video_capture_started",
                {"stream_id": state["id"], "command": command, "pid": process.pid},
            )
            while not self.stop_event.wait(0.5):
                self.engine.monitor.check()
                if time.monotonic() - self.last_prune >= 10:
                    self.prune()
                rows = []
                if listing.exists():
                    try:
                        with listing.open(newline="") as handle:
                            rows = list(csv.reader(handle))
                    except (OSError, csv.Error):
                        pass
                pending = [r for r in rows if len(r) == 3 and r[0] not in seen]
                if pending:
                    state["capture_frontier"] = max(
                        state["capture_frontier"], float(pending[-1][2])
                    )
                    if self.engine.foreground.is_set():
                        self.store.put("stream", state["id"], state)
                        continue
                    # Bound analysis backlog; preserve raw sources and explicitly log coverage loss.
                    for skipped in pending[:-1]:
                        seen.add(skipped[0])
                        state["skipped_intervals"].append(
                            [float(skipped[1]), float(skipped[2])]
                        )
                        self.store.event(
                            "video_analysis_skipped",
                            {
                                "stream_id": state["id"],
                                "start": skipped[1],
                                "end": skipped[2],
                                "reason": "bounded backlog; raw source retained",
                            },
                        )
                    row = pending[-1]
                    seen.add(row[0])
                    segment = (owned / row[0]).resolve()
                    if not segment.is_relative_to(owned.resolve()):
                        raise ValueError("Invalid capture member")
                    start, end = map(float, row[1:])
                    try:
                        self._analyze(state, segment, start, end)
                    except InterruptedError:
                        state["skipped_intervals"].append([start, end])
                        self.store.event(
                            "video_analysis_preempted",
                            {"stream_id": state["id"], "start": start, "end": end},
                        )
                    self.prune(exclude={segment})
                self.store.put("stream", state["id"], state)
                if process.poll() is not None:
                    if process.returncode:
                        raise RuntimeError(
                            "FFmpeg capture failed; see " + str(owned / "capture.log")
                        )
                    if not pending:
                        break
            state["status"] = "stopped" if self.stop_event.is_set() else "complete"
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = str(exc)
            self.store.event(
                "video_job_failed", {"stream_id": state["id"], "error": str(exc)}
            )
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
            self.store.put("stream", state["id"], state)

    def _analyze(self, state, segment, start, end):
        self.engine.monitor.check()
        temporary = self.directory / "work" / state["id"]
        temporary.mkdir(parents=True, exist_ok=True)
        wav = temporary / "audio.wav"
        image = temporary / "frame.jpg"
        audio = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(segment),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(wav),
            ],
            capture_output=True,
            timeout=20,
        )
        if audio.returncode == 0 and not self.engine.foreground.is_set():
            text = self.engine.backend.transcribe(wav)
            if text and text != "[BLANK_AUDIO]":
                self.engine.observe(
                    text,
                    stream=state["id"],
                    start=start,
                    end=end,
                    source=str(segment),
                    modality="audio",
                    public_topic=state["public_topic_id"],
                )
            state["asr_frontier"] = end
        if self.engine.foreground.is_set():
            state["skipped_intervals"].append([start, end])
            return
        offset = max(0.0, (end - start) / 2)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                str(offset),
                "-i",
                str(segment),
                "-frames:v",
                "1",
                "-vf",
                "scale=384:384:force_original_aspect_ratio=decrease",
                "-q:v",
                "3",
                str(image),
            ],
            capture_output=True,
            check=True,
            timeout=20,
        )
        description = self.engine.backend.describe(image, background=True)
        # The statement concerns one sampled frame, not every frame of this interval.
        self.engine.observe(
            description["text"],
            stream=state["id"],
            start=start + offset,
            end=start + offset,
            source=str(segment),
            modality="sampled_frame",
            public_topic=state["public_topic_id"],
        )
        state["vision_frontier"] = end
        state["analyzed_intervals"].append([start, end])
        state["analyzed_intervals"] = state["analyzed_intervals"][-512:]
        state["skipped_intervals"] = state["skipped_intervals"][-512:]
        self.store.event(
            "video_segment_analyzed",
            {
                "stream_id": state["id"],
                "start": start,
                "end": end,
                "visual_sample_time": start + offset,
                "capture_frontier": state["capture_frontier"],
                "asr_frontier": state["asr_frontier"],
                "vision_frontier": state["vision_frontier"],
                "lag_seconds": max(
                    0.0,
                    state["capture_frontier"]
                    - min(state["asr_frontier"], state["vision_frontier"]),
                ),
            },
        )

    def prune(self, exclude=()):
        # Delete only individually verified owned regular files, never recursive paths.
        now = time.time()
        self.last_prune = time.monotonic()
        excluded = {Path(p).resolve() for p in exclude}
        files = []
        for path in self.archive.glob("*/*"):
            if path.suffix not in (".mkv", ".jpg", ".wav") or path.name in (
                "inference-frame.jpg",
                "question-frame.jpg",
                "asr-input.wav",
            ):
                continue
            resolved = path.resolve()
            if (
                path.is_symlink()
                or not resolved.is_relative_to(self.archive)
                or resolved in excluded
            ):
                continue
            files.append((path.stat().st_mtime, path.stat().st_size, path))
        total = sum(size for _, size, _ in files)
        for stamp, size, path in sorted(files):
            if (
                total <= self.engine.config.archive_bytes
                and now - stamp <= self.engine.config.archive_seconds
            ):
                continue
            # Active current segment is never in a completed CSV entry; keep very recent files.
            if now - stamp < 2 * self.engine.config.segment_seconds:
                continue
            path.unlink()
            total -= size
            self.store.put(
                "expired_source",
                str(path),
                {
                    "source": str(path),
                    "deleted": now,
                    "reason": "owned archive TTL/quota",
                },
            )
            self.store.event(
                "owned_video_expired",
                {"source": str(path), "bytes": size, "original_deleted": False},
            )

    def snapshot(self, stream, evidence):
        """Reinspect retained source, rather than trusting an old generated caption."""
        if not evidence:
            return None
        item = next(
            (e for e in evidence if e.get("modality") == "sampled_frame"), evidence[0]
        )
        source = Path(item["source"]).resolve()
        owned = (self.archive / stream).resolve()
        if (
            not source.is_relative_to(owned)
            or not source.is_file()
            or source.suffix != ".mkv"
        ):
            return None
        state = self.store.get("stream", stream) or {}
        moment = float(item["start"])
        interval = next(
            (
                row
                for row in state.get("analyzed_intervals", [])
                if row[0] <= moment <= row[1]
            ),
            None,
        )
        if interval is None:
            return None
        target = owned / "question-frame.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                str(max(0.0, moment - interval[0])),
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                "scale=512:512:force_original_aspect_ratio=decrease",
                str(target),
            ],
            capture_output=True,
            check=True,
            timeout=15,
        )
        self.store.event(
            "source_video_reinspected",
            {
                "stream_id": stream,
                "time": moment,
                "source": str(source),
                "image": str(target),
            },
            "runtime",
        )
        return target

    def close(self):
        self.stop_event.set()
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for thread in self.threads:
            thread.join(timeout=5)
