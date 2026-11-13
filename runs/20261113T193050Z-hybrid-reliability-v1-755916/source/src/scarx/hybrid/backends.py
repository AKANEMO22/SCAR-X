"""One bounded CPU VLM worker and one FP32 ASR worker, owned by the gateway."""

import base64
import hashlib
import json
import secrets
import os
import socket
from pathlib import Path
import subprocess
import threading
import time
from urllib.request import Request, urlopen
from .store import encode
from .config import HybridConfig


class NativeBackends:
    def __init__(self, directory, store, monitor, config=None):
        self.directory = Path(directory)
        self.store = store
        self.monitor = monitor
        self.config = config or HybridConfig()
        self.vlm = None
        self.asr = None
        self.lock = threading.RLock()
        self.asr_lock = threading.RLock()
        self.logs = []
        self.cancel_idle = threading.Event()
        self.vlm_url = "http://127.0.0.1:18776"
        self.native_token = secrets.token_urlsafe(32)
        self.vlm_command = None
        self.asset_signatures = None

    def whisper_executable(self):
        runtime = (
            "whisper.cpp-b5454-cuda12.4"
            if self.config.execution_mode == "gpu"
            else "whisper.cpp-b5454"
        )
        return self.directory / "runtime" / runtime / "Release/whisper-cli.exe"

    def headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.native_token,
        }

    def _spawn(self, command, logname):
        if Path(command[0]).name == "llama-server.exe":
            self.vlm_command = list(command)
        # Popen gives the child its own handle; do not retain one Python file
        # handle for every frozen-ASR activation across a long-running session.
        with (self.directory / logname).open("ab") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        self.store.event(
            "worker_started",
            {
                "pid": process.pid,
                "command": [
                    (
                        "<internal-key>"
                        if i and str(command[i - 1]) == "--api-key"
                        else str(value)
                    )
                    for i, value in enumerate(command)
                ],
            },
            "runtime",
        )
        return process

    def pause_vlm_for_audio(self):
        report = self.monitor.sample()
        projected = (
            report.get("conservative_ram_bytes", report["rss_bytes"]) + 600_000_000
        )
        if self.monitor.limit is None or projected <= self.monitor.limit:
            return
        if self.vlm and self.vlm.poll() is None:
            self.vlm.terminate()
            try:
                self.vlm.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.vlm.kill()
                self.vlm.wait()
            self.store.event(
                "vlm_suspended_for_audio",
                {
                    "projected_bytes": projected,
                    "limit_bytes": self.monitor.limit,
                    "state": "frozen sensory model unloaded as a whole; SNN/evidence/checkpoint retained",
                    "cost": "next VLM call reloads whole checkpoint and loses prefix cache; no layer/weight paging",
                },
                "runtime",
            )
        self.vlm = None

    def ensure_vlm(self):
        if self.vlm is not None and self.vlm.poll() is None:
            return
        if not self.vlm_command:
            raise RuntimeError("VLM was never initialized")
        current = self._asset_signatures()
        if self.asset_signatures is not None and current != self.asset_signatures:
            from .qualification import verify_native_storage

            verify_native_storage(self.directory)
            self.asset_signatures = current
        self.monitor.check()
        started = time.monotonic()
        self.vlm = self._spawn(self.vlm_command, "vlm.log")
        self.wait(self.vlm, self.vlm_url + "/health")
        self.store.event(
            "vlm_resumed",
            {
                "seconds": time.monotonic() - started,
                "prefix_cache": "cold",
                "precision": "original BF16/F32",
            },
            "runtime",
        )

    def hibernate(self, reason, permitted=None):
        with self.lock:
            if self.cancel_idle.is_set() or (permitted is not None and not permitted()):
                raise InterruptedError("Foreground preempts model hibernation")
            if self.vlm is None or self.vlm.poll() is not None:
                self.vlm = None
                return False
            pid = self.vlm.pid
            self.vlm.terminate()
            try:
                self.vlm.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.vlm.kill()
                self.vlm.wait()
            self.vlm = None
            self.store.event(
                "vlm_hibernated",
                {
                    "pid": pid,
                    "reason": reason,
                    "ALIF_retained": True,
                    "weights_changed": False,
                    "next_answer": "reloads frozen model; cold prefix",
                },
            )
            return True

    def wake(self):
        with self.lock:
            self.ensure_vlm()

    def _asset_signatures(self):
        return tuple(
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in (
                self.directory / "models/native/qwen-native-bf16.gguf",
                self.directory / "models/native/qwen-vision-native-bf16.gguf",
            )
        )

    def start(self):
        native = self.directory / "models/native"
        runtime = (
            "llama.cpp-b12360-cuda12.4"
            if self.config.execution_mode == "gpu"
            else "llama.cpp-b12360"
        )
        llama = self.directory / "runtime" / runtime / "llama-server.exe"
        whisper = self.whisper_executable()
        for path in (
            llama,
            whisper,
            native / "qwen-native-bf16.gguf",
            native / "qwen-vision-native-bf16.gguf",
            native / "whisper-base-f32.bin",
        ):
            if not path.is_file():
                raise FileNotFoundError("Run setup/conversion first: " + str(path))
        for port in (18776,):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(
                        "Native worker port is occupied; refusing to load a second model: "
                        + str(port)
                    )
        # Modes change execution placement, never stored BF16/F32 model tensors.
        placement = (
            ["--device", "CUDA" + str(self.config.gpu_device)]
            if self.config.execution_mode == "gpu"
            else ["--no-mmproj-offload"]
        )
        self.vlm = self._spawn(
            [
                str(llama),
                "-m",
                str(native / "qwen-native-bf16.gguf"),
                "--mmproj",
                str(native / "qwen-vision-native-bf16.gguf"),
                *placement,
                "--host",
                "127.0.0.1",
                "--port",
                "18776",
                "-ngl",
                "99" if self.config.execution_mode == "gpu" else "0",
                "-t",
                "4",
                "-tb",
                "4",
                "-c",
                str(self.config.context_tokens),
                "-np",
                "1",
                "-b",
                "128",
                "-ub",
                "64",
                "--no-warmup",
                "--cache-ram",
                "0",
                "--cache-type-k",
                "bf16",
                "--cache-type-v",
                "bf16",
                "--jinja",
                "--reasoning-budget",
                "0",
                "--fit",
                "off",
                "--alias",
                "scarx-hybrid",
                "--api-key",
                self.native_token,
            ],
            "vlm.log",
        )
        self.wait(self.vlm, self.vlm_url + "/health")
        self.monitor.check()
        self.asset_signatures = self._asset_signatures()
        # Frozen ASR is activated only for an audio chunk. This is component
        # scheduling, not weight paging or reduced precision. A single VLM and
        # one audio chunk execute at a time under the aggregate RAM budget.

    def stop_asr(self):
        process = self.asr
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            self.store.event(
                "asr_yielded_memory",
                {
                    "pid": process.pid,
                    "precision": "FP32 unchanged",
                    "reason": "bounded scheduling before VLM execution",
                },
                "runtime",
            )
        self.asr = None

    def wait(self, process, url, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Native worker exited; inspect worker log")
            self.monitor.check()
            try:
                with urlopen(url, timeout=1) as response:
                    if response.status < 400:
                        return
            except Exception:
                pass
            time.sleep(0.1)
        raise TimeoutError("Native worker readiness deadline")

    def generate(
        self,
        messages,
        max_tokens=160,
        background=False,
        tools=None,
        tool_choice=None,
        response_format=None,
    ):
        if not 1 <= max_tokens <= 1024:
            raise ValueError("Bounded output")
        body = {
            "model": "scarx-hybrid",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": True,
            "stream": background,
        }
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        started = time.monotonic()
        with self.lock:
            self.stop_asr()
            self.ensure_vlm()
            if background and self.cancel_idle.is_set():
                raise InterruptedError(
                    "Foreground request preempts background inference"
                )
            self.monitor.check()
            with urlopen(
                Request(
                    self.vlm_url + "/v1/chat/completions",
                    data=encode(body).encode(),
                    headers=self.headers(),
                ),
                timeout=120,
            ) as response:
                if background:
                    parts = []
                    usage = {}
                    received = 0
                    # Token stream permits cooperative preemption between tokens.
                    for line in response:
                        if self.cancel_idle.is_set():
                            raise InterruptedError(
                                "Foreground request preempted background inference"
                            )
                        received += len(line)
                        if received > 1_000_000:
                            raise ValueError("Model stream exceeds bound")
                        if not line.startswith(b"data: "):
                            continue
                        event = line[6:].strip()
                        if event == b"[DONE]":
                            break
                        item = json.loads(event)
                        usage = item.get("usage") or usage
                        for choice in item.get("choices", []):
                            parts.append(choice.get("delta", {}).get("content") or "")
                    result = {
                        "choices": [{"message": {"content": "".join(parts)}}],
                        "usage": usage,
                    }
                else:
                    raw = response.read(1_000_001)
                    if len(raw) > 1_000_000:
                        raise ValueError("Model response exceeds bound")
                    result = json.loads(raw)
            self.monitor.check()
        text = result["choices"][0]["message"].get("content") or ""
        elapsed = time.monotonic() - started
        audit = []
        for message in messages:
            row = dict(message)
            if isinstance(row.get("content"), list):
                row["content"] = [
                    (
                        {
                            "type": "image_url",
                            "image_sha256": hashlib.sha256(
                                part["image_url"]["url"].encode()
                            ).hexdigest(),
                            "encoded_bytes": len(part["image_url"]["url"]),
                        }
                        if part.get("type") == "image_url"
                        else part
                    )
                    for part in row["content"]
                ]
            audit.append(row)
        self.store.event(
            "vlm_answer",
            {
                "messages": audit,
                "answer": text,
                "elapsed_seconds": elapsed,
                "usage": result.get("usage"),
                "timings": result.get("timings"),
            },
            "vlm",
        )
        return {
            "text": text,
            "usage": result.get("usage", {}),
            "seconds": elapsed,
            "raw": result,
        }

    def stream(self, messages, max_tokens=512, tools=None, tool_choice=None):
        body = {
            "model": "scarx-hybrid",
            "messages": messages,
            "max_tokens": min(1024, max(1, int(max_tokens))),
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        with self.lock:
            self.stop_asr()
            self.ensure_vlm()
            self.monitor.check()
            with urlopen(
                Request(
                    self.vlm_url + "/v1/chat/completions",
                    data=encode(body).encode(),
                    headers=self.headers(),
                ),
                timeout=120,
            ) as response:
                received = 0
                for line in response:
                    received += len(line)
                    if received > 2_000_000:
                        raise ValueError("Native stream exceeds bound")
                    if not line.startswith(b"data: "):
                        continue
                    value = line[6:].strip()
                    if value == b"[DONE]":
                        break
                    yield json.loads(value)
            self.monitor.check()

    def describe(self, image_path, background=False):
        data = Path(image_path).read_bytes()
        if len(data) > 2_000_000:
            raise ValueError("Image input exceeds bounded frame size")
        return self.generate(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(data).decode()
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe the visible image briefly, including shapes, colors, objects, actions and readable text. Do not infer unseen events or speech.",
                        },
                    ],
                }
            ],
            max_tokens=64,
            background=background,
        )

    def transcribe(self, wav_path):
        import uuid

        source = Path(wav_path).resolve()
        if not source.is_file() or source.stat().st_size > 2_000_000:
            raise ValueError("Audio chunk exceeds bound or is unavailable")
        started = time.monotonic()
        work = self.directory / "work" / "asr"
        work.mkdir(parents=True, exist_ok=True)
        output = work / uuid.uuid4().hex
        command = [
            str(self.whisper_executable()),
            "-m",
            str(self.directory / "models/native/whisper-base-f32.bin"),
            "-f",
            str(source),
            "-l",
            "auto",
            "-t",
            "2",
            *(
                ["-dev", str(self.config.gpu_device)]
                if self.config.execution_mode == "gpu"
                else ["-ng"]
            ),
            "-ojf",
            "-of",
            str(output),
        ]
        with self.lock, self.asr_lock:
            self.monitor.check()
            self.pause_vlm_for_audio()
            self.asr = self._spawn(command, "asr.log")
            try:
                if self.asr.wait(timeout=120):
                    raise RuntimeError("Native FP32 ASR failed; inspect asr.log")
                path = output.with_suffix(".json")
                if path.stat().st_size > 1_000_000:
                    raise ValueError("ASR result exceeds bound")
                result = json.loads(path.read_text(encoding="utf-8"))
                text = " ".join(
                    row.get("text", "").strip()
                    for row in result.get("transcription", [])
                ).strip()
                self.monitor.check()
            finally:
                self.stop_asr()
            self.store.event(
                "asr_result",
                {
                    "text": text,
                    "seconds": time.monotonic() - started,
                    "source": str(source),
                    "raw_result": str(path),
                    "precision": "FP32",
                    "device": self.config.execution_mode,
                    "backend": "one owned whisper-cli per bounded chunk; no ASR HTTP service",
                },
                "asr",
            )
        return text

    def close(self):
        for process in (self.asr, self.vlm):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in self.logs:
            log.close()
