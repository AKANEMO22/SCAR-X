"""Read-only client of the exact owned native worker; never starts/kills a model."""

import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import psutil


class SharedNative:
    def __init__(self, root):
        self.root = Path(root).resolve()
        status = self.status()
        if (not status.get("ready") or status.get("failure")
                or status["VLM_identity"]["name"] != "Qwen/Qwen3.5-0.8B"
                or status["VLM_identity"]["revision"] != "2fc06364715b967f1860aea9cf38778875588b17"
                or status["config"]["execution_mode"] != "gpu"
                or status["config"]["context_tokens"] != 16384
                or status["config"]["gpu_device"] != 0):
            raise RuntimeError("Expected current qualified native0.8B CUDA gateway")
        self.owner = psutil.Process(status["pid"])
        self.owner_created = self.owner.create_time()
        expected = self.root / "local/hybrid-v1/runtime/llama.cpp-b12360-cuda12.4/llama-server.exe"
        matches = [p for p in self.owner.children(recursive=True)
                   if Path(p.exe()).resolve() == expected]
        if len(matches) != 1:
            raise RuntimeError("Ambiguous owned native worker")
        self.worker = matches[0]
        self.worker_created = self.worker.create_time()
        args = self.worker.cmdline()

        def argument(name):
            return args[args.index(name) + 1]
        if (argument("--port") != "18776" or argument("--host") != "127.0.0.1"
                or argument("--device") != "CUDA0" or argument("-ngl") != "99"
                or argument("-c") != "16384"
                or argument("--cache-type-k") != "bf16"
                or argument("--cache-type-v") != "bf16"
                or Path(argument("-m")).resolve() !=
                   self.root / "local/hybrid-v1/models/native/qwen-native-bf16.gguf"):
            raise RuntimeError("Native placement/model identity mismatch")
        self._token = argument("--api-key")  # Ephemeral local token, never serialize.

    @staticmethod
    def status():
        with urlopen("http://127.0.0.1:8775/api/status?after=2147483647", timeout=5) as response:
            return json.load(response)

    def check(self):
        for process, created in ((self.owner, self.owner_created),
                                 (self.worker, self.worker_created)):
            if not process.is_running() or process.create_time() != created:
                raise RuntimeError("Owned native worker identity changed")

    def generate(self, body):
        self.check()
        started = time.monotonic()
        while self.status().get("cognition", {}).get("phase") == "serving":
            if time.monotonic() - started > 30:
                raise TimeoutError("Foreground user work retains priority")
            time.sleep(.1)
        self.check()
        payload = {
            "model": "scarx-hybrid", "max_tokens": 128, "temperature": 0,
            "repeat_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": True, **body,
        }
        with urlopen(Request(
            "http://127.0.0.1:18776/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self._token},
        ), timeout=90) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Native response exceeds budget")
        self.check()
        return json.loads(raw)
