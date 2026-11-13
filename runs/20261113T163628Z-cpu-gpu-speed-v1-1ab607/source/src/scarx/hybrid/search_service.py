"""One owned native SearXNG child; no Docker, WSL or unowned port takeover."""

from pathlib import Path
import socket
import subprocess
import time
from urllib.request import urlopen


class SearchService:
    def __init__(self, store, monitor, enabled=False):
        self.store, self.monitor, self.enabled = store, monitor, enabled
        self.process = None
        self.root = Path(__file__).resolve().parents[3]

    def start(self):
        if not self.enabled:
            return
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 18778)) == 0:
                raise RuntimeError(
                    "SearXNG port18778 already occupied; refusing unowned takeover"
                )
        base = self.root / "local/searxng"
        python = base / "env/Scripts/python.exe"
        if not python.is_file() or not (base / "settings.yml").is_file():
            raise RuntimeError(
                "Run setup_native_searxng.py and prepare its local config first"
            )
        with (base / "server.log").open("ab") as log:
            self.process = subprocess.Popen(
                [str(python), str(self.root / "scripts/run_native_searxng.py")],
                cwd=base,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Native SearXNG exited; inspect local/searxng/server.log"
                )
            self.monitor.check()
            try:
                with urlopen("http://127.0.0.1:18778/config", timeout=1) as r:
                    if r.status == 200:
                        self.store.event(
                            "search_service_ready",
                            {
                                "backend": "searxng",
                                "pid": self.process.pid,
                                "url": "http://127.0.0.1:18778",
                                "Docker": False,
                                "WSL": False,
                                "RAM_in_owned_tree": True,
                            },
                        )
                        return
            except OSError:
                pass
            time.sleep(0.2)
        raise TimeoutError("SearXNG local readiness timeout")

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
