"""Automatic evidence of edits in this isolated implementation only."""

import hashlib
from pathlib import Path
import threading


class WorkTrace:
    def __init__(self, store):
        self.store = store
        self.root = Path(__file__).resolve().parents[3]
        self.seen = {}
        self.stop = threading.Event()
        self.thread = None

    def scan(self):
        paths = list((self.root / "src/scarx/hybrid").glob("*.py")) + list(
            (self.root / "src/scarx/hybrid").glob("*.html")
        )
        paths += list((self.root / "scripts").glob("*hybrid*.py")) + list(
            (self.root / "tests").glob("test_hybrid*.py")
        )
        paths += list(
            (self.root / "integrations/hermes/hybrid_plugin").rglob("*.py")
        ) + list((self.root / "integrations/hermes/hybrid_plugin").rglob("*.js"))
        for path in paths:
            stat = path.stat()
            key = str(path.relative_to(self.root))
            version = (stat.st_mtime_ns, stat.st_size)
            if self.seen.get(key) == version:
                continue
            previous = self.seen.get(key)
            self.seen[key] = version
            self.store.event(
                "work_trace.activity",
                {
                    "kind": "file_changed" if previous else "source_observed",
                    "file": key,
                    "modified_epoch": stat.st_mtime,
                    "bytes": stat.st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "measured": True,
                },
                "orchestrator",
            )

    def start(self):
        self.scan()

        def loop():
            while not self.stop.wait(10):
                try:
                    self.scan()
                except OSError:
                    pass

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
