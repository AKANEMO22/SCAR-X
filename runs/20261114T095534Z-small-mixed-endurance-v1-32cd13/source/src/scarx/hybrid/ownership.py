"""One writer per data directory; recover only a provably dead owner."""

import json
import os
from pathlib import Path
import secrets
import psutil


class OwnerLock:
    def __init__(self, directory):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "runtime.lock"
        self.nonce = secrets.token_hex(16)
        if self.path.exists():
            try:
                owner = json.loads(self.path.read_text())
                process = psutil.Process(owner["pid"])
                alive = abs(process.create_time() - float(owner["created"])) < 0.01
            except psutil.NoSuchProcess:
                alive = False
            except (ValueError, KeyError, TypeError):
                raise RuntimeError("Invalid runtime lock; inspect " + str(self.path))
            if alive:
                raise RuntimeError("Another gateway owns this data directory")
            self.path.unlink()  # This exact owned lock file; never a directory.
        raw = json.dumps(
            {
                "pid": os.getpid(),
                "created": psutil.Process().create_time(),
                "nonce": self.nonce,
            }
        ).encode()
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())

    def close(self):
        if (
            self.path.exists()
            and json.loads(self.path.read_text()).get("nonce") == self.nonce
        ):
            self.path.unlink()
