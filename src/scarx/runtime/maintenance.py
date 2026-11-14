"""Short owned maintenance lease prevents auto-start during a source transaction."""

import json
import math
import os
from pathlib import Path
import secrets
import psutil


def _owner(path):
    raw = path.read_bytes()
    if len(raw) > 4096:
        raise RuntimeError("Invalid SCAR-X maintenance lease")
    try:
        value = json.loads(raw)
        if (type(value["pid"]) is not int or value["pid"] <= 0
                or type(value["created"]) not in (int, float)
                or not math.isfinite(value["created"]) or value["created"] <= 0
                or not isinstance(value["nonce"], str) or len(value["nonce"]) != 32):
            raise ValueError("Invalid owner")
        process = psutil.Process(value["pid"])
        alive = abs(process.create_time() - value["created"]) < 0.01
    except psutil.NoSuchProcess:
        alive = False
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Invalid SCAR-X maintenance lease; inspect before startup") from exc
    return value, alive


def check_startup(directory):
    path = Path(directory).resolve() / ".startup-maintenance.lock"
    try:
        _, alive = _owner(path)
    except FileNotFoundError:
        return
    if alive:
        raise RuntimeError("SCAR-X is undergoing an owned maintenance transaction; retry shortly")


class MaintenanceLease:
    def __init__(self, directory):
        self.path = Path(directory).resolve() / ".startup-maintenance.lock"
        self.nonce = secrets.token_hex(16)
        if self.path.exists():
            _, alive = _owner(self.path)
            if alive:
                raise RuntimeError("Another live process owns SCAR-X maintenance")
            self.path.unlink()  # Exact lease of a provably dead/reused process.
        payload = json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time(),
                              "nonce": self.nonce}).encode()
        with self.path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def close(self):
        try:
            owner, _ = _owner(self.path)
        except FileNotFoundError:
            return
        if owner["nonce"] == self.nonce:
            self.path.unlink()
