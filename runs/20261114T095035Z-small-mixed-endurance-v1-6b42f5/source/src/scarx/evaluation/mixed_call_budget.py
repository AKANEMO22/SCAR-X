"""Phase attempt ledger for mixed pipeline engineering, separate from ASR P0."""

import json
import os
from pathlib import Path
import tempfile

LIMITS = {"preflight": {"ASR": 12, "VLM": 32}, "long": {"ASR": 360, "VLM": 640}}


def charge(path, phase, kind):
    if phase not in LIMITS or kind not in LIMITS[phase]:
        raise ValueError("Unregistered mixed-phase call")
    path = Path(path)
    lock = path.with_suffix(".lock")
    handle = lock.open("x")
    try:
        with handle:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                "limits": LIMITS, "attempted": {p: dict.fromkeys(v, 0) for p, v in LIMITS.items()}}
            if state.get("limits") != LIMITS or set(state.get("attempted", {})) != set(LIMITS):
                raise ValueError("Mixed ledger mismatch; no reset")
            for p, caps in LIMITS.items():
                values = state["attempted"][p]
                if set(values) != set(caps) or any(type(values[k]) is not int or not 0 <= values[k] <= caps[k] for k in caps):
                    raise ValueError("Corrupt mixed ledger; no reset")
            if state["attempted"][phase][kind] >= LIMITS[phase][kind]:
                raise RuntimeError("Mixed phase aggregate attempted-call budget exhausted")
            state["attempted"][phase][kind] += 1
            descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".mixed-attempt-", suffix=".tmp")
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(state, output, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            return state["attempted"][phase][kind]
    finally:
        lock.unlink()
