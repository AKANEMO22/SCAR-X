"""Idempotent private gateway lifecycle. Never stop foreign/peer processes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "local/hybrid-v1"


def status():
    try:
        with urlopen(
            "http://127.0.0.1:8775/api/status?after=2147483647", timeout=2
        ) as response:
            value = json.load(response)
        if value.get("model") != "scarx-hybrid":
            raise RuntimeError("Port8775 belongs to another service")
        return value
    except URLError:
        return None


def start():
    current = status()
    if current:
        return {
            "already_running": True,
            "pid": current["pid"],
            "phase": current["phase"],
        }
    python = DATA / "runtime-env/Scripts/python.exe"
    if not python.exists():
        python = Path(sys.executable)
    DATA.mkdir(parents=True, exist_ok=True)
    with (
        (DATA / "gateway.stdout.log").open("ab") as out,
        (DATA / "gateway.stderr.log").open("ab") as err,
    ):
        process = subprocess.Popen(
            [str(python), str(ROOT / "run_hybrid.py")],
            cwd=ROOT,
            stdout=out,
            stderr=err,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = status()
        if current:
            return {"started": True, "pid": current["pid"], "phase": current["phase"]}
        if process.poll() is not None:
            raise RuntimeError("Gateway exited; see local gateway.stderr.log")
        time.sleep(0.2)
    raise TimeoutError("Gateway startup timeout; see local logs")


def stop():
    import psutil

    lock = DATA / "runtime.lock"
    if not lock.exists():
        return {"stopped": False, "reason": "no owner lock"}
    owner = json.loads(lock.read_text())
    process = psutil.Process(owner["pid"])
    if abs(process.create_time() - owner["created"]) > 0.01:
        raise RuntimeError("PID identity changed; refusing to terminate")
    command = process.cmdline()
    if not any(
        Path(arg).resolve() == ROOT / "run_hybrid.py"
        for arg in command[1:]
        if not arg.startswith("--")
    ):
        raise RuntimeError("Owner command does not match this checkout")
    # Stop through authenticated loopback so full checkpoint/worker cleanup runs.
    from urllib.request import Request

    with urlopen(
        Request(
            "http://127.0.0.1:8775/api/status", headers={"X-SCARX-Backend": "hermes"}
        ),
        timeout=2,
    ) as response:
        token = json.load(response)["token"]
    with urlopen(
        Request(
            "http://127.0.0.1:8775/api/shutdown",
            data=b"{}",
            headers={"Content-Type": "application/json", "X-SCARX-Token": token},
        ),
        timeout=5,
    ) as response:
        response.read()
    try:
        process.wait(timeout=15)
    except psutil.TimeoutExpired:
        raise RuntimeError("Gateway did not stop gracefully; inspect before forcing")
    return {"stopped": True, "pid": owner["pid"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    args = parser.parse_args()
    print(
        json.dumps(
            {"start": start, "stop": stop, "status": status}[args.action](),
            ensure_ascii=True,
        )
    )
