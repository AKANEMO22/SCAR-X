"""Capture only our exact generated window, actual worker and actual VLM."""

import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import encode


def api(path, body=None):
    headers = {"X-SCARX-Backend": "hermes"}
    if body is not None:
        token = api("/api/status?after=2147483647")["token"]
        headers |= {"X-SCARX-Token": token, "Content-Type": "application/json"}
    with urlopen(
        Request(
            "http://127.0.0.1:8775" + path,
            data=encode(body).encode() if body is not None else None,
            headers=headers,
        ),
        timeout=90,
    ) as response:
        return json.load(response)


if __name__ == "__main__":
    run_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-hybrid-screen-p0-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True)
    report = {
        "kind": "actual synthetic-window engineering capture; NOT P1",
        "run_id": run_id,
        "failures": [],
    }
    fixture = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/hybrid_capture_fixture.py")],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stream = None
    try:
        deadline = time.monotonic() + 10
        source = None
        while time.monotonic() < deadline:
            sources = api("/api/screen/sources")["sources"]
            source = next(
                (
                    s
                    for s in sources
                    if s["title"]
                    == "SCAR-X engineering capture fixture — synthetic shapes"
                ),
                None,
            )
            if source:
                break
            time.sleep(0.2)
        if source is None:
            raise RuntimeError("Own engineering fixture window not found")
        stream = api(
            "/api/screen/start",
            {
                "hwnd": source["id"],
                "system_audio": False,
                "session_id": "engineering-fixture",
            },
        )
        report["source"] = source
        report["stream_id"] = stream["id"]
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            state = api("/api/status?after=2147483647")
            row = next(s for s in state["streams"] if s["id"] == stream["id"])
            if row["status"] == "failed":
                raise RuntimeError(row["error"])
            if row["vision_frontier"] >= 1:
                break
            time.sleep(0.5)
        if row["vision_frontier"] < 1:
            raise TimeoutError("Capture-to-VLM engineering deadline")
        report["state"] = row
        report["resources"] = state["resources"]
        report["status"] = "CAPTURE_ENGINEERING_PASS"
    except Exception as exc:
        report["status"] = "CAPTURE_ENGINEERING_FAILED"
        report["failures"].append(str(exc))
    finally:
        if stream:
            try:
                report["stop"] = api("/api/screen/stop", {"id": stream["id"]})
            except Exception as exc:
                report["failures"].append(str(exc))
        if fixture.poll() is None:
            fixture.terminate()
            fixture.wait(timeout=3)
        (run / "receipt.json").write_text(encode(report), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True), flush=True)
