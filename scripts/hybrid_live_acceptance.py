"""Actual 21s public/synthetic engineering video + restart evidence, not P1."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import uuid

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import encode

BASE = "http://127.0.0.1:8775"


def api(path, body=None):
    headers = {"X-SCARX-Backend": "hermes"}
    if body is not None:
        token = api("/api/status")["token"]
        headers.update({"Content-Type": "application/json", "X-SCARX-Token": token})
    with urlopen(
        Request(
            BASE + path,
            data=None if body is None else encode(body).encode(),
            headers=headers,
        ),
        timeout=120,
    ) as response:
        return json.load(response)


def question(text, stream=None):
    return api(
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": text}], "scarx_stream_id": stream},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.resume:
        if "/" in args.resume or "\\" in args.resume:
            raise ValueError("Expected a run ID")
        run = ROOT / "runs" / args.resume
        before = json.loads((run / "before_restart.json").read_text(encoding="utf-8"))
        if (run / "after_restart.json").exists():
            raise ValueError("Refuse to overwrite restart evidence")
        answer = question(before["memory_question"], before["stream_id"])
        status = api("/api/status")
        status.pop("token", None)
        result = {
            "previous_run": args.resume,
            "answer": answer,
            "resources": status["resources"],
            "persisted_stream": next(
                s for s in status["streams"] if s["id"] == before["stream_id"]
            ),
            "policy_updates": status["neural"]["updates"],
            "claim": "restart retrieval engineering only; no synaptic knowledge/generalization claim",
        }
        (run / "after_restart.json").write_text(encode(result), encoding="utf-8")
        print(
            encode(
                {
                    "run_id": args.resume,
                    "after_restart_answer": answer["choices"][0]["message"]["content"],
                    "resources": status["resources"],
                }
            )
        )
        return
    run_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-hybrid-live-acceptance-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True)
    fixture = json.loads(
        (ROOT / "local/hybrid-v1/samples/engineering-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    report = {
        "run_id": run_id,
        "kind": "P0 engineering fixture; NOT P1/100-question benchmark",
        "fixture": fixture,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src/scarx/hybrid").glob("*")
            if p.is_file()
        },
        "started": time.time(),
        "failures": [],
    }
    start_cursor = (
        api("/api/status")["events"][-1]["id"] if api("/api/status")["events"] else 0
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            readiness = api("/api/status?after=2147483647")
            if readiness.get("ready"):
                break
            if readiness.get("failure"):
                raise RuntimeError(readiness["failure"])
            time.sleep(0.2)
        else:
            raise TimeoutError("Native qualification readiness deadline")
        report["greeting"] = question("Chào bạn! Trả lời một câu ngắn bằng tiếng Việt.")
        stream = api("/api/observe", {"path": fixture["video"]})
        report["stream_id"] = stream["id"]
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status = api("/api/status")
            current = next(s for s in status["streams"] if s["id"] == stream["id"])
            if current["status"] in ("failed", "stopped"):
                raise RuntimeError(current.get("error") or current["status"])
            if current["status"] == "complete":
                break
            time.sleep(1)
        else:
            raise TimeoutError("Engineering video deadline")
        report["stream"] = current
        report["memory_question"] = "Mô tả các hình và màu sắc ở đầu video."
        report["memory_answer"] = question(report["memory_question"], stream["id"])
        report["checkpoint"] = api("/api/save", {})["id"]
        status = api("/api/status")
        report["resources"] = status["resources"]
        report["neural"] = status["neural"]
        report["status"] = "ENGINEERING_COMPLETED; semantic answers require inspection"
    except Exception as exc:
        report["failures"].append(str(exc))
        report["status"] = "ENGINEERING_FAILED"
    finally:
        report["finished"] = time.time()
        (run / "before_restart.json").write_text(encode(report), encoding="utf-8")
        print(
            encode(
                {
                    "run_id": run_id,
                    "status": report["status"],
                    "failures": report["failures"],
                    "greeting": report.get("greeting", {}).get("choices"),
                    "memory_answer": report.get("memory_answer", {}).get("choices"),
                    "resources": report.get("resources"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
