"""Actual native-worker engineering measurements; never calls this P1/P2 success."""

import hashlib
import json
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlretrieve
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image, ImageDraw
from scarx.hybrid.backends import NativeBackends
from scarx.hybrid.resources import ResourceMonitor, windows_tree_commit_limit
from scarx.hybrid.store import Store, encode


def main():
    data = ROOT / "local/hybrid-v1"
    store = Store(data)
    run_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-hybrid-p0-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True)
    receipt = {
        "run_id": run_id,
        "stage": "P0",
        "claim": "engineering native inference only; no quality/hidden-learning success",
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "results": {},
        "failures": [],
        "full_model_parity": "pending",
        "integrated_hermes": "not measured",
    }
    # Observer is outside this worker tree but part of the system being measured.
    import psutil

    observer = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if (
                process.info["name"].lower().startswith("python")
                and "hybrid_observe.py" in command
                and process.info["pid"] != psutil.Process().pid
            ):
                observer.append(process.info["pid"])
        except psutil.Error:
            pass
    monitor = ResourceMonitor(extra_pids=observer).start()
    receipt["windows_job"] = windows_tree_commit_limit()
    backend = NativeBackends(data, store, monitor)
    store.put(
        "runtime",
        "status",
        {
            "phase": "P0 native measurements",
            "note": "Actual CPU workers starting; P1/P2/P3 not opened",
        },
    )
    try:
        store.event(
            "p0_started",
            {
                "run_id": run_id,
                "observer_pids": observer,
                "job": receipt["windows_job"],
            },
            "orchestrator",
        )
        backend.start()
        receipt["results"]["english"] = backend.generate(
            [{"role": "user", "content": "Say hello in one short sentence."}],
            max_tokens=32,
        )
        receipt["results"]["vietnamese"] = backend.generate(
            [
                {
                    "role": "user",
                    "content": "Chào bạn, hãy giới thiệu ngắn gọn bằng tiếng Việt.",
                }
            ],
            max_tokens=64,
        )
        image = Image.new("RGB", (384, 256), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 50, 170, 200), fill="red")
        draw.ellipse((220, 70, 340, 190), fill="blue")
        frame = run / "visible-shapes.jpg"
        image.save(frame)
        receipt["results"]["vision"] = backend.describe(frame)
        wav = data / "downloads/jfk.wav"
        if not wav.exists():
            urlretrieve(
                "https://raw.githubusercontent.com/ggml-org/whisper.cpp/b5454/samples/jfk.wav",
                wav,
            )
        receipt["results"]["asr_english"] = backend.transcribe(wav)
        receipt["results"]["audio_sha256"] = hashlib.sha256(
            wav.read_bytes()
        ).hexdigest()
        receipt["results"]["loaded_resources"] = monitor.check()
    except Exception as exc:
        receipt["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        store.event("p0_failure", receipt["failures"][-1], "orchestrator")
    finally:
        receipt["resources"] = monitor.close()
        backend.close()
        receipt["status"] = (
            "FAIL_RESOURCE"
            if receipt["resources"]["exceeded"]
            else (
                "FAIL_ENGINEERING"
                if receipt["failures"]
                else "ENGINEERING_PASS_PARITY_PENDING"
            )
        )
        receipt["downstream_stages"] = (
            "BLOCKED until native parity, persistence and integrated resource gates pass"
        )
        (run / "receipt.json").write_text(encode(receipt), encoding="utf-8")
        store.put(
            "runtime",
            "p0",
            {
                "run_id": run_id,
                "status": receipt["status"],
                "receipt": str(run / "receipt.json"),
            },
        )
        store.put(
            "runtime",
            "status",
            {"phase": receipt["status"], "note": receipt["downstream_stages"]},
        )
        store.event(
            "p0_finished",
            {
                "run_id": run_id,
                "status": receipt["status"],
                "resources": receipt["resources"],
                "failures": receipt["failures"],
                "receipt": str(run / "receipt.json"),
            },
            "orchestrator",
        )
        store.close()
        print(
            encode(
                {
                    "run_id": run_id,
                    "status": receipt["status"],
                    "resources": receipt["resources"],
                    "failures": receipt["failures"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
