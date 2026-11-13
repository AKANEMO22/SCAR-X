"""Isolated native2B actual load/vision/ASR/VRAM probe; restore in outer shell."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from scarx.hybrid import gateway as serving
from scarx.hybrid.store import encode
import hybrid_cognition_acceptance as harness
import hybrid_service


def smi():
    raw = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        .strip()
        .splitlines()[0]
    )
    used, free, total, util = [int(x.strip()) for x in raw.split(",")]
    return {
        "used_MiB": used,
        "free_MiB": free,
        "total_MiB": total,
        "utilization_percent": util,
    }


def prepare(run):
    source = ROOT / "local/hybrid-v1"
    candidate = source / "candidates/qwen3.5-2b"
    data = ROOT / "local" / run.name
    if data.exists():
        raise ValueError("Candidate lab already prepared")
    data.mkdir()
    shutil.copytree(source / "runtime", data / "runtime", copy_function=os.link)
    (data / "models/native").mkdir(parents=True)
    conversion = json.loads((run / "conversion.json").read_text(encoding="utf-8"))
    reference = json.loads((run / "reference.json").read_text(encoding="utf-8"))
    if reference["status"] != "REFERENCE_READY" or reference["mismatches"]:
        raise ValueError("Candidate source reference failed")
    exports = {}
    for name, entry in conversion["exports"].items():
        os.link(candidate / "native" / name, data / "models/native" / name)
        exports[name] = {k: entry[k] for k in ("bytes", "sha256")}
    old = json.loads((source / "native-integrity.json").read_text(encoding="utf-8"))
    name = "whisper-base-f32.bin"
    os.link(source / "models/native" / name, data / "models/native" / name)
    exports[name] = old["exports"][name]
    (data / "native-integrity.json").write_text(
        encode(
            {
                "status": "NATIVE_STORAGE_INTEGRITY_PASS",
                "run_id": run.name,
                "quantized": False,
                "exports": exports,
            }
        ),
        encoding="utf-8",
    )
    (data / "original-reference.json").write_text(encode(reference), encoding="utf-8")
    shutil.copytree(source / "samples", data / "samples", copy_function=os.link)
    (data / "config.json").write_text(
        encode(
            {"execution_mode": "gpu", "searxng_enabled": False, "gpu_keep_warm": True}
        ),
        encoding="utf-8",
    )
    questions = [
        {
            "id": "arithmetic",
            "question": "Compute17+25. Return only the integer.",
            "expected": "42",
        },
        {
            "id": "latest_setting",
            "question": "The port was8123. It was later changed to9237. Return only the current port number.",
            "expected": "9237",
        },
        {
            "id": "image_circle",
            "question": "What color is the circle on the right? Return one color word.",
            "expected": "blue",
            "image": "scene-first.png",
        },
        {
            "id": "prior_failed_update",
            "question": 'Project receipt: codeJADE-472; port initially8123. Later update: only port changed to9237. Return JSON {"code":"JADE-472","port":integer} of the final state.',
            "expected_json": {"code": "JADE-472", "port": 9237},
        },
    ]
    (run / "probe-cases.json").write_text(encode(questions), encoding="utf-8")
    (run / "probe-lock.json").write_text(
        encode(
            {
                "prepared": time.time(),
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "cases_sha256": hashlib.sha256(
                    (run / "probe-cases.json").read_bytes()
                ).hexdigest(),
                "scope": "engineeringDEV; no model promotion without native/mixed/headroom checks; no neural learning changes",
                "min_free_MiB": 512,
            }
        ),
        encoding="utf-8",
    )
    print(encode({"candidate_prepared": str(data)}), flush=True)


def run_probe(run):
    if hybrid_service.status():
        raise RuntimeError("Stop owned gateway first; no second VLM")
    if (run / "probe.json").exists() or (run / "probe.started").exists():
        raise ValueError("Refuse repeat probe")
    lock = json.loads((run / "probe-lock.json").read_text())
    if lock["script_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Frozen probe changed")
    (run / "probe.started").write_text(str(time.time()))
    data = ROOT / "local" / run.name
    holder = {}
    original = serving.Gateway
    server = serving.ThreadingHTTPServer

    class Observed(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            holder["gw"] = self

    serving.Gateway = Observed
    serving.ThreadingHTTPServer = lambda address, handler: server(
        address, harness.isolated_handler(handler, harness.NONCE)
    )
    started = time.monotonic()
    thread = threading.Thread(target=serving.serve, args=(data, 8775), daemon=True)
    thread.start()
    report = {
        "started": time.time(),
        "cases": [],
        "failures": [],
        "model": "Qwen3.5-2B original BF16/F32",
        "live_promoted": False,
        "resources_before": smi(),
    }
    sampler = None
    gpu_file = None
    try:

        def ready():
            try:
                s = harness.api("/api/status?after=2147483647")
                if s.get("failure"):
                    raise RuntimeError(s["failure"])
                return s["ready"]
            except OSError:
                return False

        harness.wait_for(ready, 120, "candidate native qualification")
        gw = holder["gw"]
        harness.api("/api/learning", {"enabled": False})
        report["boot_ready_seconds"] = time.monotonic() - started
        report["native_gate"] = gw.store.get("runtime", "native_gate")
        report["loaded_gpu"] = smi()
        gpu_file = (run / "candidate-nvidia.csv").open("w")
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                "-lms",
                "100",
            ],
            stdout=gpu_file,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for case in json.loads((run / "probe-cases.json").read_text()):
            content = case["question"]
            if case.get("image"):
                image = data / "samples" / case["image"]
                content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(image.read_bytes()).decode()
                        },
                    },
                    {"type": "text", "text": content},
                ]
            response = harness.api(
                "/v1/chat/completions",
                {
                    "model": "scarx-hybrid",
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 96,
                    "stream": True,
                },
                stream=True,
            )
            answer = response["answer"].strip()
            correct = answer.casefold() == case.get("expected", "").casefold()
            if "expected_json" in case:
                try:
                    correct = json.loads(answer) == case["expected_json"]
                except ValueError:
                    correct = False
            row = {
                "id": case["id"],
                "question": case["question"],
                "expected": case.get("expected", case.get("expected_json")),
                "output": response,
                "correct": correct,
                "gpu": smi(),
                "resources": gw.monitor.sample(),
            }
            report["cases"].append(row)
            gw.store.event("candidate2b.actual_answer", row, "orchestrator")
            print(
                encode({"case": case["id"], "correct": correct, "gpu": row["gpu"]}),
                flush=True,
            )
        # Predict ASR scratch needs before risking WDDM spill. If little room,
        # record a concurrent-mixed failure and test explicit component scheduling.
        report["before_ASR_gpu"] = smi()
        if report["before_ASR_gpu"]["free_MiB"] < 1050:
            report["concurrent_ASR"] = (
                "not attempted: reserved512MiB + conservative538MiB ASR headroom unavailable"
            )
            harness.api("/api/cognition", {"intent": "sleep"})
            harness.wait_for(
                lambda: not gw.engine.cognition.resident(),
                10,
                "whole VLM scheduling for ASR",
            )
            report["ASR_scheduled_after_unload"] = True
        else:
            report["concurrent_ASR"] = "attempted with at least1050MiB free"
        t = time.monotonic()
        transcript = gw.backend.transcribe(data / "samples/fleurs-vi.wav")
        report["ASR"] = {
            "seconds": time.monotonic() - t,
            "text": transcript,
            "gpu": smi(),
            "device": "gpu",
        }
        report["ALIF_device"] = gw.engine.model.device
        report["mixed_qualified"] = (
            not report.get("ASR_scheduled_after_unload", False)
            and min(
                [report["loaded_gpu"]["free_MiB"]]
                + [x["gpu"]["free_MiB"] for x in report["cases"]]
                + [report["ASR"]["gpu"]["free_MiB"]]
            )
            >= 512
        )
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        if sampler:
            sampler.terminate()
            sampler.wait(5)
            gpu_file.close()
        if "gw" in holder:
            gw = holder["gw"]
            report["resources"] = gw.monitor.sample()
            try:
                harness.api("/api/shutdown", {})
            except Exception as exc:
                report["failures"].append("shutdown: " + str(exc))
            thread.join(30)
            report["gateway_stopped"] = not thread.is_alive()
        report["seconds"] = time.monotonic() - started
        report["finished"] = time.time()
        (run / "probe.json").write_text(encode(report), encoding="utf-8")
        print(
            encode(
                {
                    "receipt": str(run / "probe.json"),
                    "failures": report["failures"],
                    "mixed_qualified": report.get("mixed_qualified", False),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--prepare", action="store_true")
    a = p.parse_args()
    run = a.run.resolve()
    if ROOT / "runs" not in run.parents:
        raise ValueError("External run path")
    if a.prepare:
        prepare(run)
    else:
        run_probe(run)
