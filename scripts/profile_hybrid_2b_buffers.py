"""Three fixed native2B profiles, isolated GPU gateway and real inputs."""

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from scarx.hybrid import gateway as serving
from scarx.hybrid.store import encode
import hybrid_cognition_acceptance as harness
from probe_hybrid_qwen2b import smi
import hybrid_service

PROFILES = {
    "baseline": {
        "batch_size": 128,
        "microbatch_size": 64,
        "flash_attention": "auto",
        "cuda_graphs": True,
    },
    "small_flash": {
        "batch_size": 64,
        "microbatch_size": 32,
        "flash_attention": "on",
        "cuda_graphs": True,
    },
    "small_no_graph": {
        "batch_size": 64,
        "microbatch_size": 32,
        "flash_attention": "on",
        "cuda_graphs": False,
    },
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare():
    from PIL import Image, ImageDraw

    name = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-native-2b-buffer-v2-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / name
    run.mkdir()
    old = ROOT / "local/20261113T212018Z-native-2b-vram-v1-77a5fb"
    for profile, config in PROFILES.items():
        data = ROOT / "local" / (name + "-" + profile)
        data.mkdir()
        shutil.copytree(old / "runtime", data / "runtime", copy_function=os.link)
        shutil.copytree(old / "models", data / "models", copy_function=os.link)
        shutil.copytree(old / "samples", data / "samples", copy_function=os.link)
        for f in ["native-integrity.json", "original-reference.json"]:
            shutil.copyfile(old / f, data / f)
        (data / "config.json").write_text(
            encode(
                dict(
                    config,
                    execution_mode="gpu",
                    searxng_enabled=False,
                    gpu_keep_warm=True,
                )
            ),
            encoding="utf-8",
        )
    image = Image.new("RGB", (1920, 1080), (16, 20, 28))
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 250, 650, 800), fill="red")
    draw.ellipse((1050, 250, 1650, 850), fill="blue")
    image.save(run / "large-scene.png")
    filler = " ".join(
        f"Unrelated inventory row{i}: count{i%19}, no update to the target port."
        for i in range(80)
    )
    cases = [
        {
            "id": "arithmetic",
            "question": "Compute17+25. Return only the integer.",
            "expected": "42",
        },
        {
            "id": "current_port",
            "question": "The port was8123 and then changed to9237. Return only the current port.",
            "expected": "9237",
        },
        {
            "id": "long_update",
            "question": "Target project port starts8123. "
            + filler
            + " Final authoritative update: target project port is9237. Return only its current port.",
            "expected": "9237",
        },
        {
            "id": "small_image",
            "question": "What color is the circle on the right? Return one color word.",
            "image": "scene-first.png",
            "expected": "blue",
        },
        {
            "id": "large_image",
            "question": "What color is the circle on the right? Return one color word.",
            "image": "large-scene.png",
            "expected": "blue",
        },
        {
            "id": "strict_schema",
            "question": "Project code is JADE-472. The port was8123, later changed to9237. Return JSON with code and final port.",
            "expected_json": {"code": "JADE-472", "port": 9237},
        },
    ]
    (run / "cases.json").write_text(encode(cases), encoding="utf-8")
    files = list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        Path(__file__),
        ROOT / "experiments/hybrid-native-2b-buffer-v2.md",
    ]
    hashes = {}
    for path in files:
        rel = path.relative_to(ROOT)
        out = run / "source" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(path.read_bytes())
        hashes[rel.as_posix()] = sha(path.read_bytes())
    (run / "manifest.json").write_text(
        encode(
            {
                "prepared": time.time(),
                "profiles": PROFILES,
                "source_sha256": hashes,
                "cases_sha256": sha((run / "cases.json").read_bytes()),
                "large_image_sha256": sha((run / "large-scene.png").read_bytes()),
                "context": 16384,
                "reserve_MiB": 512,
                "repetitions": 3,
                "max_total_seconds": 720,
                "source_model_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                "scope": "resource/kernel engineering; old failed profile remains historical",
            }
        ),
        encoding="utf-8",
    )
    (ROOT / "local/current-2b-buffer-run.txt").write_text(str(run))
    print(encode({"prepared": str(run)}))


def arm(run, name):
    manifest = json.loads((run / "manifest.json").read_text())
    dest = run / (name + ".json")
    if dest.exists() or dest.with_suffix(".started").exists():
        raise ValueError("Refuse repeat profile")
    for rel, h in manifest["source_sha256"].items():
        if sha((ROOT / rel).read_bytes()) != h:
            raise ValueError("Frozen source changed: " + rel)
    if hybrid_service.status():
        raise RuntimeError("Owned normal gateway must be stopped")
    dest.with_suffix(".started").write_text(str(time.time()))
    data = ROOT / "local" / (run.name + "-" + name)
    holder = {}
    original = serving.Gateway
    server = serving.ThreadingHTTPServer

    class Observed(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["gw"] = self

    serving.Gateway = Observed
    serving.ThreadingHTTPServer = lambda addr, handler: server(
        addr, harness.isolated_handler(handler, harness.NONCE)
    )
    started = time.monotonic()
    thread = threading.Thread(target=serving.serve, args=(data, 8775), daemon=True)
    thread.start()
    report = {
        "profile": name,
        "config": PROFILES[name],
        "started": time.time(),
        "before_gpu": smi(),
        "cases": [],
        "failures": [],
        "resource_failures": [],
    }
    sampler = None
    log = None
    try:

        def ready():
            try:
                value = harness.api("/api/status?after=2147483647")
                if value.get("failure"):
                    raise RuntimeError(value["failure"])
                return value["ready"]
            except OSError:
                return False

        harness.wait_for(ready, 120, "native qualification")
        gw = holder["gw"]
        harness.api("/api/learning", {"enabled": False})
        report["boot_ready_seconds"] = time.monotonic() - started
        report["native_gate"] = gw.store.get("runtime", "native_gate")
        report["loaded_gpu"] = smi()
        report["loaded_sharedGPU"] = gw.monitor.sample()["gpu_shared"]["bytes"]
        gw.store.put(
            "runtime",
            "status",
            {
                "phase": "Native2B buffer profile: " + name,
                "note": "Actual isolated GPUtest; model weights/16K unchanged; normalchat restored afterwards",
            },
        )
        log = (run / (name + "-nvidia.csv")).open("w")
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                "-lms",
                "100",
            ],
            stdout=log,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if report["loaded_gpu"]["free_MiB"] < 512:
            report["resource_failures"].append(
                "Post-load freeVRAM below512MiB; no larger inputs attempted"
            )
        else:
            for rep in range(3):
                for case in json.loads(
                    (run / "cases.json").read_text(encoding="utf-8")
                ):
                    content = case["question"]
                    if case.get("image"):
                        p = (
                            run / "large-scene.png"
                            if case["image"] == "large-scene.png"
                            else data / "samples" / case["image"]
                        )
                        content = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(p.read_bytes()).decode()
                                },
                            },
                            {"type": "text", "text": content},
                        ]
                    body = {
                        "model": "scarx-hybrid",
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 96,
                        "stream": True,
                    }
                    if case.get("expected_json"):
                        body["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "final_setting",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "code": {"type": "string"},
                                        "port": {"type": "integer"},
                                    },
                                    "required": ["code", "port"],
                                    "additionalProperties": False,
                                },
                            },
                        }
                    response = harness.api("/v1/chat/completions", body, stream=True)
                    answer = response["answer"].strip()
                    correct = answer.casefold() == case.get("expected", "").casefold()
                    if case.get("expected_json"):
                        try:
                            correct = json.loads(answer) == case["expected_json"]
                        except ValueError:
                            correct = False
                    row = {
                        "id": case["id"],
                        "repeat": rep,
                        "question": case["question"],
                        "response": response,
                        "correct": correct,
                        "gpu": smi(),
                        "resources": gw.monitor.sample(),
                    }
                    report["cases"].append(row)
                    gw.store.event(
                        "buffer_profile.actual_answer",
                        dict(profile=name, **row),
                        "orchestrator",
                    )
                    print(
                        encode(
                            {
                                "profile": name,
                                "case": case["id"],
                                "repeat": rep,
                                "correct": correct,
                                "free_MiB": row["gpu"]["free_MiB"],
                            }
                        ),
                        flush=True,
                    )
                    if row["gpu"]["free_MiB"] < 512:
                        report["resource_failures"].append(
                            "After " + case["id"] + " freeVRAM below512MiB"
                        )
                        break
                if report["resource_failures"]:
                    break
            if not report["resource_failures"]:
                before = smi()
                t = time.monotonic()
                transcript = gw.backend.transcribe(data / "samples/fleurs-vi.wav")
                report["ASR"] = {
                    "before_gpu": before,
                    "seconds": time.monotonic() - t,
                    "text": transcript,
                    "after_gpu": smi(),
                    "VLM_released": not gw.engine.cognition.resident(),
                }
                report["after_ASR_wake"] = harness.ask("Hello")
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        if sampler:
            sampler.terminate()
            sampler.wait(5)
            log.close()
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
        p = run / (name + "-nvidia.csv")
        samples = list(csv.reader(p.read_text().splitlines())) if p.exists() else []
        report["sampled_min_free_MiB"] = min(
            [int(x[3]) for x in samples if len(x) == 4]
            + [report.get("loaded_gpu", report["before_gpu"])["free_MiB"]]
        )
        if report["sampled_min_free_MiB"] < 512 and not report["resource_failures"]:
            report["resource_failures"].append("Transient sampled reserve violation")
        shared_spike = report.get("resources", {}).get("gpu_shared", {}).get(
            "bytes", 0
        ) - report.get("loaded_sharedGPU", 0)
        report["shared_gpu_delta_bytes"] = shared_spike
        report["technical_candidate"] = (
            not report["failures"]
            and not report["resource_failures"]
            and len(report["cases"]) == 18
            and all(r["correct"] for r in report["cases"])
            and "ASR" in report
            and shared_spike <= 256 * 1024**2
        )
        dest.write_text(encode(report), encoding="utf-8")
        print(
            encode(
                {
                    "profile": name,
                    "technical_candidate": report["technical_candidate"],
                    "failures": report["failures"],
                    "resources_failed": report["resource_failures"],
                    "min_free_MiB": report["sampled_min_free_MiB"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", type=Path)
    p.add_argument("--profile", choices=list(PROFILES))
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.run and a.profile:
        arm(a.run.resolve(), a.profile)
    else:
        p.error("--prepare or --run+--profile required")
