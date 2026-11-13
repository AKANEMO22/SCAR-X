"""Matched frozen VLM engineering and mixed GPU headroom, one native candidate."""

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

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from scarx.hybrid import gateway as serving
from scarx.hybrid.store import encode
import hybrid_cognition_acceptance as harness
from probe_hybrid_qwen2b import smi
import hybrid_service

IDENTITY = {
    "name": "LiquidAI/LFM2.5-VL-1.6B",
    "revision": "919fde3d022e3f90a4716006f993938ee8c2eb97",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(run):
    from PIL import Image, ImageDraw

    if (run / "paired-lock.json").exists():
        raise ValueError("Probe already prepared")
    root = ROOT / "local/hybrid-v1"
    candidate = root / "candidates/lfm2.5-vl-1.6b"
    reference = json.loads(
        (run / "reference-keymap-v2.json").read_text(encoding="utf-8")
    )
    conversion = json.loads((run / "conversion.json").read_text(encoding="utf-8"))
    if (
        reference["status"] != "REFERENCE_READY"
        or reference["mismatches"]
        or reference["unused_tensors"]
    ):
        raise ValueError("Reference integrity gate not met")
    for name in ["baseline", "candidate"]:
        data = ROOT / "local" / (run.name + "-" + name)
        data.mkdir()
        shutil.copytree(root / "runtime", data / "runtime", copy_function=os.link)
        shutil.copytree(root / "samples", data / "samples", copy_function=os.link)
        (data / "models/native").mkdir(parents=True)
        original = json.loads(
            (root / "native-integrity.json").read_text(encoding="utf-8")
        )
        exports = {}
        if name == "baseline":
            for filename in ["qwen-native-bf16.gguf", "qwen-vision-native-bf16.gguf"]:
                os.link(
                    root / "models/native" / filename, data / "models/native" / filename
                )
                exports[filename] = original["exports"][filename]
            shutil.copyfile(
                root / "original-reference.json", data / "original-reference.json"
            )
            config = {
                "execution_mode": "gpu",
                "searxng_enabled": False,
                "gpu_keep_warm": True,
            }
        else:
            for src, dest in [
                ("model-native-bf16.gguf", "qwen-native-bf16.gguf"),
                ("vision-native-bf16.gguf", "qwen-vision-native-bf16.gguf"),
            ]:
                os.link(candidate / "native" / src, data / "models/native" / dest)
                exports[dest] = {
                    k: conversion["exports"][src][k] for k in ["bytes", "sha256"]
                }
            (data / "original-reference.json").write_text(
                encode(reference), encoding="utf-8"
            )
            config = {
                "execution_mode": "gpu",
                "searxng_enabled": False,
                "gpu_keep_warm": True,
                "model_name": IDENTITY["name"],
                "model_revision": IDENTITY["revision"],
            }
        filename = "whisper-base-f32.bin"
        os.link(root / "models/native" / filename, data / "models/native" / filename)
        exports[filename] = original["exports"][filename]
        integrity = {
            "status": "NATIVE_STORAGE_INTEGRITY_PASS",
            "run_id": run.name,
            "quantized": False,
            "exports": exports,
        }
        if name == "candidate":
            integrity["model_identity"] = IDENTITY
        (data / "native-integrity.json").write_text(encode(integrity), encoding="utf-8")
        (data / "config.json").write_text(encode(config), encoding="utf-8")
    image = Image.new("RGB", (1920, 1080), "#101820")
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 230, 700, 850), fill="red")
    draw.ellipse((1100, 240, 1700, 840), fill="blue")
    image.save(run / "large-scene.png")
    filler = " ".join(
        f"Unrelated record{i}: shelf{i%17}, no target-setting change."
        for i in range(80)
    )
    cases = [
        {
            "id": "math_en",
            "language": "en",
            "question": "Compute17+25. Return only the integer.",
            "expected": "42",
        },
        {
            "id": "math_vi",
            "language": "vi",
            "question": "Tính17+25. Chỉ trả lời số nguyên.",
            "expected": "42",
        },
        {
            "id": "instruction_vi",
            "language": "vi",
            "question": "Chỉ viết đúng một từ: xong",
            "expected": "xong",
        },
        {
            "id": "setting_en",
            "language": "en",
            "question": "The port was8123, later changed to9237. Return only its current value.",
            "expected": "9237",
        },
        {
            "id": "setting_vi",
            "language": "vi",
            "question": "Cổng ban đầu8123, sau đó đổi thành9237. Chỉ trả về cổng hiện tại.",
            "expected": "9237",
        },
        {
            "id": "distractor",
            "language": "en",
            "question": "Target port starts8123. "
            + filler
            + " Final approved update: target port9237. Return only the final target port.",
            "expected": "9237",
        },
        {
            "id": "strict_json",
            "language": "en",
            "question": "Project code JADE-472, port starts8123 then changes to9237. Return final state as JSON.",
            "expected_json": {"code": "JADE-472", "port": 9237},
            "schema": {
                "type": "object",
                "properties": {"code": {"type": "string"}, "port": {"type": "integer"}},
                "required": ["code", "port"],
                "additionalProperties": False,
            },
        },
        {
            "id": "long_compact_regression",
            "language": "en",
            "question": "CodeJADE-472 and port8123 were recorded. "
            + filler
            + " A later correction changes only port to9237. Return only the current port.",
            "expected": "9237",
        },
        {
            "id": "missing_fact",
            "language": "en",
            "question": "The source says Niva carries a green bag. No birthdate was supplied. Return JSON with birth_year null.",
            "expected_json": {"birth_year": None},
            "schema": {
                "type": "object",
                "properties": {"birth_year": {"type": "null"}},
                "required": ["birth_year"],
                "additionalProperties": False,
            },
        },
        {
            "id": "small_vision",
            "language": "en",
            "question": "What color is the circle on the right? Return one color word.",
            "expected": "blue",
            "image": "scene-first.png",
        },
        {
            "id": "large_vision",
            "language": "en",
            "question": "What color is the circle on the right? Return one color word.",
            "expected": "blue",
            "image": "large-scene.png",
        },
        {
            "id": "temporal_vi",
            "language": "vi",
            "question": "Lan đặt chìa khóa vào hộp đỏ. Sau đó Minh chuyển chìa khóa sang hộp xanh. Chìa khóa hiện ở hộp nào? Chỉ trả lời một từ: đỏ hoặc xanh.",
            "expected": "xanh",
        },
    ]
    (run / "paired-cases.json").write_text(encode(cases), encoding="utf-8")
    hashes = {}
    for path in list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        Path(__file__),
        ROOT / "experiments/hybrid-native-lfm16-vram-v1.md",
    ]:
        rel = path.relative_to(ROOT)
        p = run / "paired-source" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(path.read_bytes())
        hashes[rel.as_posix()] = sha(path.read_bytes())
    (run / "paired-lock.json").write_text(
        encode(
            {
                "prepared": time.time(),
                "cases_sha256": sha((run / "paired-cases.json").read_bytes()),
                "source_sha256": hashes,
                "identity": IDENTITY,
                "native_context_tokens": 16384,
                "output_budget": 96,
                "global_reserve_MiB": 512,
                "one_configuration": True,
                "no_live_promotion": True,
            }
        ),
        encoding="utf-8",
    )
    print(encode({"paired_prepared": str(run)}), flush=True)


def run_arm(run, arm):
    lock = json.loads((run / "paired-lock.json").read_text())
    dest = run / ("paired-" + arm + ".json")
    if dest.exists() or dest.with_suffix(".started").exists():
        raise ValueError("Refuse repeat candidate/configuration")
    for rel, h in lock["source_sha256"].items():
        if sha((ROOT / rel).read_bytes()) != h:
            raise ValueError("Frozen source changed: " + rel)
    if hybrid_service.status():
        raise RuntimeError("Stop normal service first")
    dest.with_suffix(".started").write_text(str(time.time()))
    data = ROOT / "local" / (run.name + "-" + arm)
    holder = {}
    original = serving.Gateway
    server = serving.ThreadingHTTPServer

    class Observed(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            holder["gw"] = self

    serving.Gateway = Observed
    serving.ThreadingHTTPServer = lambda addr, handler: server(
        addr, harness.isolated_handler(handler, harness.NONCE)
    )
    started = time.monotonic()
    thread = threading.Thread(target=serving.serve, args=(data, 8775), daemon=True)
    thread.start()
    report = {
        "started": time.time(),
        "arm": arm,
        "cases": [],
        "cycles": [],
        "failures": [],
        "resource_failures": [],
        "live_promoted": False,
        "before_gpu": smi(),
    }
    sampler = None
    log = None

    def ask_case(case):
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
            "stream": True,
            "max_tokens": 96,
        }
        if case.get("schema"):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "case_output",
                    "strict": True,
                    "schema": case["schema"],
                },
            }
        result = harness.api("/v1/chat/completions", body, stream=True)
        answer = result["answer"].strip()
        correct = answer.casefold() == case.get("expected", "").casefold()
        if "expected_json" in case:
            try:
                correct = json.loads(answer) == case["expected_json"]
            except ValueError:
                correct = False
        return {
            "id": case["id"],
            "language": case["language"],
            "question": case["question"],
            "response": result,
            "correct": correct,
            "gpu": smi(),
        }

    try:

        def ready():
            try:
                s = harness.api("/api/status?after=2147483647")
                if s.get("failure"):
                    raise RuntimeError(s["failure"])
                return s["ready"]
            except OSError:
                return False

        harness.wait_for(ready, 120, "native gate")
        gw = holder["gw"]
        harness.api("/api/learning", {"enabled": False})
        report["boot_ready_seconds"] = time.monotonic() - started
        report["native_gate"] = gw.store.get("runtime", "native_gate")
        report["identity"] = gw.status()["VLM_identity"]
        report["loaded_gpu"] = smi()
        report["loaded_shared"] = gw.monitor.sample()["gpu_shared"]["bytes"]
        gw.store.put(
            "runtime",
            "status",
            {
                "phase": "Paired native VLM engineering: " + arm,
                "note": "ActualVi/En/image/ASR tests; frozenweights, no neural-policy promotion.",
            },
        )
        log = (run / ("paired-" + arm + "-gpu.csv")).open("w")
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
            raise MemoryError("Postload512MiB reserve failed")
        for case in json.loads((run / "paired-cases.json").read_text(encoding="utf-8")):
            row = ask_case(case)
            report["cases"].append(row)
            gw.store.event(
                "nativeVLM.actual_answer", dict(arm=arm, **row), "orchestrator"
            )
            print(
                encode(
                    {
                        "arm": arm,
                        "case": row["id"],
                        "correct": row["correct"],
                        "free": row["gpu"]["free_MiB"],
                    }
                ),
                flush=True,
            )
            if row["gpu"]["free_MiB"] < 512:
                raise MemoryError("Reserve after " + case["id"])
        # Direct model tool transport is a distinct capability, not replaced by
        # the workspace runtime prelude or a fake successful tool result.
        tool = {
            "type": "function",
            "function": {
                "name": "read_project_file",
                "description": "Read a named local project file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
        result = harness.api(
            "/v1/chat/completions",
            {
                "model": "scarx-hybrid",
                "messages": [
                    {
                        "role": "user",
                        "content": "Use read_project_file to read README.md. Do not guess its content.",
                    }
                ],
                "tools": [tool],
                "max_tokens": 96,
                "stream": True,
            },
            stream=True,
        )
        calls = [
            c
            for chunk in result.get("chunks", [])
            for choice in chunk.get("choices", [])
            for c in choice.get("delta", {}).get("tool_calls", [])
        ]
        report["tool_transport"] = {"raw": result, "tool_calls_emitted": bool(calls)}
        for i in range(3):
            t = time.monotonic()
            before = smi()
            text = gw.backend.transcribe(data / "samples/fleurs-vi.wav")
            released = not gw.engine.cognition.resident()
            answer = harness.ask("Hello")
            report["cycles"].append(
                {
                    "iteration": i,
                    "ASR_text": text,
                    "seconds": time.monotonic() - t,
                    "before_gpu": before,
                    "after_gpu": smi(),
                    "VLM_released_for_audio": released,
                    "answer": answer,
                    "neural_device": gw.engine.model.device,
                }
            )
        pin = harness.api("/api/save", {})["id"]
        payload = gw.store.snapshot(pin)
        harness.wait_for(
            lambda: not gw.engine.cognition.work_active.is_set(), 5, "quiescent learner"
        )
        harness.api("/api/rollback", {"id": pin})
        report["full_state_rollback_exact"] = (
            gw.engine.model.dumps(gw.engine.state) == payload
        )
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
        csvpath = run / ("paired-" + arm + "-gpu.csv")
        rows = (
            list(csv.reader(csvpath.read_text().splitlines()))
            if csvpath.exists()
            else []
        )
        report["sampled_min_free_MiB"] = min(
            [int(r[3]) for r in rows if len(r) == 4]
            + [report.get("loaded_gpu", report["before_gpu"])["free_MiB"]]
        )
        report["score"] = sum(c["correct"] for c in report["cases"])
        report["Vi_score"] = sum(
            c["correct"] for c in report["cases"] if c["language"] == "vi"
        )
        report["seconds"] = time.monotonic() - started
        report["finished"] = time.time()
        dest.write_text(encode(report), encoding="utf-8")
        print(
            encode(
                {
                    "arm": arm,
                    "score": report["score"],
                    "Vi_score": report["Vi_score"],
                    "failures": report["failures"],
                    "min_free_MiB": report["sampled_min_free_MiB"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--arm", choices=["baseline", "candidate"])
    a = p.parse_args()
    run = a.run.resolve()
    if ROOT / "runs" not in run.parents:
        raise ValueError("External run")
    if a.prepare:
        prepare(run)
    else:
        run_arm(run, a.arm)
