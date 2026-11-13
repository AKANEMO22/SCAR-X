"""Real isolated mixed workload, streamed at localhost, with bounded duration.

Production restoration belongs to the outer shell, never this process's job.
"""

import argparse
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
from scarx.hybrid.learner import LocalALIF
import hybrid_service
import hybrid_cognition_acceptance as harness


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(engine):
    with engine.lock:
        return {
            "arrays": {
                k: sha(engine.model.host(a).tobytes())
                for k, a in engine.model.arrays().items()
            },
            "rng": engine.model.rng.bit_generator.state,
            "signal_rng": engine.model.signal_rng.bit_generator.state,
            "replay": engine.model.replay,
            "replay_seen": engine.model.replay_seen,
            "updates": engine.model.updates,
            "hidden_updates": engine.model.hidden_updates,
            "review": json.loads(encode(engine.state["review"])),
            "replay_used": engine.state["cognition"]["replay_used"],
            "replay_revision": engine.state["cognition"]["replay_revision"],
        }


def prepare():
    name = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-hybrid-reliability-v1-"
        + uuid.uuid4().hex[:6]
    )
    run, data = ROOT / "runs" / name, ROOT / "local" / name
    run.mkdir()
    data.mkdir()
    source = ROOT / "local/hybrid-v1"
    shutil.copytree(source / "runtime", data / "runtime", copy_function=os.link)
    (data / "models/native").mkdir(parents=True)
    for p in (source / "models/native").iterdir():
        if p.is_file():
            os.link(p, data / "models/native" / p.name)
    for name in ("native-integrity.json", "original-reference.json"):
        shutil.copyfile(source / name, data / name)
    shutil.copytree(source / "samples", data / "samples", copy_function=os.link)
    config = {
        "execution_mode": "gpu",
        "searxng_enabled": False,
        "review_quiet_seconds": 10,
        "review_interval_seconds": 30,
        "consolidation_after_seconds": 10,
        "consolidation_interval_seconds": 2,
        "sleep_replay_budget": 2,
        "rest_after_seconds": 20,
        "gpu_keep_warm": True,
    }
    (data / "config.json").write_text(encode(config), encoding="utf-8")
    fixture = data / "fixture-project"
    fixture.mkdir()
    (fixture / "README.md").write_text(
        "# Jade inventory\nThis local program sums inventory stock counts.\nConfiguration port: 9237.\n",
        encoding="utf-8",
    )
    (fixture / "main.py").write_text(
        'def total_stock(items):\n    return sum(item["count"] for item in items)\n',
        encoding="utf-8",
    )
    jobs = []
    prompts = [
        (
            "workspace-vi",
            "Phân tích workspace này làm gì. Đọc README.md và main.py; chỉ đọc, không sửa file hoặc tìm trên web.",
            ["9237"],
        ),
        (
            "workspace-en",
            "Inspect this project folder. Read README.md and main.py and explain the purpose briefly. Do not search the web or edit files.",
            ["9237"],
        ),
        ("workspace-short", "đọc folder này xem nó làm gì", ["9237"]),
        (
            "actual-project",
            "đọc folder app z fold xem nó làm gì. Chỉ đọc các tài liệu và file nguồn cục bộ, không sửa, không chạy code dự án, không tìm web.",
            [],
        ),
    ]
    for ident, prompt, expected in prompts:
        cwd = (
            Path("C:/Users/hachimi/Downloads/app z fold")
            if ident == "actual-project"
            else fixture
        )
        job = {
            "id": ident,
            "cwd": str(cwd),
            "home": str(data / ("home-" + ident)),
            "hermes_source": str(ROOT / "local/hermes-reliability-source"),
            "prompt": prompt,
            "expected_terms": expected,
        }
        (run / (ident + ".job.json")).write_text(encode(job), encoding="utf-8")
        jobs.append(ident)
    job = {
        "id": "compression",
        "operation": "compression",
        "cwd": str(fixture),
        "home": str(data / "home-compression"),
        "hermes_source": str(ROOT / "local/hermes-reliability-source"),
    }
    (run / "compression.job.json").write_text(encode(job), encoding="utf-8")
    jobs.append("compression")
    sources = list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        ROOT / "scripts/hybrid_reliability_soak.py",
        ROOT / "scripts/hybrid_agent_workload.py",
        ROOT / "scripts/hybrid_cognition_acceptance.py",
        ROOT / "experiments/hybrid-workspace-gpu-reliability-v1.md",
    ]
    hashes = {}
    for p in sources:
        rel = p.relative_to(ROOT)
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        hashes[rel.as_posix()] = sha(p.read_bytes())
    for rel in [
        "agent/agent_init.py",
        "agent/conversation_compression.py",
        "tools/tool_search.py",
    ]:
        p = ROOT / "local/hermes-reliability-source" / rel
        dest = run / "source/hermes" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    manifest = {
        "run_id": run.name,
        "data_dir": str(data),
        "prepared": time.time(),
        "parent_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config": config,
        "duration_seconds": 1800,
        "jobs": jobs,
        "source_sha256": hashes,
        "isolation": "fresh learning datastore, all HTTP mutations nonce-gated; actual project read-only; no private transcripts exported",
        "scope": "engineering reliability, not lifelong/LoRA/quality confirmation",
        "restore": "outer shell finally only",
    }
    (run / "manifest.json").write_text(encode(manifest), encoding="utf-8")
    (ROOT / "local/current-reliability-run.txt").write_text(str(run), encoding="utf-8")
    print(encode({"prepared": str(run)}), flush=True)


def run_workload(run, manifest, resume=False, cpu=False):
    if hybrid_service.status():
        raise RuntimeError("Stop owned gateway before isolated run")
    for rel, digest in manifest["source_sha256"].items():
        if sha((ROOT / rel).read_bytes()) != digest:
            raise ValueError("Frozen source changed: " + rel)
    dest = run / ("cpu.json" if cpu else "resume.json" if resume else "soak.json")
    if dest.exists() or dest.with_suffix(".started").exists():
        raise ValueError("Refuse rerun/overwrite")
    dest.with_suffix(".started").write_text(str(time.time()))
    data = Path(manifest["data_dir"])
    if cpu:
        config = dict(manifest["config"], execution_mode="cpu")
        (data / "config.json").write_text(encode(config), encoding="utf-8")
    elif resume:
        (data / "config.json").write_text(encode(manifest["config"]), encoding="utf-8")
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
    thread = threading.Thread(target=serving.serve, args=(data, 8775), daemon=True)
    thread.start()
    started = time.monotonic()
    report = {
        "started": time.time(),
        "resume": resume,
        "cpu": cpu,
        "cases": [],
        "failures": [],
    }
    gpu_log = None
    gpu_sampler = None

    def record(name, fn):
        gw = holder["gw"]
        gw.store.event(
            "reliability.started", {"case": name, "run_id": run.name}, "orchestrator"
        )
        t = time.monotonic()
        try:
            value = fn()
            row = {
                "case": name,
                "completed": True,
                "result": value,
                "seconds": time.monotonic() - t,
            }
        except Exception as exc:
            row = {
                "case": name,
                "completed": False,
                "error": type(exc).__name__ + ": " + str(exc),
                "seconds": time.monotonic() - t,
            }
            report["failures"].append(row["error"])
        report["cases"].append(row)
        gw.store.event("reliability.finished", row, "orchestrator")
        with (
            run
            / (
                "cpu.progress.jsonl"
                if cpu
                else "resume.progress.jsonl" if resume else "soak.progress.jsonl"
            )
        ).open("a", encoding="utf-8") as f:
            f.write(encode(row) + "\n")
        print(encode({k: v for k, v in row.items() if k != "result"}), flush=True)
        return row.get("result")

    try:

        def ready():
            try:
                s = harness.api("/api/status?after=2147483647")
                if s.get("failure"):
                    raise RuntimeError(s["failure"])
                return s.get("ready")
            except OSError:
                return False

        harness.wait_for(ready, 90, "native qualification")
        gw = holder["gw"]
        e = gw.engine
        report["boot_ready_seconds"] = time.monotonic() - started
        report["placement"] = gw.status()["execution_placement"]
        gpu_log = (
            run
            / (
                "nvidia-cpu.csv"
                if cpu
                else "nvidia-resume.csv" if resume else "nvidia-soak.csv"
            )
        ).open("w")
        gpu_sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-lms",
                "200",
            ],
            stdout=gpu_log,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        def text(prompt):
            return harness.ask(prompt)

        if resume or cpu:
            expected = json.loads((run / "soak.json").read_text(encoding="utf-8"))

            def cold():
                assert (
                    fingerprint(e) == expected["fingerprint"]
                ), "Cold full neural/controller state changed"
                assert (
                    sha(gw.store.snapshot(expected["pinned"]["id"]))
                    == expected["pinned"]["sha256"]
                )
                return {
                    "neural_rng_replay_review_exact": True,
                    "device": e.model.device,
                    "updates": e.model.updates,
                }

            record("cold_state", cold)
            record("real_answer", lambda: text("Hello"))
            if cpu:
                record(
                    "cpu_ASR",
                    lambda: {
                        "text": gw.backend.transcribe(data / "samples/fleurs-vi.wav"),
                        "mode": gw.config.execution_mode,
                    },
                )
        else:
            record("warm_answer", lambda: text("Hello"))

            def job(ident):
                output = (data if ident == "actual-project" else run) / (
                    ident + ".result.json"
                )
                log = (data if ident == "actual-project" else run) / (
                    ident + ".process.log"
                )
                env = dict(os.environ, SCARX_AUDIT_NONCE=harness.NONCE)
                with log.open("w", encoding="utf-8") as f:
                    p = subprocess.run(
                        [
                            str(
                                ROOT
                                / "local/hermes-reliability-test-env/Scripts/python.exe"
                            ),
                            str(ROOT / "scripts/hybrid_agent_workload.py"),
                            "--job",
                            str(run / (ident + ".job.json")),
                            "--out",
                            str(output),
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=95,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                r = json.loads(output.read_text(encoding="utf-8"))
                gw.store.event(
                    "reliability.actual_agent_answer",
                    {
                        "job": ident,
                        "answer": r.get("answer"),
                        "tool_calls": r.get("tool_calls"),
                        "completed": r.get("completed"),
                        "local_read_used": r.get("local_read_used"),
                        "web_used": r.get("web_used"),
                        "scope": "real Hermes/VLM result; engineering request",
                    },
                    "orchestrator",
                )
                result = {
                    k: v
                    for k, v in r.items()
                    if k not in ("transcript", "compacted_history")
                }
                if ident == "actual-project":
                    result.pop("answer", None)
                    result.update(
                        private_output=str(output),
                        private_output_sha256=sha(output.read_bytes()),
                    )
                return result

            for ident in manifest["jobs"]:
                record("Hermes_" + ident, lambda ident=ident: job(ident))
            if report["failures"]:
                raise RuntimeError(
                    "Prerequisite agent jobs failed; do not spend endurance budget"
                )
            record(
                "GPU_ASR",
                lambda: {
                    "text": gw.backend.transcribe(data / "samples/fleurs-vi.wav"),
                    "mode": gw.config.execution_mode,
                },
            )

            def learn():
                g = harness.api(
                    "/api/goal",
                    {"question": "12+3", "reason": "closed verifier in isolated soak"},
                )
                harness.wait_for(
                    lambda: e.state["goals"][g["id"]]["status"] == "resolved_verified",
                    45,
                    "idle verifier",
                )
                harness.wait_for(
                    lambda: e.state["cognition"]["replay_used"] == 2,
                    30,
                    "bounded replay",
                )
                return {
                    "updates": e.model.updates,
                    "hidden_updates": e.model.hidden_updates,
                    "device": e.model.device,
                    "replay_count": len(e.model.replay),
                    "replay_used": e.state["cognition"]["replay_used"],
                }

            record("automatic_ALIF_verify_and_replay", learn)

            def sleepwake():
                harness.api("/api/cognition", {"intent": "sleep"})
                harness.wait_for(
                    lambda: not e.cognition.resident()
                    and not e.cognition.work_active.is_set(),
                    10,
                    "VLM sleep",
                )
                before = gw.monitor.sample()
                answer = text("Hello")
                return {
                    "sleep_resources": before,
                    "cold_answer": answer,
                    "awake_resources": gw.monitor.sample(),
                }

            record("explicit_sleep_and_cold_wake", sleepwake)
            # Actual small requests throughout the elapsed interval. Idle gaps are
            # genuine scheduler time, not fabricated neuron activity.
            mixed_start = time.monotonic()
            deadline = mixed_start + manifest["duration_seconds"]
            n = 0
            while time.monotonic() < deadline:
                n += 1
                record(
                    "mixed_%03d" % n,
                    lambda: text(
                        "Hello" if n % 2 else "Return only the result of 12+3."
                    ),
                )
                if n % 10 == 0:
                    record("sleepwake_%03d" % n, sleepwake)
                with e.lock:
                    state = {
                        "elapsed_seconds": time.monotonic() - mixed_start,
                        "resources": gw.monitor.sample(),
                        "cognition": e.cognition.summary(),
                        "updates": e.model.updates,
                    }
                gw.store.event("reliability.soak_tick", state, "orchestrator")
                (run / "latest.json").write_text(encode(state), encoding="utf-8")
                remaining = min(40, deadline - time.monotonic())
                if remaining > 0:
                    time.sleep(remaining)
            report["mixed_elapsed_seconds"] = time.monotonic() - mixed_start
            report["mixed_requests"] = n
            harness.api("/api/learning", {"enabled": False})
            harness.api("/api/cognition", {"intent": "sleep"})
            harness.wait_for(
                lambda: not e.cognition.resident()
                and not e.cognition.work_active.is_set(),
                10,
                "quiescent checkpoint",
            )
            pin = harness.api("/api/save", {})["id"]
            payload = gw.store.snapshot(pin)
            harness.api("/api/rollback", {"id": pin})
            with e.lock:
                assert e.model.dumps(e.state) == payload
            report["pinned"] = {"id": pin, "sha256": sha(payload)}
            report["full_rollback_exact"] = True
            report["fingerprint"] = fingerprint(e)
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        if gpu_sampler:
            gpu_sampler.terminate()
            gpu_sampler.wait(5)
            gpu_log.close()
        if "gw" in holder:
            gw = holder["gw"]
            report["resources"] = gw.monitor.sample()
            report["native_gate"] = gw.store.get("runtime", "native_gate")
            # Keep detailed evidence in the isolated local datastore; whitelisted
            # engineering event summaries and agent fixture outputs are exportable.
            events = [
                v
                for v in harness.all_events(gw.store)
                if v["kind"].startswith(
                    (
                        "reliability.",
                        "sleep_replay_",
                        "autonomous_action",
                        "machine_verification",
                        "cognitive_",
                        "vlm_hibernated",
                        "vlm_resumed",
                    )
                )
            ]
            (
                run
                / (
                    "cpu.events.json"
                    if cpu
                    else "resume.events.json" if resume else "soak.events.json"
                )
            ).write_text(encode(events), encoding="utf-8")
            try:
                harness.api("/api/shutdown", {})
            except Exception as exc:
                report["failures"].append("shutdown: " + str(exc))
            thread.join(30)
            report["gateway_stopped"] = not thread.is_alive()
        report["seconds"] = time.monotonic() - started
        report["finished"] = time.time()
        dest.write_text(encode(report), encoding="utf-8")
        print(
            encode({"receipt": str(dest), "failures": report["failures"]}), flush=True
        )
    return int(bool(report["failures"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.run:
        run = a.run.resolve()
        if ROOT / "runs" not in run.parents:
            raise ValueError("External run path")
        sys.exit(
            run_workload(
                run,
                json.loads((run / "manifest.json").read_text(encoding="utf-8")),
                a.resume,
                a.cpu,
            )
        )
    else:
        p.error("--prepare or --run required")
