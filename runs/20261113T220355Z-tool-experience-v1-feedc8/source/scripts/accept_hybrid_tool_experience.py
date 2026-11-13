"""Actual isolated gateway/ALIF + real Hermes tools, persistence and feedback path."""

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
import hybrid_cognition_acceptance as harness
import hybrid_service


def prepare():
    name = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-tool-experience-v1-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / name
    data = ROOT / "local" / name
    run.mkdir()
    data.mkdir()
    source = ROOT / "local/hybrid-v1"
    for folder in ["runtime", "models/native"]:
        shutil.copytree(source / folder, data / folder, copy_function=os.link)
    for f in ["native-integrity.json", "original-reference.json"]:
        shutil.copyfile(source / f, data / f)
    (data / "config.json").write_text(
        encode(
            {
                "execution_mode": "gpu",
                "searxng_enabled": False,
                "gpu_keep_warm": True,
                "review_quiet_seconds": 10,
                "review_interval_seconds": 30,
            }
        ),
        encoding="utf-8",
    )
    hashes = {}
    paths = list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        Path(__file__),
        ROOT / "scripts/hermes_tool_experience_fixture.py",
        ROOT / "experiments/hybrid-tool-experience-loop-engineering-v1.md",
        ROOT / "integrations/hermes/hybrid_plugin/learning_hooks.py",
    ]
    for path in paths:
        rel = path.relative_to(ROOT)
        out = run / "source" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = path.read_bytes()
        out.write_bytes(raw)
        hashes[rel.as_posix()] = hashlib.sha256(raw).hexdigest()
    (run / "manifest.json").write_text(
        encode(
            {
                "prepared": time.time(),
                "source_sha256": hashes,
                "data": str(data),
                "scope": "engineering feedback and actual tool observation; no improvement/retention claim",
                "max_seconds": 180,
                "production_weights_untouched": True,
            }
        ),
        encoding="utf-8",
    )
    (ROOT / "local/current-tool-experience-run.txt").write_text(str(run))
    print(run)


def execute(run):
    if hybrid_service.status():
        raise RuntimeError("Stop normal owner first")
    if (run / "started.json").exists():
        raise ValueError("Refuse rerun")
    manifest = json.loads((run / "manifest.json").read_text())
    for rel, h in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != h:
            raise ValueError("Frozen source changed: " + rel)
    (run / "started.json").write_text(str(time.time()))
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
    thread = threading.Thread(
        target=serving.serve, args=(Path(manifest["data"]), 8775), daemon=True
    )
    thread.start()
    report = {"started": time.time(), "scope": manifest["scope"], "failures": []}
    try:

        def ready():
            try:
                s = harness.api("/api/status?after=2147483647")
                if s.get("failure"):
                    raise RuntimeError(s["failure"])
                return s["ready"]
            except OSError:
                return False

        harness.wait_for(ready, 90, "native qualification")
        gw = holder["gw"]
        e = gw.engine
        harness.api("/api/learning", {"enabled": False})
        before = {
            k: hashlib.sha256(
                e.model.host(getattr(e.model.core, k)).tobytes()
            ).hexdigest()
            for k in e.model.core.weight_names
        }
        log = run / "hermes-fixture.log"
        with log.open("w", encoding="utf-8") as f:
            process = subprocess.run(
                [
                    str(ROOT / "local/hermes-reliability-test-env/Scripts/python.exe"),
                    str(ROOT / "scripts/hermes_tool_experience_fixture.py"),
                    str(run / "actual-tool-results.json"),
                ],
                cwd=ROOT,
                env=dict(os.environ, SCARX_AUDIT_NONCE=harness.NONCE),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        actual = json.loads(
            (run / "actual-tool-results.json").read_text(encoding="utf-8")
        )
        assert not actual["errors"] and len(actual["results"]) == 3
        incidents = e.incidents.summary()["recent"]
        assert len(incidents) == 1, "Expected exact route-scoped incident"
        report["incident"] = incidents[0]
        report["actual_missing_file"] = actual["actual_file_absent"]
        report["weights_unchanged_by_observations"] = all(
            hashlib.sha256(e.model.host(getattr(e.model.core, k)).tobytes()).hexdigest()
            == v
            for k, v in before.items()
        )
        assert report["weights_unchanged_by_observations"] and e.model.updates == 0
        report["created_goals"] = len(e.state["goals"])
        assert report["created_goals"] == 1
        harness.api("/api/learning", {"enabled": True})
        harness.wait_for(
            lambda: e.state["question"],
            40,
            "Real idle policy reaches owner clarification",
        )
        report["question_queued"] = e.state["question"]
        report["incident_after_policy"] = e.incidents.summary()["recent"][0]
        assert e.model.updates == 0
        harness.api("/api/learning", {"enabled": False})
        harness.wait_for(
            lambda: not e.cognition.work_active.is_set(), 5, "quiescent controller"
        )
        pin = harness.api("/api/save", {})["id"]
        payload = gw.store.snapshot(pin)
        response = harness.api(
            "/api/feedback",
            {
                "id": e.state["question"]["id"],
                "text": "Engineering fixture owner signal: this clarification was not useful.",
                "helpful": False,
            },
        )
        # This is an injected explicit feedback signal for API plumbing, not a
        # real user's judgement or evidence of autonomous root-cause learning.
        report["injected_negative_feedback"] = response
        report["negative_signal_source"] = (
            "engineering fixture, not actual owner judgement"
        )
        report["updates_after_fixture"] = e.model.updates
        harness.api("/api/rollback", {"id": pin})
        with e.lock:
            assert e.model.dumps(e.state) == payload
        report["full_rollback_exact"] = True
        report["incident_audit_survives"] = (
            e.incidents.summary()["recent"][0]["last_feedback"]["kind"]
            == "explicit_owner_feedback"
        )
        report["restored_updates"] = e.model.updates
        report["actual_streamed_answer"] = harness.ask("Hello")
        report["native_gate"] = gw.store.get("runtime", "native_gate")
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        if "gw" in holder:
            gw = holder["gw"]
            report["resources"] = gw.monitor.sample()
            events = [
                v
                for v in harness.all_events(gw.store)
                if v["kind"].startswith(
                    (
                        "tool_incident",
                        "tool_experience",
                        "autonomous_action",
                        "local_learning_update",
                        "cognitive_",
                    )
                )
            ]
            (run / "events.json").write_text(encode(events), encoding="utf-8")
            try:
                harness.api("/api/shutdown", {})
            except Exception as exc:
                report["failures"].append("shutdown: " + str(exc))
            thread.join(30)
            report["gateway_stopped"] = not thread.is_alive()
        report["seconds"] = time.monotonic() - started
        (run / "result.json").write_text(encode(report), encoding="utf-8")
        print(
            encode(
                {
                    "receipt": str(run / "result.json"),
                    "failures": report["failures"],
                    "seconds": report["seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.run:
        execute(a.run.resolve())
    else:
        p.error("--prepare or --run required")
