"""Single isolated, real GPU wake/sleep acceptance; never restores from a child job.

Prepare, stop normal service, run first+resume from separate Python processes,
then restore normal service from the OUTER shell's finally block.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from scarx.hybrid import gateway as serving
from scarx.hybrid.store import encode
from hybrid_pipeline_audit import isolated_handler
from hybrid_context_benchmark import consume
import hybrid_service

BASE = "http://127.0.0.1:8775"
NONCE = secrets.token_urlsafe(32)
PROTOCOL = ROOT / "experiments/hybrid-cognitive-loop-engineering-v1.md"
LAB = {
    "execution_mode": "gpu",
    "searxng_enabled": False,
    "review_quiet_seconds": 10,
    "review_interval_seconds": 30,
    "consolidation_after_seconds": 10,
    "consolidation_interval_seconds": 2,
    "sleep_replay_budget": 2,
    "rest_after_seconds": 20,
    "release_vlm_on_rest": True,
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def api(path, body=None, stream=False):
    headers = {"X-SCARX-Backend": "hermes", "X-SCARX-Audit": NONCE}
    if body is not None:
        token = api("/api/status?after=2147483647")["token"]
        headers |= {"X-SCARX-Token": token, "Content-Type": "application/json"}
    started = time.monotonic()
    with urlopen(
        Request(
            BASE + path,
            data=None if body is None else encode(body).encode(),
            headers=headers,
        ),
        timeout=90,
    ) as response:
        if not stream:
            return json.load(response)
        headers_at = time.monotonic() - started
        output = consume(response)
        first = output["first_after_headers_seconds"]
        output["ttft_seconds"] = None if first is None else headers_at + first
        output["total_seconds"] = time.monotonic() - started
        return output


def ask(question):
    return api(
        "/v1/chat/completions",
        {
            "model": "scarx-hybrid",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 96,
            "stream": True,
            "scarx_session_id": "cognition-engineering-isolated",
        },
        stream=True,
    )


def wait_for(predicate, timeout, label):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise TimeoutError(label)


def fingerprint(engine):
    with engine.lock:
        return {
            "arrays": {k: sha(v.tobytes()) for k, v in engine.model.arrays().items()},
            "rng": sha(encode(engine.model.rng.bit_generator.state).encode()),
            "replay": sha(encode(engine.model.replay).encode()),
            "replay_seen": engine.model.replay_seen,
            "updates": engine.model.updates,
            "hidden_updates": engine.model.hidden_updates,
            "review": engine.state["review"],
            "replay_used": engine.state["cognition"]["replay_used"],
            "replay_revision": engine.state["cognition"]["replay_revision"],
        }


def prepare():
    name = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-cognition-engineering-v1-"
        + uuid.uuid4().hex[:6]
    )
    run, data = ROOT / "runs" / name, ROOT / "local" / name
    run.mkdir()
    data.mkdir()
    source = ROOT / "local/hybrid-v1"
    shutil.copytree(source / "runtime", data / "runtime", copy_function=os.link)
    (data / "models/native").mkdir(parents=True)
    for path in (source / "models/native").iterdir():
        if path.is_file():
            os.link(path, data / "models/native" / path.name)
    for name in ("native-integrity.json", "original-reference.json"):
        shutil.copyfile(source / name, data / name)
    (data / "config.json").write_text(encode(LAB), encoding="utf-8")
    files = list((ROOT / "src/scarx/hybrid").glob("*.py"))
    files += [
        ROOT / "src/scarx/hybrid/web.html",
        Path(__file__),
        PROTOCOL,
        ROOT / "scripts/hybrid_pipeline_audit.py",
        ROOT / "scripts/hybrid_context_benchmark.py",
    ]
    hashes = {}
    for path in files:
        rel = path.relative_to(ROOT)
        target = run / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[rel.as_posix()] = sha(path.read_bytes())
    manifest = {
        "run_id": run.name,
        "data_dir": str(data),
        "lab_config": LAB,
        "parent_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": hashes,
        "prepared": time.time(),
        "model_integrity": json.loads(
            (source / "native-integrity.json").read_text(encoding="utf-8")
        ),
        "scope": "engineering lifecycle; no language/retention/hidden-advantage quality claim",
        "isolation": "fresh datastore, no private chat copy; nonce gates all POSTs; source models read-only hardlinks",
        "restore": "outer shell only; do not spawn production as child of this job-owned process",
        "ponytail": "not available in tool inventory",
    }
    (run / "manifest.json").write_text(encode(manifest), encoding="utf-8")
    print(encode({"run": str(run)}), flush=True)


def execute(run, resume):
    if ROOT / "runs" not in run.parents:
        raise ValueError("Run outside workspace")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["source_sha256"].items():
        if sha((ROOT / rel).read_bytes()) != expected:
            raise ValueError("Frozen source changed: " + rel)
    dest = run / ("after_restart.json" if resume else "before_restart.json")
    if dest.exists() or dest.with_suffix(".started").exists():
        raise ValueError("Refusing rerun/overwrite")
    if hybrid_service.status():
        raise RuntimeError("Stop owned normal gateway before isolated acceptance")
    dest.with_suffix(".started").write_text(str(time.time()))
    holder = {}
    original, original_server = serving.Gateway, serving.ThreadingHTTPServer

    class ObservedGateway(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            holder["gateway"] = self

    serving.Gateway = ObservedGateway
    serving.ThreadingHTTPServer = lambda address, handler: original_server(
        address, isolated_handler(handler, NONCE)
    )
    thread = threading.Thread(
        target=serving.serve, args=(Path(manifest["data_dir"]), 8775), daemon=True
    )
    thread.start()
    report = {
        "run_id": run.name,
        "started": time.time(),
        "resume": resume,
        "cases": [],
        "failures": [],
    }

    def record(name, fn):
        gw = holder["gateway"]
        gw.store.event(
            "cognition_acceptance.started",
            {"case": name, "run_id": run.name},
            "orchestrator",
        )
        print(encode({"started": name}), flush=True)
        started = time.monotonic()
        try:
            result = fn()
            row = {
                "case": name,
                "completed": True,
                "seconds": time.monotonic() - started,
                "result": result,
            }
        except Exception as exc:
            row = {
                "case": name,
                "completed": False,
                "seconds": time.monotonic() - started,
                "error": type(exc).__name__ + ": " + str(exc),
            }
            report["failures"].append(row["error"])
        report["cases"].append(row)
        gw.store.event("cognition_acceptance.finished", row, "orchestrator")
        print(encode(row), flush=True)
        return row.get("result")

    try:

        def ready():
            try:
                state = api("/api/status?after=2147483647")
                if state.get("failure"):
                    raise RuntimeError(state["failure"])
                return state if state.get("ready") else None
            except OSError:
                return None

        wait_for(ready, 90, "Native gateway qualification")
        gw = holder["gateway"]
        e = gw.engine

        def isolation():
            try:
                with urlopen(
                    Request(
                        BASE + "/v1/chat/completions",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=3,
                ):
                    raise AssertionError("Foreign mutation accepted")
            except HTTPError as exc:
                assert exc.code == 503
                return {"foreign_POST_status": exc.code}

        record("mutation_isolation", isolation)
        if resume:
            previous = json.loads(
                (run / "before_restart.json").read_text(encoding="utf-8")
            )

            def cold():
                actual = fingerprint(e)
                assert (
                    actual == previous["fingerprint"]
                ), "Cold SNN/RNG/replay/controller mismatch"
                identifier = previous["pinned_checkpoint"]["id"]
                assert (
                    sha(gw.store.snapshot(identifier))
                    == previous["pinned_checkpoint"]["sha256"]
                )
                return {
                    "neural_rng_replay_review_exact": True,
                    "updates": e.model.updates,
                    "budget_used": e.state["cognition"]["replay_used"],
                }

            record("cold_restart_exact", cold)
            record("paused_learning_still_answers_and_wakes", lambda: ask("Hello"))
            assert e.model.updates == previous["fingerprint"]["updates"]
        else:
            record("real_streamed_answer", lambda: ask("Hello"))

            def verifier():
                goal = api(
                    "/api/goal",
                    {
                        "question": "12+3",
                        "reason": "engineering closed arithmetic verifier",
                    },
                )
                wait_for(
                    lambda: e.state["goals"][goal["id"]]["status"]
                    == "resolved_verified",
                    35,
                    "Automatic local verifier",
                )
                assert e.model.updates >= 1 and e.model.hidden_updates >= 1
                return {
                    "goal": e.state["goals"][goal["id"]],
                    "updates": e.model.updates,
                    "replay_items": len(e.model.replay),
                }

            record("idle_goal_verification", verifier)

            def sleep():
                before = gw.monitor.sample()
                api("/api/cognition", {"intent": "sleep"})
                wait_for(
                    lambda: e.state["cognition"]["replay_used"] == 2
                    and not e.cognition.resident(),
                    20,
                    "Actual unload + two bounded replay updates",
                )
                wait_for(
                    lambda: not e.cognition.work_active.is_set(), 5, "Replay completion"
                )
                time.sleep(3)
                assert e.model.updates == 3 and e.model.hidden_updates == 3
                assert e.model.replay_seen == 1 and len(e.model.replay) == 1
                return {
                    "before_resources": before,
                    "after_resources": gw.monitor.sample(),
                    "updates": e.model.updates,
                    "hidden_updates": e.model.hidden_updates,
                    "budget_used": e.state["cognition"]["replay_used"],
                    "no_self_growth": True,
                    "vlm_resident": e.cognition.resident(),
                }

            record("sleep_unload_and_bounded_real_replay", sleep)
            record("cold_model_wake_answer", lambda: ask("Hello"))

            def preempt():
                api("/api/cognition", {"intent": "awake"})
                source = ask("Briefly explain what an ALIF neuron does.")
                updates = e.model.updates
                wait_for(
                    lambda: e._reviewing_answer, 40, "Actual background review began"
                )
                pending = e._reviewing_answer
                answer = ask("Hello")
                wait_for(
                    lambda: e._reviewing_answer is None, 5, "Background review yielded"
                )
                events = gw.store.events(0)
                interrupted = any(
                    v["kind"] == "idle_review_preempted"
                    and v["payload"].get("answer_id") == pending
                    for v in events
                )
                assert e.model.updates == updates
                return {
                    "reviewed_actual_answer": source,
                    "foreground_answer": answer,
                    "interrupted": interrupted,
                    "review_still_pending": pending in e.state["review"]["pending"],
                    "warm_ttft_target_under_1s": answer["ttft_seconds"] is not None
                    and answer["ttft_seconds"] < 1,
                    "no_updates_during_chat": e.model.updates == updates,
                }

            record("real_review_preemption", preempt)
        api("/api/learning", {"enabled": False})
        api("/api/cognition", {"intent": "sleep"})
        wait_for(
            lambda: not e.cognition.resident() and not e.cognition.work_active.is_set(),
            10,
            "Final paused rest",
        )
        pin = api("/api/save", {})["id"]
        payload = gw.store.snapshot(pin)
        api("/api/rollback", {"id": pin})
        with e.lock:
            assert e.model.dumps(e.state) == payload, "Rollback full state mismatch"
        report["pinned_checkpoint"] = {"id": pin, "sha256": sha(payload)}
        report["fingerprint"] = fingerprint(e)
        report["full_rollback_exact"] = True
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        if "gateway" in holder:
            gw = holder["gateway"]
            report["resources"] = gw.monitor.sample()
            report["native_gate"] = gw.store.get("runtime", "native_gate")
            # Only whitelisted engineering kinds; no private source edits or owners' chats.
            events = [
                v
                for v in gw.store.events(0)
                if v["kind"].startswith(
                    (
                        "cognitive_",
                        "sleep_replay_",
                        "vlm_hibernated",
                        "vlm_resumed",
                        "idle_review_",
                        "learning_",
                        "neural_",
                        "cognition_acceptance.",
                    )
                )
            ]
            (
                run / ("events_resume.json" if resume else "events_first.json")
            ).write_text(encode(events), encoding="utf-8")
            try:
                api("/api/shutdown", {})
            except Exception as exc:
                report["failures"].append("shutdown: " + str(exc))
            thread.join(30)
            report["gateway_stopped"] = not thread.is_alive()
        report["finished"] = time.time()
        dest.write_text(encode(report), encoding="utf-8")
        print(
            encode({"receipt": str(dest), "failures": report["failures"]}), flush=True
        )
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        sys.exit(execute(args.run.resolve(), args.resume))
    else:
        parser.error("--prepare or --run required")
