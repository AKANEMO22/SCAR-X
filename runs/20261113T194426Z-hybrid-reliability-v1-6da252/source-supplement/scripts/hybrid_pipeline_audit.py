"""Actual isolated gateway/HTTP/media/persistence audit; no model-output mocks."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from urllib.request import Request, urlopen

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from scarx.hybrid import gateway as serving
from scarx.hybrid.store import encode
from scarx.hybrid.learner import LocalALIF
import hybrid_service
from hybrid_context_benchmark import consume

BASE = "http://127.0.0.1:8775"
AUDIT_TOKEN = secrets.token_urlsafe(32)


def request_is_owned(header, expected):
    return isinstance(header, str) and secrets.compare_digest(header, expected)


def isolated_handler(handler, token):
    class Isolated(handler):
        def do_POST(self):
            if not request_is_owned(self.headers.get("X-SCARX-Audit"), token):
                self.send(
                    {
                        "error": {
                            "message": "Isolated pipeline audit is running; normal chat resumes automatically."
                        }
                    },
                    503,
                )
                return
            return super().do_POST()

    return Isolated


def sha(data):
    return hashlib.sha256(data).hexdigest()


def api(path, body=None, stream=False):
    headers = {"X-SCARX-Backend": "hermes", "X-SCARX-Audit": AUDIT_TOKEN}
    if body is not None:
        token = api("/api/status?after=2147483647")["token"]
        headers |= {"X-SCARX-Token": token, "Content-Type": "application/json"}
    start = time.monotonic()
    with urlopen(
        Request(
            BASE + path,
            data=None if body is None else encode(body).encode(),
            headers=headers,
        ),
        timeout=120,
    ) as response:
        if not stream:
            return json.load(response)
        header = time.monotonic() - start
        output = consume(response)
        output["ttft_seconds"] = (
            None
            if output["first_after_headers_seconds"] is None
            else header + output["first_after_headers_seconds"]
        )
        output["total_seconds"] = time.monotonic() - start
        return output


def ask(prompt, stream=None, normal=True, streaming=False):
    msgs = (
        [
            {
                "role": "system",
                "content": "Answer the question directly in English. Follow supplied evidence; return only the answer requested.",
            }
        ]
        if normal
        else []
    )
    msgs.append({"role": "user", "content": prompt})
    return api(
        "/v1/chat/completions",
        {
            "model": "scarx-hybrid",
            "messages": msgs,
            "scarx_stream_id": stream,
            "scarx_session_id": "engineering-pipeline-audit",
            "max_tokens": 64,
            "stream": streaming,
        },
        stream=streaming,
    )


def text(result):
    return (
        result.get("answer", "")
        if "answer" in result
        else result["choices"][0]["message"].get("content", "")
    )


def word_error(hypothesis, reference):
    def words(s):
        return re.findall(r"\w+", unicodedata.normalize("NFC", s).casefold())

    hyp, ref = words(hypothesis), words(reference)
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        row = [i]
        for j, b in enumerate(hyp, 1):
            row.append(min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = row
    return {
        "word_edits": prev[-1],
        "reference_words": len(ref),
        "wer": prev[-1] / max(1, len(ref)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.prepare:
        name = (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + "-hybrid-pipeline-audit-v1-"
            + uuid.uuid4().hex[:6]
        )
        run = ROOT / "runs" / name
        run.mkdir()
        data = ROOT / "local" / name
        data.mkdir()
        source = ROOT / "local/hybrid-v1"
        shutil.copytree(source / "runtime", data / "runtime", copy_function=os.link)
        (data / "models/native").mkdir(parents=True)
        for f in (source / "models/native").iterdir():
            if f.is_file():
                os.link(f, data / "models/native" / f.name)
        for f in ("native-integrity.json", "original-reference.json"):
            shutil.copyfile(source / f, data / f)
        fixture = json.loads(
            (source / "samples/engineering-manifest.json").read_text(encoding="utf-8")
        )
        manifest = {
            "run_id": name,
            "data_dir": str(data),
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "script_sha256": sha(Path(__file__).read_bytes()),
            "protocol_sha256": sha(
                (ROOT / "experiments/hybrid-pipeline-audit-v1.md").read_bytes()
            ),
            "fixture": fixture,
            "prepared": time.time(),
            "request_isolation": "scarx-pipeline-audit-v2: per-process nonce gates ALL mutation routes; no foreign chat/presence/upload",
            "model_integrity": json.loads(
                (source / "native-integrity.json").read_text(encoding="utf-8")
            ),
        }
        (run / "manifest.json").write_text(encode(manifest), encoding="utf-8")
        (run / "source").mkdir()
        shutil.copyfile(__file__, run / "source/hybrid_pipeline_audit.py")
        shutil.copyfile(
            ROOT / "experiments/hybrid-pipeline-audit-v1.md", run / "source/protocol.md"
        )
        shutil.copyfile(
            ROOT / "experiments/hybrid-pipeline-audit-harness-v2.md",
            run / "source/harness-v2.md",
        )
        print(encode({"prepared": name, "run": str(run)}))
        return
    if args.run is None:
        p.error("--prepare or --run required")
    run = args.run.resolve()
    if ROOT / "runs" not in run.parents:
        raise ValueError("Run outside project")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if sha(Path(__file__).read_bytes()) != manifest["script_sha256"]:
        raise ValueError("Frozen runner changed")
    data = Path(manifest["data_dir"])
    dest = run / ("after_restart.json" if args.resume else "before_restart.json")
    if dest.exists():
        raise ValueError("Refuse overwrite")
    if hybrid_service.status():
        raise RuntimeError("Stop the owned normal gateway before isolated test")
    holder = {}
    original = serving.Gateway
    original_server = serving.ThreadingHTTPServer

    def server_factory(address, handler):
        return original_server(address, isolated_handler(handler, AUDIT_TOKEN))

    serving.ThreadingHTTPServer = server_factory

    class ObservedGateway(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            holder["gateway"] = self

    # Constructor observation only; all serving/inference/learning code is real.
    serving.Gateway = ObservedGateway
    thread = threading.Thread(target=serving.serve, args=(data, 8775), daemon=True)
    thread.start()
    report = {
        "run_id": run.name,
        "phase": "after_restart" if args.resume else "before_restart",
        "started": time.time(),
        "cases": [],
        "failures": [],
        "scope": "Engineering fixtures, NOT general quality/lifelong PASS",
    }
    fixture_process = None

    def event(kind, payload):
        holder["gateway"].store.event("pipeline_audit." + kind, payload, "orchestrator")
        print(encode({"kind": kind, **payload}), flush=True)

    def case(name, fn):
        event("stage_started", {"case": name})
        started = time.monotonic()
        try:
            result = fn()
            row = {
                "case": name,
                "result": result,
                "seconds": time.monotonic() - started,
                "completed": True,
            }
        except Exception as exc:
            row = {
                "case": name,
                "completed": False,
                "error": type(exc).__name__ + ": " + str(exc),
                "seconds": time.monotonic() - started,
            }
            report["failures"].append(row["error"])
        report["cases"].append(row)
        event("stage_finished", row)
        return row.get("result")

    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                s = api("/api/status?after=2147483647")
                if s.get("failure"):
                    raise RuntimeError(s["failure"])
                if s.get("ready"):
                    break
            except OSError:
                pass
            time.sleep(0.2)
        else:
            raise TimeoutError("Isolated native readiness timeout")
        gw = holder["gateway"]
        gw.store.put(
            "runtime",
            "status",
            {
                "phase": "Pipeline audit — isolated data",
                "note": "Real local inference, no production knowledge or training. Automatic trace.",
            },
        )
        api("/api/learning", {"enabled": False})
        if args.resume:
            before = json.loads(
                (run / "before_restart.json").read_text(encoding="utf-8")
            )
            saved = before["checkpoint"]
            payload = gw.store.snapshot(saved["id"])
            report["cold_persistence"] = {
                "payload_hash_matches": sha(payload) == saved["payload_sha256"],
                "evidence_count": gw.store.db.execute(
                    "select count(*) from evidence"
                ).fetchone()[0],
                "expected_min_evidence_count": saved["evidence_count"],
                "synapse_updates": gw.engine.model.updates,
            }
            with gw.engine.lock:
                report["cold_persistence"]["active_state_exact"] = (
                    sha(gw.engine.model.dumps(gw.engine.state))
                    == before["final_active_checkpoint"]["payload_sha256"]
                )
            sid = before["video_stream"]
            case(
                "video_after_cold_restart",
                lambda: {
                    "response": ask(
                        "At the start of the video, what color is the circle on the right? Reply with one color word.",
                        sid,
                    ),
                    "expected": "blue",
                },
            )
            return
        case(
            "streamed_chat",
            lambda: {
                "response": ask(
                    "What is 17+25? Reply with the number only.", streaming=True
                ),
                "expected": "42",
            },
        )

        def tools():
            tool = {
                "type": "function",
                "function": {
                    "name": "lookup_number",
                    "description": "Look up the fixture value for an exact key; do not guess.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            }
            msgs = [
                {
                    "role": "system",
                    "content": "Use the provided function. Return its numeric result without explanation.",
                },
                {"role": "user", "content": "Use lookup_number for key ALPHA."},
            ]
            response = api(
                "/v1/chat/completions",
                {"messages": msgs, "tools": [tool], "max_tokens": 96},
            )
            msg = response["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            result = {
                "first_response": response,
                "expected_function": "lookup_number",
                "expected_arguments": {"key": "ALPHA"},
            }
            if not calls:
                return result
            call = calls[0]
            if call["function"]["name"] != "lookup_number" or json.loads(
                call["function"]["arguments"]
            ) != {"key": "ALPHA"}:
                return result
            msgs += [msg, {"role": "tool", "tool_call_id": call["id"], "content": "17"}]
            result["second_response"] = api(
                "/v1/chat/completions",
                {"messages": msgs, "tools": [tool], "max_tokens": 64},
            )
            result["expected_answer"] = "17"
            return result

        case("function_round_trip", tools)

        def image():
            source = ROOT / "local/hybrid-v1/samples/scene-first.png"
            msg = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color is the circle? Reply with one color word.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(source.read_bytes()).decode()
                            },
                        },
                    ],
                }
            ]
            return {
                "response": api(
                    "/v1/chat/completions", {"messages": msg, "max_tokens": 64}
                ),
                "expected": "blue",
                "image_sha256": sha(source.read_bytes()),
            }

        case("inline_vision", image)

        def video():
            st = api("/api/observe", {"path": manifest["fixture"]["video"]})
            report["video_stream"] = st["id"]
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                row = gw.store.get("stream", st["id"])
                if row["status"] in ("complete", "failed", "stopped"):
                    return row
                time.sleep(0.5)
            return row | {"audit_timeout": True}

        case("uploaded_video", video)
        sid = report.get("video_stream")
        if sid:
            for name, prompt, expected, normal in (
                (
                    "video_start_agent",
                    "At the start of the video, what color is the circle on the right? Reply with one color word.",
                    "blue",
                    True,
                ),
                (
                    "video_end_agent",
                    "At the end of the video, what color is the circle on the right? Reply with one color word.",
                    "yellow",
                    True,
                ),
                (
                    "video_start_debug",
                    "At the start of the video, what color is the circle on the right? Reply with one color word.",
                    "blue",
                    False,
                ),
            ):
                case(
                    name,
                    lambda p=prompt, e=expected, n=normal: {
                        "response": ask(p, sid, normal=n),
                        "expected": e,
                    },
                )
            audio = gw.store.timeline(sid, limit=12, modality="audio")
            transcript = " ".join(e["text"] for e in audio)
            report["ASR"] = {
                "transcript": transcript,
                "reference": manifest["fixture"]["independent_transcript"],
                **word_error(transcript, manifest["fixture"]["independent_transcript"]),
            }

        def screen():
            nonlocal fixture_process
            fixture_process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/hybrid_capture_fixture.py"),
                    "--seconds",
                    "50",
                ],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                source = next(
                    (
                        s
                        for s in api("/api/screen/sources")["sources"]
                        if s["title"]
                        == "SCAR-X engineering capture fixture — synthetic shapes"
                    ),
                    None,
                )
                if source:
                    break
                time.sleep(0.2)
            if not source:
                raise RuntimeError("Own fixture window missing")
            st = api(
                "/api/screen/start",
                {
                    "hwnd": source["id"],
                    "system_audio": False,
                    "session_id": "engineering-pipeline-audit",
                },
            )
            try:
                deadline = time.monotonic() + 35
                while time.monotonic() < deadline:
                    row = gw.store.get("stream", st["id"])
                    if row["status"] == "failed" or row["vision_frontier"] >= 1:
                        break
                    time.sleep(0.5)
                return {
                    "stream": row,
                    "microphone_used": False,
                    "system_audio_requested": False,
                }
            finally:
                api("/api/screen/stop", {"id": st["id"]})
                if fixture_process.poll() is None:
                    fixture_process.terminate()
                    fixture_process.wait(timeout=5)

        case("screen_capture_own_fixture", screen)
        saved = api("/api/save", {})["id"]
        payload = gw.store.snapshot(saved)
        count = gw.store.db.execute("select count(*) from evidence").fetchone()[0]
        report["checkpoint"] = {
            "id": saved,
            "payload_sha256": sha(payload),
            "evidence_count": count,
        }

        def rollback():
            ask("Engineering audit checkpoint marker. Reply OK.")
            old_events = gw.store.db.execute("select max(id) from events").fetchone()[0]
            api("/api/rollback", {"id": saved})
            with gw.engine.lock:
                exact = sha(gw.engine.model.dumps(gw.engine.state)) == sha(payload)
            return {
                "full_snapshot_exact": exact,
                "evidence_retained": gw.store.db.execute(
                    "select count(*) from evidence"
                ).fetchone()[0]
                >= count,
                "events_retained": gw.store.db.execute(
                    "select max(id) from events"
                ).fetchone()[0]
                > old_events,
            }

        case("rollback", rollback)

        def idle():
            initial = gw.engine.state["review"]["completed"]
            api("/api/learning", {"enabled": True})
            deadline = time.monotonic() + 75
            while time.monotonic() < deadline:
                if gw.engine.state["review"]["completed"] > initial:
                    break
                time.sleep(0.5)
            return {
                "new_review_count": gw.engine.state["review"]["completed"] - initial,
                "review": gw.engine.state["review"],
                "synapse_updates": gw.engine.model.updates,
                "hidden_updates": gw.engine.model.hidden_updates,
                "verified_replay_items": len(gw.engine.model.replay),
            }

        case("automatic_idle_review", idle)
        api("/api/learning", {"enabled": False})
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        if fixture_process and fixture_process.poll() is None:
            fixture_process.terminate()
            fixture_process.wait(timeout=5)
        if "gateway" in holder:
            gw = holder["gateway"]
            report["resources"] = gw.monitor.sample()
            active = gw.store.get("runtime", "active_checkpoint")
            if active:
                report["final_active_checkpoint"] = {
                    "id": active["id"],
                    "payload_sha256": sha(gw.store.snapshot(active["id"])),
                }
            # Full audit journal stays in the isolated SQLite database, never
            # copied into a distributable run receipt by default.
            report["events_in_private_sqlite"] = True
            report["native_gate"] = gw.store.get("runtime", "native_gate")
            try:
                api("/api/shutdown", {})
            except Exception as exc:
                report["failures"].append("shutdown: " + str(exc))
            thread.join(timeout=30)
            report["gateway_stopped"] = not thread.is_alive()
        report["finished"] = time.time()
        dest.write_text(encode(report), encoding="utf-8")
        print(
            encode(
                {
                    "receipt": str(dest),
                    "failures": report["failures"],
                    "cases": len(report["cases"]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
