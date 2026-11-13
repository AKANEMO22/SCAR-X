"""Bounded sequential context pilot. Never changes serving code/config/weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from scarx.hybrid.resources import ResourceMonitor
from scarx.hybrid.store import Store, encode
from scarx.hybrid.learner import LocalALIF
import psutil
import hybrid_service

CONTEXTS = (16384, 24576, 32768)
CAP = 4_000_000_000
OUT = 96


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def api(url, body=None, key=None, timeout=20):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    with urlopen(Request(url, data=None if body is None else encode(body).encode(),
                         headers=headers), timeout=timeout) as response:
        return json.load(response)


def score(answer, expected):
    try:
        actual = json.loads(answer.strip())
    except (ValueError, AttributeError):
        return False
    return type(actual) is dict and actual.keys() == expected.keys() and all(
        type(actual[k]) is type(v) and actual[k] == v for k, v in expected.items())


def fits(tokens, context):
    return tokens + OUT < context


def messages(prompt):
    return [{"role": "system", "content": "Answer using only the supplied data. Return exactly the requested JSON object, without explanation or markdown."},
            {"role": "user", "content": prompt}]


def dataset(token_count):
    cases = [
        {"id": "short-vi-revision", "prompt": 'Cấu hình ban đầu: cổng 8123, giữ log 4 ngày. Chỉ đổi cổng thành 9237, sau đó chỉ đổi giữ log thành 6 ngày. Trả JSON {"port":số,"days":số} của cấu hình cuối.', "expected": {"port": 9237, "days": 6}},
        {"id": "short-en-math", "prompt": 'A box contains 17 red and 9 blue tokens. Remove 5 red tokens, then double the remaining blue tokens. Return {"red":integer,"total":integer}.', "expected": {"red": 12, "total": 30}},
        {"id": "short-code", "prompt": 'Execute this Python mentally: a=[3,7,2]; a.append(a.pop(0)+4); answer=sum(a[1:]). Return {"answer":integer}.', "expected": {"answer": 9}},
        {"id": "short-unanswerable", "prompt": 'The supplied record says: Niva has a green bag. It gives no date of birth. Return {"birth_year":null} if the year of birth is not supplied. Do not invent a year.', "expected": {"birth_year": None}},
    ]
    for target, family in ((12000, "relation"), (12000, "temporal"),
                           (22000, "relation"), (30000, "temporal")):
        rng = random.Random(20261113 + target + (family == "temporal"))
        code = "VK" + str(rng.randrange(1000, 9999))
        facts = ([f"TARGET FACT: crate {code} is assigned to team Niva.",
                  "TARGET FACT: team Niva is stationed in room Cedar; its keeper is Ruma.",
                  "TARGET FACT: team Niva moves from room Cedar to room Amber; its keeper does not change."]
                 if family == "relation" else
                 [f"TARGET EVENT 1: Ruma puts token {code} in drawer Cedar.",
                  f"TARGET EVENT 2: Niva moves token {code} from drawer Cedar to drawer Amber.",
                  f"TARGET EVENT 3: The bell rings. After the bell, Ruma moves token {code} from drawer Amber to drawer Birch."])
        query = (f'Where is crate {code} assigned now and who keeps it? Return {{"room":"name","keeper":"name"}}.'
                 if family == "relation" else
                 f'For token {code}, return its drawer immediately before and after the bell: {{"before":"name","after":"name"}}.')
        expected = ({"room": "Amber", "keeper": "Ruma"} if family == "relation"
                    else {"before": "Amber", "after": "Birch"})
        noise = [f"Unrelated ticket {i}: batch {rng.randrange(10000,99999)} checked by staff {rng.choice(('Meko','Savi','Tulo','Bepi'))}; archive shelf {rng.randrange(10,99)}."
                 for i in range(1800)]

        def make(n):
            rows = noise[:n].copy()
            for fraction, fact in reversed(list(zip((.05,.5,.9), facts))):
                rows.insert(int(n * fraction), fact)
            return 'Read this chronological log. Unrelated tickets do not change target facts.\n' + '\n'.join(rows) + '\nQUESTION: ' + query

        low, high = 1, len(noise)
        while low < high:
            middle = (low + high) // 2
            if token_count(messages(make(middle))) < target:
                low = middle + 1
            else:
                high = middle
        prompt = make(low)
        cases.append({"id": f"long-{target}-{family}", "prompt": prompt,
                      "expected": expected, "target_tokens": target})
    for row in cases:
        row["messages"] = messages(row.pop("prompt"))
        row["prompt_tokens"] = token_count(row["messages"])
        row["common"] = row["prompt_tokens"] < 15000
    return cases


def consume(response):
    started = time.monotonic()
    first = None
    chunks, parts, finish, usage, timings = [], [], None, {}, {}
    for line in response:
        if not line.startswith(b"data: "):
            continue
        content = line[6:].strip()
        if content == b"[DONE]":
            break
        item = json.loads(content)
        chunks.append(item)
        usage = item.get("usage") or usage
        timings = item.get("timings") or timings
        for choice in item.get("choices", []):
            text = choice.get("delta", {}).get("content") or ""
            if text and first is None:
                first = time.monotonic() - started
            parts.append(text)
            finish = choice.get("finish_reason") or finish
    return {"answer": "".join(parts), "first_after_headers_seconds": first,
            "stream_read_seconds": time.monotonic() - started, "finish_reason": finish,
            "usage": usage, "timings": timings, "chunks": chunks}


PAGE = """<!doctype html><meta charset=utf-8><title>SCAR-X context benchmark</title>
<style>body{background:#111820;color:#dce6f4;font:15px system-ui;margin:30px;max-width:1200px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-bottom:1px solid #456;padding:10px}a{color:#9cf}</style>
<h1>SCAR-X · Benchmark context thực</h1><p>16K / 24K / 32K · trọng số nguyên vẹn · không huấn luyện. Chat tạm ngừng trong lúc đo.</p>
<div id=s></div><section id=e></section><script>let c=0;async function p(){try{let s=await(await fetch('/api/status?after='+c)).json();document.getElementById('s').textContent=s.phase+' · '+s.note;for(let e of s.events){c=e.id;let a=document.createElement('article');let h=document.createElement('b');h.textContent=e.kind;let v=document.createElement('pre');v.textContent=JSON.stringify(e.payload,null,2);a.append(h,v);document.getElementById('e').prepend(a)}while(document.getElementById('e').children.length>80)document.getElementById('e').lastChild.remove()}catch(e){}setTimeout(p,1500)}p();</script>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    live = hybrid_service.status()
    if not live or not live.get("ready"):
        raise RuntimeError("Owned serving gateway must be ready before benchmark")
    owner = psutil.Process(live["pid"])
    worker = [p for p in owner.children() if p.name() == "llama-server.exe"]
    if len(worker) != 1:
        raise RuntimeError("Expected exactly one owned native worker")
    command = worker[0].cmdline()
    key = command[command.index("--api-key") + 1]
    native = "http://127.0.0.1:18776"

    if args.prepare:
        run = ROOT / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-hybrid-context-v1-" + uuid.uuid4().hex[:6])
        run.mkdir()

        def count(msgs):
            prompt = api(native + "/apply-template", {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}}, key)["prompt"]
            return len(api(native + "/tokenize", {"content": prompt, "parse_special": True}, key)["tokens"])

        rows = dataset(count)
        (run / "cases.json").write_text(encode(rows), encoding="utf-8")
        model_paths = [Path(command[command.index(flag) + 1]) for flag in ("-m", "--mmproj")]
        manifest = {"source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "script_sha256": digest(__file__), "protocol_sha256": digest(ROOT / "experiments/hybrid-context-engineering-v1.md"),
                    "cases_sha256": digest(run / "cases.json"), "contexts": CONTEXTS,
                    "native_binary_sha256": digest(command[0]),
                    "model_sha256": {p.name: digest(p) for p in model_paths},
                    "locked_before_inference": time.time(), "pony_tail": "unavailable"}
        (run / "manifest.json").write_text(encode(manifest), encoding="utf-8")
        print(encode({"run": str(run), "cases": [{k:v for k,v in row.items() if k not in ("messages",)} for row in rows]}))
        return
    if args.run is None:
        parser.error("--prepare or --run required")
    run = args.run.resolve()
    if ROOT / "runs" not in run.parents or (run / "receipt.json").exists() or (run / "started.json").exists():
        raise ValueError("Refusing external path, rerun or overwrite")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for expected, actual in ((manifest["script_sha256"], digest(__file__)),
                             (manifest["protocol_sha256"], digest(ROOT / "experiments/hybrid-context-engineering-v1.md")),
                             (manifest["cases_sha256"], digest(run / "cases.json"))):
        if expected != actual:
            raise ValueError("Frozen source/protocol/data changed")
    cases = json.loads((run / "cases.json").read_text(encoding="utf-8"))
    (run / "started.json").write_text(encode({"started": time.time(), "gateway_pid": owner.pid}), encoding="utf-8")
    old_root_rss = owner.memory_info().rss
    restored = None
    store, server, proc, monitor = None, None, None, None
    report = {"manifest": manifest, "arms": [], "failures": [], "scope": "engineering DEV text-only; not P1, autonomy or continual-learning benchmark"}
    try:
        hybrid_service.stop()
        store = Store(ROOT / "local/hybrid-v1")
        # Load an independent frozen H128 controller footprint; no update/feedback.
        frozen_snn = LocalALIF(260113)
        state = {"phase": "benchmark", "note": "Preparing fixed context arms"}
        monitor = ResourceMonitor(CAP, include_hermes=True)
        resource_lock = threading.RLock()

        def resource():
            with resource_lock:
                raw = monitor.sample()
                reserve = max(0, old_root_rss - psutil.Process().memory_info().rss)
                return raw | {"controller_reserve_bytes": reserve,
                              "accounted_bytes": max(raw["conservative_ram_bytes"], raw["private_commit_bytes"]) + reserve}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/":
                    data, mime = PAGE.encode(), "text/html; charset=utf-8"
                elif self.path.startswith("/api/status"):
                    from urllib.parse import parse_qs, urlparse
                    after = int(parse_qs(urlparse(self.path).query).get("after", ["0"])[0])
                    data, mime = encode(state | {"model": "scarx-hybrid", "ready": False, "pid": os.getpid(), "resources": resource(), "events": store.events(after)}).encode(), "application/json"
                else:
                    self.send_error(503, "Context benchmark is running; chat will resume automatically")
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                self.send_error(503, "Context benchmark is running; chat will resume automatically")

        server = ThreadingHTTPServer(("127.0.0.1", 8775), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def event(kind, payload):
            store.event("context_benchmark." + kind, payload, "orchestrator")
            print(encode({"kind": kind, **payload}), flush=True)

        event("started", {"run": run.name, "contexts": CONTEXTS, "data_sha256": manifest["cases_sha256"]})
        deadline = time.monotonic() + 1200
        for context in CONTEXTS:
            if time.monotonic() > deadline:
                break
            arm = {"context": context, "records": [], "peak_accounted_bytes": 0, "status": "running"}
            report["arms"].append(arm)
            breached = threading.Event()
            monitor.exceeded = False
            monitor.peak_ram = monitor.peak_rss = monitor.peak_commit = 0
            stop_sampling = threading.Event()

            def sampler():
                while not stop_sampling.wait(.05):
                    value = resource()
                    arm["peak_accounted_bytes"] = max(arm["peak_accounted_bytes"], value["accounted_bytes"])
                    if value["accounted_bytes"] > CAP:
                        if not breached.is_set():
                            arm["breach_resources"] = value
                        breached.set()
                        if proc is not None and proc.poll() is None:
                            proc.terminate()

            sample_thread = threading.Thread(target=sampler, daemon=True)
            sample_thread.start()
            try:
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", 18776)) == 0:
                        raise RuntimeError("Native port occupied; no second model allowed")
                cmd = list(command)
                cmd[cmd.index("-c") + 1] = str(context)
                key = secrets.token_urlsafe(32)
                cmd[cmd.index("--api-key") + 1] = key
                cmd.append("--no-context-shift")
                (run / f"{context}.command.json").write_text(encode(["<internal-key>" if i and cmd[i-1] == "--api-key" else s for i,s in enumerate(cmd)]), encoding="utf-8")
                state["note"] = f"Loading context {context}"
                event("arm_started", {"context": context})
                started = time.monotonic()
                with (run / f"{context}.native.log").open("xb") as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                while True:
                    if breached.is_set() or proc.poll() is not None:
                        raise RuntimeError("Worker stopped during startup")
                    if time.monotonic() - started > 120:
                        raise TimeoutError("Worker startup timeout")
                    try:
                        api(native + "/health", timeout=1)
                        break
                    except Exception:
                        time.sleep(.1)
                arm["load_seconds"] = time.monotonic() - started
                props = api(native + "/props", key=key)
                (run / f"{context}.props.json").write_text(encode(props), encoding="utf-8")
                arm["actual_context"] = props["default_generation_settings"]["n_ctx"]
                if arm["actual_context"] != context:
                    raise ValueError("Actual context differs from requested context")
                for case in cases:
                    if breached.is_set() or time.monotonic() > deadline:
                        break
                    row = {"case": case["id"], "common": case["common"], "expected": case["expected"], "prompt_tokens": case["prompt_tokens"], "admitted": fits(case["prompt_tokens"], context)}
                    arm["records"].append(row)
                    if not row["admitted"]:
                        row["status"] = "NOT_ADMITTED"
                        continue
                    state["note"] = f"{context}: {case['id']} ({case['prompt_tokens']} input tokens)"
                    event("case_started", {"context": context, "case": case["id"], "prompt_tokens": case["prompt_tokens"]})
                    started = time.monotonic()
                    try:
                        body = {"model": "scarx-hybrid", "messages": case["messages"], "max_tokens": OUT, "temperature": 0.0, "seed": 20261113, "stream": True, "stream_options": {"include_usage": True}, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
                        req = Request(native + "/v1/chat/completions", data=encode(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
                        # Wall timeout includes prefill; socket timeout alone is not a wall limit.
                        timed_out = threading.Event()
                        def timeout_worker():
                            timed_out.set()
                            if proc is not None and proc.poll() is None:
                                proc.terminate()
                        timer = threading.Timer(min(180, max(1, deadline-time.monotonic())), timeout_worker)
                        timer.start()
                        try:
                            with urlopen(req, timeout=180) as response:
                                header_seconds = time.monotonic() - started
                                output = consume(response)
                        finally:
                            timer.cancel()
                            timer.join()
                        (run / f"{context}-{case['id']}.raw.json").write_text(encode(output), encoding="utf-8")
                        row.update({k:v for k,v in output.items() if k != "chunks"})
                        row["ttft_seconds"] = None if output["first_after_headers_seconds"] is None else header_seconds + output["first_after_headers_seconds"]
                        row["correct"] = output["finish_reason"] == "stop" and score(output["answer"], case["expected"])
                        row["status"] = "COMPLETED"
                    except Exception as exc:
                        row["status"] = "ERROR"
                        row["error"] = type(exc).__name__ + ": " + str(exc)
                        row["correct"] = False
                        if 'timed_out' in locals() and timed_out.is_set():
                            row["status"] = "TIMEOUT"
                    row["elapsed_seconds"] = time.monotonic() - started
                    row["resources"] = resource()
                    event("answer", {"context": context, **row})
                    if proc.poll() is not None:
                        break
                arm["status"] = "RAM_FAIL" if breached.is_set() else ("COMPLETED" if proc.poll() is None else "WORKER_STOPPED")
            except Exception as exc:
                arm["status"] = "RAM_FAIL" if breached.is_set() else "ERROR"
                arm["error"] = type(exc).__name__ + ": " + str(exc)
            finally:
                stop_sampling.set()
                sample_thread.join(timeout=2)
                arm["final_resources"] = resource()
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                proc = None
                (run / f"{context}.result.json").write_text(encode(arm), encoding="utf-8")
                event("arm_finished", {"context": context, "status": arm["status"], "peak_accounted_bytes": arm["peak_accounted_bytes"], "correct": sum(r.get("correct",False) for r in arm["records"]), "completed": sum(r["status"] == "COMPLETED" for r in arm["records"])})
        state["note"] = "Restoring unchanged 16K serving gateway"
        event("finished", {"run": run.name, "arms": [{k:v for k,v in a.items() if k in ('context','status','peak_accounted_bytes')} for a in report["arms"]]})
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
        if server:
            server.shutdown()
            server.server_close()
        if store:
            store.close()
        if monitor:
            monitor.close()
        try:
            restored = hybrid_service.start()
        except Exception as exc:
            report["failures"].append("RESTORE: " + str(exc))
        report["restored"] = restored
        report["finished"] = time.time()
        (run / "receipt.json").write_text(encode(report), encoding="utf-8")
        print(encode({"run": run.name, "restored": restored, "failures": report["failures"]}), flush=True)


if __name__ == "__main__":
    main()
