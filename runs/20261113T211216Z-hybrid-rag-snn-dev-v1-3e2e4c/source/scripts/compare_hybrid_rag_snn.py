"""Matched isolated SNN-vs-RAG DEV, real frozen VLM and immutable raw outputs."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace
from urllib.request import Request, urlopen

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import psutil
from scarx.evaluation.hybrid_rag_snn_dev import (
    dataset,
    SEEDS,
    FAMILIES,
    pair_features,
    select_bm25,
    select_rule,
    select_neural,
    Episodic,
    exact_source,
    forgetting,
    summary_metric,
)
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.config import HybridConfig
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.resources import ResourceMonitor
from scarx.hybrid.store import Store, encode

ARMS = (
    "bm25",
    "source_rule",
    "episodic",
    "frozen",
    "readout_only",
    "hidden",
    "hidden_reset",
)
PROTOCOL = ROOT / "experiments/hybrid-rag-snn-comparison-dev-v1.md"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def get(path):
    with urlopen(
        Request("http://127.0.0.1:8775" + path, headers={"X-SCARX-Backend": "hermes"}),
        timeout=10,
    ) as r:
        return json.load(r)


def control(path, body):
    token = get("/api/status?after=2147483647")["token"]
    with urlopen(
        Request(
            "http://127.0.0.1:8775" + path,
            data=encode(body).encode(),
            headers={"X-SCARX-Token": token, "Content-Type": "application/json"},
        ),
        timeout=10,
    ) as r:
        return json.load(r)


def prepare():
    name = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-hybrid-rag-snn-dev-v1-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / name
    run.mkdir()
    (ROOT / "local" / name).mkdir()
    raw = encode(dataset()).encode()
    (run / "data.json").write_bytes(raw)
    paths = [
        PROTOCOL,
        Path(__file__),
        ROOT / "src/scarx/evaluation/hybrid_rag_snn_dev.py",
    ]
    paths += list((ROOT / "src/scarx/hybrid").glob("*.py")) + [
        ROOT / "src/scarx/neural/core.py",
        ROOT / "src/scarx/evaluation/readout_eligibility.py",
    ]
    hashes = {}
    for p in paths:
        rel = p.relative_to(ROOT)
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        hashes[rel.as_posix()] = sha(p.read_bytes())
    config = {
        "seeds": SEEDS,
        "arms": ARMS,
        "stages": 20,
        "updates_per_stage": 24,
        "max_native_calls": 1000,
        "max_seconds": 900,
        "same_candidate_pool": True,
        "learning_rate_search": False,
        "train_feedback": "owned closed event contract; label authority is external to learner",
        "hidden_rows": 128,
        "device": "gpu",
        "dtype": "float32",
        "test_response_cache": "shared identical native inputs; not independent inference replicates",
        "threshold": 0.5,
    }
    manifest = {
        "prepared": time.time(),
        "run_id": name,
        "parent_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "data_sha256": sha(raw),
        "source_sha256": hashes,
        "config": config,
        "data_source": "procedural owned synthetic event records; no private data",
        "ponytail": "not available in enabled tools",
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ROOT / "local/current-rag-snn-run.txt").write_text(str(run))
    print(encode({"prepared": str(run)}), flush=True)


def predict(arm, case, models, episodic):
    if arm == "bm25":
        return select_bm25(case)
    if arm == "source_rule":
        return select_rule(case)
    if arm == "episodic":
        return episodic.select(case)
    model = models[arm].model if hasattr(models[arm], "model") else models[arm]
    return select_neural(model, case)


def probe(arm, cases, models, episodic):
    rows = [
        {
            "family": c["family"],
            "source_correct": exact_source(predict(arm, c, models, episodic), c),
        }
        for c in cases
    ]
    return summary_metric(rows)


def select_native(status):
    owner = psutil.Process(status["pid"])
    workers = [
        p for p in owner.children(recursive=True) if p.name() == "llama-server.exe"
    ]
    if len(workers) != 1:
        raise RuntimeError("Expected exactly one owned VLM")
    c = workers[0].cmdline()
    if c[c.index("-ngl") + 1] != "99" or c[c.index("--device") + 1] != "CUDA0":
        raise RuntimeError("Native VLM is not CUDA0 all-layer configuration")
    return workers[0], c[c.index("--api-key") + 1]


def actual_answer(case, ident, key):
    doc = next((d for d in case["docs"] if d["id"] == ident), None)
    evidence = (
        None
        if doc is None
        else {k: doc[k] for k in ["id", "text", "entity", "time", "revision", "status"]}
    )
    messages = [
        {
            "role": "system",
            "content": 'Answer using only the supplied project record. Unverified/withdrawn records do not establish the requested approved value. Return JSON {"value":integer} or {"value":null} if the evidence cannot establish the requested value. Never guess.',
        },
        {
            "role": "user",
            "content": encode({"question": case["query"], "record": evidence}),
        },
    ]
    body = {
        "model": "scarx-hybrid",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 96,
        "repeat_penalty": 1.05,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_value",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "value": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        },
    }
    return body


def run(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    data = json.loads((run / "data.json").read_text(encoding="utf-8"))
    if (run / "started.json").exists():
        raise ValueError("Refuse rerun or final-test reuse")
    if sha((run / "data.json").read_bytes()) != manifest["data_sha256"]:
        raise ValueError("Dataset changed")
    for rel, h in manifest["source_sha256"].items():
        if sha((ROOT / rel).read_bytes()) != h:
            raise ValueError("Source changed: " + rel)
    live_status = get("/api/status?after=2147483647")
    if not live_status["ready"] or live_status["config"]["execution_mode"] != "gpu":
        raise RuntimeError("Owned GPU service required")
    if live_status["cognition"]["foreground_requests"]:
        raise RuntimeError("Foreground request in flight; retry after it finishes")
    worker, native_key = select_native(live_status)
    (run / "started.json").write_text(
        encode(
            {
                "started": time.time(),
                "gateway_pid": live_status["pid"],
                "native_pid": worker.pid,
            }
        ),
        encoding="utf-8",
    )
    private = ROOT / "local" / run.name
    trace = Store(ROOT / "local/hybrid-v1")
    monitor = ResourceMonitor(
        None, extra_pids=[live_status["pid"]], include_hermes=True
    ).start()
    initially_learning = live_status["learning_enabled"]
    control("/api/learning", {"enabled": False})
    started = time.monotonic()
    report = {
        "run_id": run.name,
        "seeds": {},
        "failures": [],
        "native_calls": 0,
        "budget_seconds": 900,
        "research_checkpoints_promoted": False,
    }
    native_cache = {}
    selection = []
    all_engines = []

    def emit(kind, payload):
        trace.event(
            "rag_snn_dev." + kind, dict(run_id=run.name, **payload), "orchestrator"
        )
        print(
            encode(
                {
                    "stage": kind,
                    **{
                        k: v
                        for k, v in payload.items()
                        if k in ["seed", "stage", "method", "native_calls"]
                    },
                }
            ),
            flush=True,
        )

    def guard():
        if time.monotonic() - started > 900:
            raise TimeoutError("Registered15minute DEV budget exhausted")
        monitor.check()
        if not worker.is_running():
            raise RuntimeError("Owned native worker stopped")

    try:
        emit(
            "started",
            {
                "scope": "isolated models, live VLM frozen; not live neural counters",
                "config": manifest["config"],
            },
        )
        for seed in SEEDS:
            models = {}
            initial = LocalALIF(seed).dumps()
            episodic = Episodic()
            curves = {a: [] for a in ARMS if a != "hidden_reset"}
            for arm, mode in [
                ("frozen", "frozen"),
                ("readout_only", "readout_only"),
                ("hidden", "learned"),
            ]:
                store = Store(private / (str(seed) + "-" + arm))
                e = HybridEngine(
                    store,
                    SimpleNamespace(),
                    monitor,
                    HybridConfig(execution_mode="gpu", seed=seed),
                )
                e.model = LocalALIF(seed, mode).to_device("gpu")
                models[arm] = e
                all_engines.append(e)
            initial_states = {a: models[a].model.dumps() for a in models}
            update_logs = {a: [] for a in ["readout_only", "hidden"]}
            cold_exact = {}
            for arm in curves:
                curves[arm].append(
                    {"stage": 0, **probe(arm, data["probes"], models, episodic)}
                )
            for stage, items in enumerate(data["train_stages"], 1):
                guard()
                order = list(range(len(items)))
                random.Random(seed + stage).shuffle(order)
                for ix in order:
                    item = items[ix]
                    case = item["case"]
                    doc = next(d for d in case["docs"] if d["id"] == item["doc_id"])
                    target = item["target"]
                    seq = pair_features(case, doc)
                    evidence = {
                        "kind": "machine_verifier",
                        "id": "event-source-contract:"
                        + case["id"]
                        + ":"
                        + item["doc_id"],
                    }
                    for arm in ["readout_only", "hidden"]:
                        e = models[arm]
                        before_p = float(e.model.forward(seq, "relevance")[0][1])
                        t = time.monotonic()
                        with e.lock:
                            result = e._guarded_learn(
                                seq, target, "relevance", 1.0, evidence
                            )
                        update_logs[arm].append(
                            {
                                "stage": stage,
                                "case": case["id"],
                                "doc": doc["id"],
                                "target": target,
                                "prior_wrong": int(before_p > 0.5) != target,
                                "seconds": time.monotonic() - t,
                                **result,
                            }
                        )
                    episodic.observe(case, doc, target)
                for arm in curves:
                    curves[arm].append(
                        {"stage": stage, **probe(arm, data["probes"], models, episodic)}
                    )
                emit(
                    "learning_stage",
                    {
                        "seed": seed,
                        "stage": stage,
                        "arms": {
                            a: {
                                "updates": models[a].model.updates,
                                "hidden_updates": models[a].model.hidden_updates,
                                "guard_rejections": sum(
                                    not x.get("accepted", False) for x in update_logs[a]
                                ),
                                "probe_source_accuracy": curves[a][-1]["accuracy"],
                                "last_learning": update_logs[a][-1],
                            }
                            for a in update_logs
                        },
                    },
                )
            for arm in models:
                payload = models[arm].model.dumps(models[arm].state)
                (run / f"{seed}-{arm}.npz").write_bytes(payload)
                a = models[arm].model
                b, _ = LocalALIF.loads(
                    (run / f"{seed}-{arm}.npz").read_bytes(), device="gpu"
                )
                case = data["train_stages"][0][0]["case"]
                doc = case["docs"][0]
                seq = pair_features(case, doc)
                ev = {"kind": "machine_verifier", "id": "persistence-fixture-next-step"}
                a.learn(seq, 0, head="relevance", evidence=ev)
                b.learn(seq, 0, head="relevance", evidence=ev)
                cold_exact[arm] = a.dumps() == b.dumps()
                models[arm].model, _ = LocalALIF.loads(payload, device="gpu")
            reset, _ = LocalALIF.loads(models["hidden"].model.dumps(), device="gpu")
            base, _ = LocalALIF.loads(initial, device="gpu")
            for name in ["w_in", "w_rec"]:
                getattr(reset.core, name)[:] = getattr(base.core, name)
            models["hidden_reset"] = reset
            for arm in ARMS:
                for case in data["test"]:
                    ident = predict(arm, case, models, episodic)
                    selection.append(
                        {
                            "seed": seed,
                            "method": arm,
                            "id": case["id"],
                            "family": case["family"],
                            "selected": ident,
                            "gold_id": case["gold_id"],
                            "gold": case["gold"],
                            "source_correct": exact_source(ident, case),
                        }
                    )
            report["seeds"][str(seed)] = {
                "curves": curves,
                "forgetting": {a: forgetting(c) for a, c in curves.items()},
                "cold_same_device_next_update_exact": cold_exact,
                "source_selection": {
                    a: summary_metric(
                        [r for r in selection if r["seed"] == seed and r["method"] == a]
                    )
                    for a in ARMS
                },
                "learning": {
                    a: {
                        "attempts": len(log),
                        "accepted": sum(x.get("accepted", False) for x in log),
                        "rejected": sum(not x.get("accepted", False) for x in log),
                        "prior_wrong": sum(x["prior_wrong"] for x in log),
                        "seconds": sum(x["seconds"] for x in log),
                    }
                    for a, log in update_logs.items()
                },
            }
            (run / f"{seed}-learning.json").write_text(
                encode(update_logs), encoding="utf-8"
            )
            (run / "selection.json").write_text(encode(selection), encoding="utf-8")
            (run / "partial.json").write_text(encode(report), encoding="utf-8")
            emit(
                "seed_complete",
                {
                    "seed": seed,
                    "source_selection": report["seeds"][str(seed)]["source_selection"],
                    "forgetting": report["seeds"][str(seed)]["forgetting"],
                },
            )
            for e in models.values():
                if hasattr(e, "store"):
                    e.store.close()
            del models, reset, base, a, b
        # One frozen VLM call per unique input, shared across all arms; no cached replicate inflation.
        cases = {c["id"]: c for c in data["test"]}
        native_file = run / "native.jsonl"
        for row in selection:
            guard()
            case = cases[row["id"]]
            body = actual_answer(case, row["selected"], native_key)
            key = sha(encode(body).encode())
            row["native_key"] = key
            if key not in native_cache:
                if len(native_cache) >= 1000:
                    raise RuntimeError("Unique VLM request budget exceeded")
                t = time.monotonic()
                with urlopen(
                    Request(
                        "http://127.0.0.1:18776/v1/chat/completions",
                        data=encode(body).encode(),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": "Bearer " + native_key,
                        },
                    ),
                    timeout=60,
                ) as response:
                    raw = json.load(response)
                value = raw["choices"][0]["message"].get("content", "")
                record = {
                    "key": key,
                    "query": case["query"],
                    "selected": row["selected"],
                    "request": body,
                    "response": raw,
                    "seconds": time.monotonic() - t,
                }
                native_cache[key] = record
                with native_file.open("a", encoding="utf-8") as f:
                    f.write(encode(record) + "\n")
                emit(
                    "actual_answer",
                    {
                        "method": row["method"],
                        "seed": row["seed"],
                        "query": case["query"],
                        "selected": row["selected"],
                        "model_answer": value,
                        "expected": case["gold"],
                        "source_correct": row["source_correct"],
                        "native_calls": len(native_cache),
                        "seconds": record["seconds"],
                        "cached": False,
                    },
                )
            record = native_cache[key]
            answer = record["response"]["choices"][0]["message"].get("content", "")
            try:
                parsed = json.loads(answer)
                correct = (
                    type(parsed) is dict
                    and set(parsed) == {"value"}
                    and parsed["value"] == case["gold"]
                    and (
                        type(parsed["value"]) is int
                        if case["gold"] is not None
                        else parsed["value"] is None
                    )
                )
            except (ValueError, TypeError):
                correct = False
            row.update(
                answer=answer,
                answer_correct=correct,
                grounded_correct=bool(correct and row["source_correct"]),
                native_seconds=record["seconds"],
            )
        report["native_calls"] = len(native_cache)
        report["predictions"] = selection
        report["methods"] = {
            a: {
                "source": summary_metric([r for r in selection if r["method"] == a]),
                "grounded_QA": summary_metric(
                    [r for r in selection if r["method"] == a], "grounded_correct"
                ),
                "per_seed": {
                    str(seed): summary_metric(
                        [
                            r
                            for r in selection
                            if r["method"] == a and r["seed"] == seed
                        ],
                        "grounded_correct",
                    )
                    for seed in SEEDS
                },
            }
            for a in ARMS
        }
        emit(
            "completed",
            {
                "methods": report["methods"],
                "native_calls": len(native_cache),
                "quality_status": "must audit strongest controls/CI/ablation, no automatic promotion",
            },
        )
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
        emit("failed", {"error": report["failures"][-1]})
    finally:
        report["seconds"] = time.monotonic() - started
        report["resources"] = monitor.sample()
        monitor.close()
        trace.close()
        if initially_learning:
            try:
                control("/api/learning", {"enabled": True})
            except Exception as exc:
                report["failures"].append("restore learning: " + str(exc))
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
    return int(bool(report["failures"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.run:
        target = a.run.resolve()
        if ROOT / "runs" not in target.parents:
            raise ValueError("Run outside workspace")
        sys.exit(run(target))
    else:
        p.error("--prepare or --run required")
