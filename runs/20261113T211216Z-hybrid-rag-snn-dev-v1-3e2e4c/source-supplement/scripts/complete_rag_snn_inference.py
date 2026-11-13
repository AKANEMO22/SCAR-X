"""No-learning completion of immutable source selections after parent timeout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from compare_hybrid_rag_snn import get, control, select_native, actual_answer
from scarx.hybrid.store import Store, encode


def main(run):
    if (run / "completion.started").exists():
        raise ValueError("Refuse second completion")
    parent = (run / "result.json").read_bytes()
    selection = (run / "selection.json").read_bytes()
    data = json.loads((run / "data.json").read_text(encoding="utf-8"))
    report = json.loads(parent)
    rows = json.loads(selection)
    if len(rows) != 1008 or len(report["seeds"]) != 3:
        raise ValueError("Parent did not finish all fixed selections/seeds")
    status = get("/api/status?after=2147483647")
    worker, key = select_native(status)
    learning = status["learning_enabled"]
    control("/api/learning", {"enabled": False})
    store = Store(ROOT / "local/hybrid-v1")
    cache = {}
    for line in (run / "native.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        cache[d["key"]] = d
    lock = {
        "started": time.time(),
        "parent_sha256": hashlib.sha256(parent).hexdigest(),
        "selection_sha256": hashlib.sha256(selection).hexdigest(),
        "existing_native_inputs": len(cache),
        "budget_seconds": 180,
        "no_training": True,
        "native_pid": worker.pid,
    }
    (run / "completion.started").write_text(encode(lock), encoding="utf-8")
    started = time.monotonic()
    errors = []
    new_count = 0
    cases = {c["id"]: c for c in data["test"]}
    try:
        for row in rows:
            if time.monotonic() - started > 180:
                raise TimeoutError("Completion180s exhausted")
            case = cases[row["id"]]
            body = actual_answer(case, row["selected"], key)
            digest = hashlib.sha256(encode(body).encode()).hexdigest()
            row["native_key"] = digest
            if digest not in cache:
                if len(cache) >= 1000:
                    raise RuntimeError("Total request cap")
                t = time.monotonic()
                with urlopen(
                    Request(
                        "http://127.0.0.1:18776/v1/chat/completions",
                        data=encode(body).encode(),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": "Bearer " + key,
                        },
                    ),
                    timeout=45,
                ) as response:
                    raw = json.load(response)
                record = {
                    "key": digest,
                    "query": case["query"],
                    "selected": row["selected"],
                    "request": body,
                    "response": raw,
                    "seconds": time.monotonic() - t,
                }
                cache[digest] = record
                new_count += 1
                with (run / "native-completion.jsonl").open("a", encoding="utf-8") as f:
                    f.write(encode(record) + "\n")
                store.event(
                    "rag_snn_dev.actual_answer_completion",
                    {
                        "run_id": run.name,
                        "seed": row["seed"],
                        "method": row["method"],
                        "query": case["query"],
                        "selected": row["selected"],
                        "model_answer": raw["choices"][0]["message"].get("content", ""),
                        "expected": case["gold"],
                        "native_calls_total": len(cache),
                        "cached": False,
                        "no_learning": True,
                    },
                    "orchestrator",
                )
            record = cache[digest]
            answer = record["response"]["choices"][0]["message"].get("content", "")
            try:
                value = json.loads(answer)
                correct = (
                    type(value) is dict
                    and set(value) == {"value"}
                    and value["value"] == case["gold"]
                    and (
                        type(value["value"]) is int
                        if case["gold"] is not None
                        else value["value"] is None
                    )
                )
            except (ValueError, TypeError):
                correct = False
            row.update(
                answer=answer,
                answer_correct=bool(correct),
                grounded_correct=bool(correct and row["source_correct"]),
                native_seconds=record["seconds"],
            )
        report["predictions"] = rows
    except Exception as exc:
        errors.append(type(exc).__name__ + ": " + str(exc))
    finally:
        report["parent_failures"] = report["failures"]
        report["failures"] = errors
        report["native_calls"] = len(cache)
        report["completion"] = {
            **lock,
            "seconds": time.monotonic() - started,
            "new_native_inputs": new_count,
            "parent_protocol_budget_compliance": False,
        }
        report["registered_parent_budget_pass"] = False
        (run / "completed-inference.json").write_text(encode(report), encoding="utf-8")
        store.close()
        if learning:
            control("/api/learning", {"enabled": True})
        print(
            encode(
                {
                    "completed": not errors,
                    "new_calls": new_count,
                    "all_unique_calls": len(cache),
                    "seconds": time.monotonic() - started,
                    "failures": errors,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    main(a.run.resolve())
