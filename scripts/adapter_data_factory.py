"""Bounded actual AGY High/High wording generation; no private data upload."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import Store, encode

MODEL = "gemini-3.8-flash-high"
CLI = Path(os.environ["LOCALAPPDATA"]) / "agy/bin/agy.exe"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if (ROOT / "artifacts/budget-locks/adapter-data-v1.json").exists():
        raise ValueError(
            "This two-call dataset budget is closed; use a new reviewed dataset version"
        )
    run_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-adapter-data-v1-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / run_id
    run.mkdir()
    private = ROOT / "local" / run_id
    private.mkdir()
    candidates = ROOT / "data/candidates" / run_id
    candidates.mkdir(parents=True)
    store = Store(ROOT / "local/hybrid-v1")

    def event(kind, payload):
        store.event("adapter_data." + kind, payload, "orchestrator")
        print(encode({"kind": kind, **payload}), flush=True)

    version = subprocess.check_output(
        [str(CLI), "--version"], text=True, timeout=10
    ).strip()
    labels = ("base", "memory", "repair", "verify", "reuse_adapter", "propose_adapter")
    all_rows = []
    receipts = []
    for batch in range(2):
        ids = [
            f"{label}-{i}"
            for label in labels
            for i in range(batch * 8, (batch + 1) * 8)
        ]
        properties = {
            "id": {"type": "string", "enum": ids},
            "vi": {"type": "string"},
            "en": {"type": "string"},
        }
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 48,
                    "maxItems": 48,
                    "items": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties),
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }
        job = private / str(batch)
        job.mkdir()
        (job / "schema.json").write_text(encode(schema), encoding="utf-8")
        prompt = (
            """Generate synthetic bilingual scenario wordings for teaching a local AI how to manage learning. Do not use tools or inspect files. Output only the requested JSON schema. One entry for EVERY ID, no missing/duplicate IDs. Each vi/en must be a natural task statement 12-45 words, describing only the user's task; do NOT mention the routing label, LoRA, adapter, statistics, hidden test, or prescribe a decision. Never claim facts about a real user. Use invented/public toy tasks. Vary Vietnamese and English phrasing. Six ID groups describe: base=ordinary arithmetic or simple formatting already known; memory=one-off personal preference or new event fact; repair=bad crop/missing audio/empty retrieved file; verify=ambiguous/unverified/contradictory instruction; reuse_adapter=recurring specialized structured-output or glossary task already supported by a reusable skill; propose_adapter=recurring specialized output convention with many corrected examples. Keep the task itself distinct from workflow decision. Scenario measurements/labels are independently supplied by our harness, you do NOT invent evaluation results.\nIDS: """
            + json.dumps(ids)
        )
        (job / "prompt.txt").write_text(prompt, encoding="utf-8")
        command = [
            str(CLI),
            "--print",
            prompt,
            "--model",
            MODEL,
            "--effort",
            "high",
            "--mode",
            "plan",
            "--sandbox",
            "--disable-slash-commands",
            "--output-format",
            "stream-json",
            "--json-schema",
            str(job / "schema.json"),
            "--print-timeout",
            "4m",
        ]
        event(
            "started",
            {
                "run": run_id,
                "batch": batch + 1,
                "model_requested": MODEL,
                "effort": "high",
                "synthetic_only": True,
            },
        )
        started = time.monotonic()
        error = None
        items = []
        reported = None
        usage = None
        proc = None
        with (
            (job / "events.jsonl").open("xb") as out,
            (job / "stderr.log").open("xb") as err,
        ):
            try:
                proc = subprocess.Popen(
                    command,
                    cwd=job,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                while proc.poll() is None:
                    if time.monotonic() - started > 255:
                        raise TimeoutError("Registered call deadline")
                    if (job / "events.jsonl").stat().st_size > 10_000_000:
                        raise ValueError("Stream cap")
                    time.sleep(0.5)
                events = []
                for line in (
                    (job / "events.jsonl").read_text(encoding="utf-8-sig").splitlines()
                ):
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        continue
                inits = [e["init"] for e in events if e.get("event") == "init"]
                reported = inits[-1].get("model") if inits else None
                if reported != MODEL:
                    raise ValueError("Reported model mismatch")
                result = [e["result"] for e in events if e.get("event") == "result"]
                if (
                    proc.returncode
                    or not result
                    or result[-1].get("status") != "SUCCESS"
                ):
                    raise ValueError("AGY did not complete SUCCESS")
                result = result[-1]
                output = result.get("structured_output")
                if output is None:
                    output = json.loads(result["response"])
                items = output["items"]
                if len(items) != 48 or {r["id"] for r in items} != set(ids):
                    raise ValueError("Wrong IDs/count")
                if any(
                    set(r) != set(properties)
                    or not all(
                        isinstance(r[k], str) and 8 <= len(r[k]) <= 1500
                        for k in ("vi", "en")
                    )
                    for r in items
                ):
                    raise ValueError("Bad wording schema")
                toolsteps = [
                    e["step_update"]
                    for e in events
                    if e.get("step_update", {}).get("step_type") == "tool"
                ]
                if any(s.get("tool_name") != "finish" for s in toolsteps):
                    raise ValueError("Unexpected teacher tool invocation; quarantine")
                usage = {
                    k: v
                    for k, v in result.get("usage", {}).items()
                    if k
                    in (
                        "input_tokens",
                        "output_tokens",
                        "thinking_tokens",
                        "total_tokens",
                    )
                }
            except Exception as exc:
                error = type(exc).__name__ + ": " + str(exc)
                items = []
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        receipt = {
            "batch": batch + 1,
            "model_requested": MODEL,
            "model_reported": reported,
            "effort_requested": "high",
            "items": len(items),
            "seconds": time.monotonic() - started,
            "error": error,
            "usage": usage,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "raw_stream_sha256": hashlib.sha256(
                (job / "events.jsonl").read_bytes()
            ).hexdigest(),
        }
        receipts.append(receipt)
        all_rows += items
        (run / f"batch-{batch+1}.json").write_text(encode(receipt), encoding="utf-8")
        (candidates / f"batch-{batch+1}.json").write_text(
            encode(
                {"items": items, "teacher_labels_trusted": False, "synthetic": True}
            ),
            encoding="utf-8",
        )
        event("finished", receipt | {"run": run_id})
        if error:
            break
    result = {
        "run": run_id,
        "cli_version": version,
        "model": MODEL,
        "effort": "high",
        "items": len(all_rows),
        "data": str(candidates.relative_to(ROOT)),
        "batches": receipts,
        "automatic_training_labels": False,
        "private_data_sent": False,
        "status": (
            "WORDING_READY_REVIEW_REQUIRED"
            if len(all_rows) == 96
            else "INCOMPLETE_NO_AUTOMATIC_RETRY"
        ),
    }
    (candidates / "wordings.json").write_text(encode(all_rows), encoding="utf-8")
    (run / "receipt.json").write_text(encode(result), encoding="utf-8")
    event("complete", result)
    store.close()


if __name__ == "__main__":
    main()
