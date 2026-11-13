"""One identified local Hermes-owner turn; no private ticket/history export."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from websockets.asyncio.client import connect


async def probe(run):
    binary = Path(os.environ["LOCALAPPDATA"]) / "hermes/bin/hermes.exe"
    grant = json.loads(
        subprocess.check_output(
            [str(binary), "--profile", "default", "gateway", "attach", "--json"],
            text=True,
        )
    )
    record = {
        "scope": "one actual native Hermes-owner local turn; no general agent-quality claim",
        "prompt": "Reply exactly PIPELINE_OK. Do not call any tools.",
        "expected": "PIPELINE_OK",
        "model": "scarx-hybrid",
        "provider": "custom:scarx-local",
        "max_turns": 2,
    }
    async with connect(
        grant["url"],
        subprotocols=grant["protocols"],
        proxy=None,
        max_size=8 * 1024 * 1024,
    ) as ws:
        counter = 0

        async def rpc(method, params, timeout=20):
            nonlocal counter
            counter += 1
            identifier = counter
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": identifier,
                        "method": method,
                        "params": params,
                    }
                )
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                response = json.loads(
                    await asyncio.wait_for(
                        ws.recv(), max(0.1, deadline - time.monotonic())
                    )
                )
                if response.get("id") == identifier:
                    if response.get("error"):
                        raise RuntimeError(str(response["error"]))
                    return response["result"]
            raise TimeoutError(method)

        sid = None
        snapshot = None
        try:
            root = Path(__file__).resolve().parents[1]
            cwd = root / "local" / run.name / "private-hermes-workspace"
            cwd.mkdir(exist_ok=False)
            started = time.monotonic()
            created = await rpc(
                "session.create",
                {
                    "request_id": uuid.uuid4().hex,
                    "source": "gui",
                    "profile": "default",
                    "cwd": str(cwd),
                    "cwd_explicit": True,
                    "model": "scarx-hybrid",
                    "provider": "custom:scarx-local",
                    "max_turns": 2,
                    "title": "SCAR-X pipeline audit",
                },
            )
            sid = created.get("session_id") or created.get("id")
            if not sid:
                raise ValueError("Owner did not return session identity")
            record["session_id"] = sid
            admitted = await rpc(
                "prompt.submit",
                {
                    "session_id": sid,
                    "profile": "default",
                    "submission_id": uuid.uuid4().hex,
                    "text": record["prompt"],
                },
                timeout=20,
            )
            record["admission_id"] = admitted.get("admission_id")
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                snapshot = await rpc(
                    "session.resume",
                    {
                        "session_id": sid,
                        "profile": "default",
                        "display_projection": True,
                    },
                    timeout=15,
                )
                rows = snapshot.get("display_messages") or []
                answers = [
                    r
                    for r in rows
                    if r.get("role") == "assistant"
                    and (r.get("text") or r.get("content"))
                ]
                record["tool_call_count"] = sum(
                    len(r.get("tool_calls") or []) for r in rows
                )
                if not snapshot.get("running") and answers:
                    content = answers[-1].get("text") or answers[-1].get("content")
                    record["answer"] = (
                        content if isinstance(content, str) else json.dumps(content)
                    )
                    record["correct"] = record["answer"].strip() == record["expected"]
                    record["elapsed_seconds"] = time.monotonic() - started
                    record["finished"] = True
                    break
                await asyncio.sleep(1)
            else:
                record["finished"] = False
                record["timeout_seconds"] = 90
        except Exception as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
        finally:
            if sid and not record.get("finished"):
                try:
                    snapshot = await rpc(
                        "session.resume",
                        {
                            "session_id": sid,
                            "profile": "default",
                            "omit_messages": True,
                        },
                    )
                    if snapshot.get("running"):
                        await rpc(
                            "session.interrupt",
                            {
                                "session_id": sid,
                                "profile": "default",
                                "execution_generation": snapshot[
                                    "execution_generation"
                                ],
                            },
                        )
                        record["own_test_turn_interrupted"] = True
                except Exception as exc:
                    record["cleanup_error"] = str(exc)
    with (run / "hermes-owner-probe.json").open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    asyncio.run(probe(args.run.resolve()))
