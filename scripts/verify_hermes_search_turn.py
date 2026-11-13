"""One bounded actual Hermes-agent web turn; no old user session is modified."""

import asyncio
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from websockets.asyncio.client import connect


async def main(label=None):
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/20261113T160831Z-hermes-free-web-v1"
    output = run / (
        "hermes-agent-turn-" + label + ".json" if label else "hermes-agent-turn.json"
    )
    if output.exists() or output.with_name(output.stem + ".public.json").exists():
        raise ValueError("No overwrite or retry")
    work = root / "local/hermes-free-web-probe" / uuid.uuid4().hex
    work.mkdir(parents=True)
    binary = Path(os.environ["LOCALAPPDATA"]) / "hermes/bin/hermes.exe"
    grant = json.loads(
        subprocess.check_output([str(binary), "gateway", "attach", "--json"], text=True)
    )
    prompt = "Hãy dùng web_search tìm nguồn Chính phủ về số thành phố trực thuộc trung ương Việt Nam hiện nay. Nếu cần tìm công cụ, tìm đúng tên web_search. Đọc kết quả rồi trả lời ngắn gọn bằng tiếng Việt, kèm URL nguồn. Không trả lời bằng trí nhớ cũ."
    result = {
        "prompt": prompt,
        "model": "scarx-hybrid",
        "provider": "custom:scarx-local",
        "max_turns": 5,
        "scope": "one actual tool-using engineering turn, not a capability benchmark",
    }
    async with connect(
        grant["url"],
        subprotocols=grant["protocols"],
        proxy=None,
        max_size=8 * 1024 * 1024,
    ) as ws:
        counter = 0

        async def rpc(method, params, timeout=15):
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
                r = json.loads(
                    await asyncio.wait_for(
                        ws.recv(), max(0.1, deadline - time.monotonic())
                    )
                )
                if r.get("id") == identifier:
                    if r.get("error"):
                        raise RuntimeError(str(r["error"]))
                    return r["result"]
            raise TimeoutError(method)

        sid = None
        try:
            start = time.monotonic()
            created = await rpc(
                "session.create",
                {
                    "request_id": uuid.uuid4().hex,
                    "source": "gui",
                    "profile": "default",
                    "cwd": str(work),
                    "cwd_explicit": True,
                    "title": "SCAR-X web search verification",
                    "model": "scarx-hybrid",
                    "provider": "custom:scarx-local",
                    "max_turns": 5,
                    "toolsets": ["web"],
                },
            )
            sid = created.get("session_id") or created.get("id")
            result["session_id"] = sid
            await rpc(
                "prompt.submit",
                {
                    "session_id": sid,
                    "profile": "default",
                    "submission_id": uuid.uuid4().hex,
                    "text": prompt,
                },
                timeout=20,
            )
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                s = await rpc(
                    "session.resume",
                    {
                        "session_id": sid,
                        "profile": "default",
                        "display_projection": True,
                    },
                )
                display = s.get("display_messages") or []
                # Only this newly created public test turn is recorded.
                result["display_messages"] = display
                result["tool_calls"] = [
                    c
                    for row in s.get("messages", [])
                    for c in row.get("tool_calls", [])
                ]
                if not s.get("running") and any(
                    row.get("role") == "assistant"
                    and (row.get("text") or row.get("content"))
                    for row in display
                ):
                    result["finished"] = True
                    result["seconds"] = time.monotonic() - start
                    break
                await asyncio.sleep(1)
            else:
                result.update(finished=False, timeout_seconds=150)
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            if sid and not result.get("finished"):
                s = await rpc(
                    "session.resume",
                    {"session_id": sid, "profile": "default", "omit_messages": True},
                )
                if s.get("running"):
                    await rpc(
                        "session.interrupt",
                        {
                            "session_id": sid,
                            "profile": "default",
                            "execution_generation": s["execution_generation"],
                        },
                    )
                    result["own_test_interrupted"] = True
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    answers = [
        r.get("text") or r.get("content")
        for r in result.get("display_messages", [])
        if r.get("role") == "assistant"
    ]
    print(
        json.dumps(
            {
                "finished": result.get("finished"),
                "seconds": result.get("seconds"),
                "answer": answers[-1] if answers else None,
                "tool_calls": result.get("tool_calls"),
                "error": result.get("error"),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["eager", "fresh"])
    args = parser.parse_args()
    asyncio.run(main(args.label))
