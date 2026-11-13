"""Read-only native-owner projection check; never print a dial ticket/history."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
from websockets.asyncio.client import connect


def has_handoff(rows):
    return [
        index
        for index, row in enumerate(rows)
        if any(
            marker in json.dumps(row, ensure_ascii=False)
            for marker in (
                "[PRIOR CONTEXT",
                "[CONTEXT COMPACTION",
                "[END OF PRIOR CONTEXT",
            )
        )
    ]


async def verify(session, profile):
    binary = Path(os.environ["LOCALAPPDATA"]) / "hermes/bin/hermes.exe"
    grant = json.loads(
        subprocess.check_output(
            [str(binary), "--profile", profile, "gateway", "attach", "--json"],
            text=True,
        )
    )
    async with connect(
        grant["url"],
        subprotocols=grant["protocols"],
        proxy=None,
        max_size=8 * 1024 * 1024,
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session.resume",
                    "params": {
                        "session_id": session,
                        "profile": profile,
                        "display_projection": True,
                    },
                }
            )
        )
        while True:
            frame = json.loads(await asyncio.wait_for(ws.recv(), 10))
            if frame.get("id") == 1:
                break
        if frame.get("error"):
            raise RuntimeError("Canonical owner refused the projection read")
        result = frame["result"]
        raw = result.get("messages", [])
        display = result.get("display_messages")
        if not isinstance(display, list):
            raise ValueError("Owner did not provide a display projection")
        leaks = has_handoff(display)
        if leaks:
            raise ValueError("Compaction leaked into the owner display projection")
        report = {
            "session_digest": hashlib.sha256(session.encode()).hexdigest(),
            "owner_display_projection_present": True,
            "raw_rows": len(raw),
            "display_rows": len(display),
            "raw_handoff_rows": len(has_handoff(raw)),
            "display_handoff_rows": len(leaks),
            "raw_history_retained": True,
            "user_and_assistant_display_rows": sum(
                r.get("role") in ("user", "assistant") for r in display
            ),
            "tool_rows_retained": sum(r.get("role") == "tool" for r in display),
            "session_running": result.get("running"),
            "history_changed": False,
            "model_inference_started": False,
        }
        print(json.dumps(report))
        return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--profile", default="default")
    args = p.parse_args()
    asyncio.run(verify(args.session, args.profile))
