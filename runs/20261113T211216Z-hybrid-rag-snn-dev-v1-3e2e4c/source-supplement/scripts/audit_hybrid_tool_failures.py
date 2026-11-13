"""Read-only mechanical incident audit; hypotheses are not neural understanding."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def analyze(messages):
    pending = {}
    observations = []
    for row in messages:
        for call in row.get("tool_calls") or []:
            function = call.get("function") or {}
            raw = function.get("arguments", "{}")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except ValueError:
                args = {"unparsed": raw}
            canonical = json.dumps(
                args, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            pending[call.get("id")] = (
                function.get("name", ""),
                hashlib.sha256(canonical.encode()).hexdigest(),
            )
        if row.get("role") == "tool":
            tool, args_hash = pending.get(
                row.get("tool_call_id"), (row.get("tool_name", ""), None)
            )
            content = row.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            # Strip runtime warning suffix before hashing tool payload, so a
            # changing repeat-counter notice does not hide identical results.
            core = (
                content.split("[Tool loop warning:")[0]
                .split("[hermes note:")[0]
                .strip()
            )
            core = core.removesuffix("</untrusted_tool_result>").strip()
            observations.append(
                {
                    "tool": tool,
                    "args_sha256": args_hash,
                    "result_sha256": hashlib.sha256(core.encode()).hexdigest(),
                    "content_chars": len(content),
                    "persisted_output": "<persisted-output>" in content,
                    "guard_reported_repeat": "consecutive identical call" in content
                    or "identical_call_streak" in content,
                    "blocked": "Blocked by plugin" in content
                    or "pre_tool_call" in content,
                    "empty_web": '"web": []' in core or '"web":[]' in core,
                    "warning_or_halt": "Tool loop " in content,
                }
            )
    incidents = []
    streak = 0
    previous = None
    for i, row in enumerate(observations):
        signature = (row["tool"], row["args_sha256"], row["result_sha256"])
        if row["tool"] in ("process_manage", "process"):
            streak = 0
            previous = None
            continue
        streak = streak + 1 if previous == signature else 1
        previous = signature
        if streak == 3:
            incidents.append(
                {
                    "index": i,
                    "kind": "identical_nonprogress_streak",
                    "tool": row["tool"],
                    "verified_scope": "exact repeated tool/arguments/result only",
                    "observed_empty_web": row["empty_web"],
                    "root_cause": "not established by repetition alone",
                    "needed_probe": "inspect workspace/source intent, error status and whether a different authorized source/action resolves the task",
                    "verified_corrective_action": False,
                    "neural_self_awareness": False,
                }
            )
    return {
        "observed_tool_results": len(observations),
        "tools": {
            name: sum(r["tool"] == name for r in observations)
            for name in sorted({r["tool"] for r in observations})
        },
        "observations": observations,
        "incidents": incidents,
        "automatic_repeat_detection_possible": bool(incidents),
        "semantic_error_detection_proven": False,
        "root_cause_learning_proven": False,
        "scope": "read-only reanalysis of actual transcript; deterministic detector, no weight updates or model-generated reasoning",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError("Refuse overwrite incident audit")
    path = Path("C:/Users/hachimi/AppData/Local/hermes/state.db")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "select role,content,tool_calls,tool_call_id,tool_name from messages where session_id=? order by id",
        (a.session,),
    ).fetchall()
    db.close()
    messages = []
    for row in rows:
        d = dict(row)
        d["tool_calls"] = json.loads(d["tool_calls"]) if d["tool_calls"] else []
        messages.append(d)
    report = analyze(messages)
    report["session_sha256"] = hashlib.sha256(a.session.encode()).hexdigest()
    report["raw_transcript_sha256"] = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "observations"}))


if __name__ == "__main__":
    main()
