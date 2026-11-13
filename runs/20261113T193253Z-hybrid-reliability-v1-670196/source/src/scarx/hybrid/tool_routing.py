"""Conservative first-tool routing for an explicit local workspace inspection.

This is an observable controller rule, not a learned neural capability. Tool
schemas/history are unchanged and Hermes retains execution/approval ownership.
"""

import re
import unicodedata
import json
import shlex


def text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def workspace_tool_choice(messages, tools, requested=None):
    if requested not in (None, "auto") or not tools:
        return requested, None
    terminal = any(t.get("function", {}).get("name") == "terminal" for t in tools)
    if not terminal:
        return requested, None
    latest = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "user"
        ),
        None,
    )
    if latest is None or any(
        m.get("tool_calls") or m.get("role") == "tool" for m in messages[latest + 1 :]
    ):
        return requested, None
    query = text_content(messages[latest].get("content", "")).lower()
    query = "".join(
        c for c in unicodedata.normalize("NFKD", query) if not unicodedata.combining(c)
    ).replace("đ", "d")
    inspect = re.search(
        r"\b(doc|xem|phan tich|kiem tra|read|inspect|analy[sz]e|explain)\b", query
    )
    local = re.search(r"\b(workspace|thu muc|folder|project|du an)\b", query)
    explicitly_web = any(
        not re.search(
            r"\b(khong|do not|dont|not)\b.{0,35}$",
            query[max(0, m.start() - 45) : m.start()],
        )
        for m in re.finditer(
            r"\b(tim tren web|tim tren mang|search the web|search online)\b", query
        )
    )
    if not inspect or not local or explicitly_web:
        return requested, None
    cwd = None
    for m in messages:
        if m.get("role") != "system":
            continue
        match = re.search(
            r"(?im)^Current working directory:\s*([^\r\n]+)",
            text_content(m.get("content", "")),
        )
        if match:
            cwd = match[1].strip()
    if not cwd:
        return requested, None
    return {"type": "function", "function": {"name": "terminal"}}, {
        "kind": "explicit_local_workspace_inspection",
        "cwd": cwd,
        "selector": "runtime first-tool rule; not learned policy",
        "first_tool": "terminal",
        "schemas_and_history_unchanged": True,
    }


def workspace_prelude(messages, tools, requested=None):
    """One real read-only terminal request. Never claims to be VLM generation."""
    choice, event = workspace_tool_choice(messages, tools, requested)
    if event is None:
        return None
    code = """import pathlib,json
p=pathlib.Path.cwd()
items=sorted(p.iterdir(),key=lambda f:(not f.is_dir(),f.name.casefold()))
result={"cwd":str(p),"entries":[{"name":f.name,"directory":f.is_dir()} for f in items[:45]],"entry_count":len(items),"files":[]}
preferred=["README.md","README.txt","pyproject.toml","package.json","Cargo.toml","main.py"]
docs=[p/n for n in preferred]
docs += [f for f in items if f.suffix.lower()==".md" and not f.is_symlink()][:3]
seen=set();remaining=14000
for f in docs:
 if f.name in seen or f.is_symlink() or not f.is_file() or remaining<=0:continue
 seen.add(f.name)
 with f.open("r",encoding="utf-8",errors="replace") as stream: text=stream.read(min(5000,remaining))
 result["files"].append({"path":str(f),"text":text,"truncated":f.stat().st_size>len(text.encode("utf-8"))})
 remaining-=len(text)
print(json.dumps(result,ensure_ascii=False))"""
    event = dict(
        event, selector="runtime read-only prelude; not VLM-generated tool call"
    )
    return {
        "name": "terminal",
        "arguments": json.dumps(
            {
                "command": "python -c " + shlex.quote(code),
                "workdir": event["cwd"],
                "timeout": 15,
            },
            ensure_ascii=False,
        ),
        "event": event,
    }
