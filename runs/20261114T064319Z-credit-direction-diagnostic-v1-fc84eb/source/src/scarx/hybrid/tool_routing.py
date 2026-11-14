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


def task_user_index(messages):
    scaffolding = (
        "You just executed tool calls but returned an empty response. Please process the tool results above and continue with the task.",
        "You've reached the maximum number of tool-calling iterations allowed. Please provide a final response summarizing what you've found and accomplished so far, without calling any more tools.",
    )
    return next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "user"
            and text_content(messages[i].get("content", "")) not in scaffolding
        ),
        None,
    )


def workspace_tool_choice(messages, tools, requested=None, *, after_tools=False):
    if requested not in (None, "auto") or not tools:
        return requested, None
    terminal = any(t.get("function", {}).get("name") == "terminal" for t in tools)
    if not terminal:
        return requested, None
    latest = task_user_index(messages)
    if latest is None or (
        not after_tools
        and any(
            m.get("tool_calls") or m.get("role") == "tool"
            for m in messages[latest + 1 :]
        )
    ):
        return requested, None
    query = text_content(messages[latest].get("content", "")).lower()
    query = "".join(
        c for c in unicodedata.normalize("NFKD", query) if not unicodedata.combining(c)
    ).replace("đ", "d")
    inspect = re.search(
        r"\b(doc|xem|phan tich|kiem tra|read|inspect|analy[sz]e|explain)\b", query
    ) or re.search(r"\bwhat .{0,80}(does|is)\b", query)
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
        matches = re.findall(
            r"(?im)^Current working directory:\s*([^\r\n]+)",
            text_content(m.get("content", "")),
        )
        if matches:
            # Hermes appends the actual runtime block after project prose.
            cwd = matches[-1].strip()
    if not cwd:
        return requested, None
    return {"type": "function", "function": {"name": "terminal"}}, {
        "kind": "explicit_local_workspace_inspection",
        "cwd": cwd,
        "selector": "runtime first-tool rule; not learned policy",
        "first_tool": "terminal",
        "schemas_and_history_unchanged": True,
    }


def workspace_finalization(messages, tools, requested=None):
    _, event = workspace_tool_choice(messages, tools, requested, after_tools=True)
    if event is None:
        return requested, None
    latest = task_user_index(messages)
    query = text_content(messages[latest].get("content", "")).lower()
    query = "".join(
        c for c in unicodedata.normalize("NFKD", query) if not unicodedata.combining(c)
    ).replace("đ", "d")
    overview = re.search(r"\b(lam gi|purpose|overview|what .{0,30} do)\b", query)
    editing = any(
        not re.search(
            r"\b(khong|do not|dont|not)\b.{0,20}$",
            query[max(0, m.start() - 30) : m.start()],
        )
        for m in re.finditer(
            r"\b(sua|fix|debug|implement|patch|refactor|viet code|add feature)\b", query
        )
    )
    results = sum(m.get("role") == "tool" for m in messages[latest + 1 :])
    if not overview or editing or results < 4:
        return requested, None
    return "none", {
        "kind": "workspace_overview_budget",
        "tool_results": results,
        "limit": 4,
        "cwd": event["cwd"],
        "selector": "runtime finalization boundary; VLM must answer from actual sources",
        "schemas_and_history_unchanged": True,
    }


def workspace_halt_text(messages, event):
    """Controlled failure notice with raw observations, never a model conclusion."""
    lines = [
        "Trạng thái từ runtime SCAR-X: đã dừng vòng tìm kiếm sau "
        + str(event["tool_results"])
        + " lượt đọc để tránh lặp.",
        "Workspace: `" + event["cwd"] + "`.",
        "Model hiện tại chưa tạo được bản phân tích đáng tin cậy. Đây là thông báo điều phối, không phải kết luận do model sinh.",
    ]
    call_ids = {
        tc.get("id")
        for m in messages
        if m.get("role") == "assistant"
        for tc in m.get("tool_calls") or []
        if str(tc.get("id", "")).startswith("scarx_read_")
    }
    for m in messages:
        if m.get("role") != "tool" or m.get("tool_call_id") not in call_ids:
            continue
        raw = text_content(m.get("content", ""))
        try:
            outer = json.JSONDecoder().raw_decode(raw[raw.index("{") :])[0]
            data = json.loads(outer.get("output", "{}"))
        except (ValueError, TypeError, AttributeError):
            continue
        if not isinstance(data, dict) or data.get("cwd") != event["cwd"]:
            continue
        lines.append(
            "Dữ liệu từ lượt đọc thực: "
            + str(data.get("entry_count", "?"))
            + " mục ở thư mục gốc."
        )
        for marker in data.get("manifest_markers", [])[:4]:
            lines.append(
                "- "
                + str(marker.get("format", ""))
                + ": `"
                + str(marker.get("path", ""))
                + "`"
                + (
                    " · package `" + str(marker["package"]) + "`"
                    if marker.get("package")
                    else ""
                )
            )
        for f in data.get("files", [])[:4]:
            lines.append(
                "- Đã đọc `"
                + str(f.get("path", ""))
                + "`"
                + (" (trích đoạn)" if f.get("truncated") else "")
            )
        break
    return "\n\n".join(lines)


def workspace_prelude(messages, tools, requested=None):
    """One real read-only terminal request. Never claims to be VLM generation."""
    choice, event = workspace_tool_choice(messages, tools, requested)
    if event is None:
        return None
    code = """import pathlib,json,re
p=pathlib.Path.cwd()
items=sorted(p.iterdir(),key=lambda f:(not f.is_dir(),f.name.casefold()))
result={"cwd":str(p),"entries":[{"name":f.name,"directory":f.is_dir()} for f in items[:24]],"entry_count":len(items),"listing_truncated":len(items)>24,"files":[]}
result["root_source_extensions"]={ext:sum(f.is_file() and f.suffix.lower()==ext for f in items) for ext in (".smali",".py",".js",".ts",".kt",".java")}
result["manifest_markers"]=[]
preferred=["README.md","README.txt","pyproject.toml","package.json","Cargo.toml","main.py","AndroidManifest.xml","apktool.yml","build.gradle","settings.gradle"]
docs=[p/n for n in preferred]
folders=[f for f in items if f.is_dir() and not f.is_symlink() and f.resolve().is_relative_to(p.resolve()) and f.name not in (".git","node_modules",".venv","local")][:24]
for folder in folders:
 docs += [folder/n for n in ("AndroidManifest.xml","apktool.yml","package.json","pyproject.toml","Cargo.toml","build.gradle")]
docs += [f for f in items if f.suffix.lower() in (".smali",".py",".js",".ts",".kt",".java") and not f.is_symlink()][:3]
docs += [f for f in items if f.suffix.lower()==".md" and not f.is_symlink()][:3]
seen=set();remaining=2800
for f in docs:
 if str(f) in seen or f.is_symlink() or not f.is_file() or not f.resolve().is_relative_to(p.resolve()) or remaining<=0:continue
 seen.add(str(f))
 with f.open("rb") as stream: raw=stream.read(min(1000,remaining))
 if b'\\x00' in raw:continue
 text=raw.decode("utf-8",errors="replace")
 if f.name=="AndroidManifest.xml":
  package=re.search(r'package="([^"]+)"',text)
  result["manifest_markers"].append({"path":str(f),"format":"Android application manifest","package":package.group(1) if package else None})
 if f.suffix.lower()==".smali":
  cls=re.search(r'(?m)^\\.class[^\\n]*',text)
  result["manifest_markers"].append({"path":str(f),"format":"Smali Android bytecode source","class_header":cls.group(0) if cls else None})
 result["files"].append({"path":str(f),"text":text,"truncated":f.stat().st_size>len(text.encode("utf-8"))})
 remaining-=len(text)
result["payload_truncated"]=False
for _ in range(200):
 encoded=json.dumps(result,ensure_ascii=False)
 if len(encoded)<=3200:break
 result["payload_truncated"]=True
 longest=max(result["files"],key=lambda f:len(f["text"]),default=None)
 if longest and len(longest["text"])>160:
  longest["text"]=longest["text"][:len(longest["text"])//2];longest["truncated"]=True
 elif len(result["entries"])>6:
  result["entries"]=result["entries"][:len(result["entries"])//2];result["listing_truncated"]=True
 elif len(result["files"])>2:result["files"].pop()
 elif len(result["manifest_markers"])>2:result["manifest_markers"].pop()
 else:break
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


def is_context_summary(messages, tools=None):
    if tools:
        return False
    user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    text = text_content(user.get("content", ""))
    return (
        text.startswith("You are a summarization agent creating a context checkpoint.")
        and "TURNS TO SUMMARIZE:" in text
        and "## Goal" in text
    )
