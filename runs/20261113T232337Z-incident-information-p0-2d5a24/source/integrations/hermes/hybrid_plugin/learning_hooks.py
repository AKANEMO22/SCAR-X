"""Published Hermes observation hooks; no raw private tool contents leave Hermes."""

from collections import OrderedDict
import hashlib
import json
import threading
import time
import logging
import re
import ntpath
import posixpath
import unicodedata
from urllib.parse import urlsplit


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def tool_observation(
    tool_name, args, result, scope, call, status="unknown", error_type="", duration_ms=0
):
    value = result
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    if (
        isinstance(value, dict)
        and tool_name == "terminal"
        and "output" in value
        and "exit_code" in value
    ):
        # Only known terminal-envelope wall timing is transport metadata. Keep
        # cwd, pid, source timestamps and arbitrary tool data in the digest.
        value = {
            k: v
            for k, v in value.items()
            if k
            not in (
                "duration",
                "elapsed",
                "elapsed_seconds",
            )
        }
    raw = (
        result
        if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False, sort_keys=True)
    )
    empty = isinstance(value, dict) and (
        (isinstance(value.get("data"), dict) and value["data"].get("web") == [])
        or value.get("results") == []
    )
    return {
        "scope": scope,
        "call_id": digest(call),
        "tool": tool_name,
        "args_sha256": digest(args),
        "result_sha256": digest(value),
        "reported_status": status if status in ("ok", "error") else "unknown",
        "error_type": str(error_type or "")[:120],
        "empty_result": bool(empty),
        "spillover": "<persisted-output>" in raw,
        "duration_ms": max(0, min(float(duration_ms or 0), 86_400_000)),
    }


class ToolExperienceObserver:
    def __init__(self, send, probe=None):
        self.send = send
        self.probe = probe
        self.contexts = OrderedDict()
        self.previous = {}
        self.seen_calls = {}
        self.probed = set()
        self.routes = OrderedDict()
        self.tasks = OrderedDict()
        self.lock = threading.Lock()
        self.failures = 0

    def pre_api(
        self, model="", base_url="", session_id="", turn_id="", task_id="", **kwargs
    ):
        key = (str(session_id), str(turn_id))
        url = urlsplit(str(base_url))
        accepted = (
            model == "scarx-hybrid"
            and url.scheme == "http"
            and url.hostname in ("127.0.0.1", "localhost")
            and url.port == 8775
            and url.path.rstrip("/") == "/v1"
            and all(key)
        )
        with self.lock:
            self.routes.pop(key, None)
            self.contexts.pop(key, None)
            if task_id:
                self.tasks.pop(str(task_id), None)
            if accepted:
                self.routes[key] = time.monotonic()
                request = kwargs.get("request") or {}
                tools = request.get("tools", []) if isinstance(request, dict) else []
                message = kwargs.get("user_message", "")
                message = message[:2000] if isinstance(message, str) else ""
                plain = "".join(
                    c
                    for c in unicodedata.normalize("NFKD", message.casefold())
                    if not unicodedata.combining(c)
                ).replace("đ", "d")
                self.contexts[key] = {
                    "file_tool": "read_file"
                    in {
                        t.get("function", {}).get("name")
                        for t in tools
                        if isinstance(t, dict)
                    },
                    "local_intent": bool(
                        re.search(
                            r"\b(workspace|folder|thu muc|du an|project)\b", plain
                        )
                    ),
                }
                if task_id:
                    self.tasks[str(task_id)] = (str(session_id), time.monotonic())
            else:
                self.previous.pop(key, None)
                self.seen_calls.pop(key, None)
                self.probed.discard(key)
            while len(self.routes) > 64:
                old, _ = self.routes.popitem(last=False)
                self.contexts.pop(old, None)
                self.previous.pop(old, None)
                self.seen_calls.pop(old, None)
                self.probed.discard(old)
            while len(self.tasks) > 64:
                self.tasks.popitem(last=False)

    def session_for_task(self, task_id):
        with self.lock:
            entry = self.tasks.get(str(task_id))
        return entry[0] if entry and time.monotonic() - entry[1] < 3600 else None

    def post_tool(
        self,
        tool_name="",
        args=None,
        result=None,
        session_id="",
        turn_id="",
        tool_call_id="",
        status="unknown",
        error_type="",
        duration_ms=0,
        **kwargs,
    ):
        key = (str(session_id), str(turn_id))
        with self.lock:
            stamp = self.routes.get(key)
        if stamp is None or time.monotonic() - stamp > 3600 or not tool_call_id:
            return
        try:
            payload = tool_observation(
                tool_name,
                args or {},
                result,
                digest(key),
                tool_call_id,
                status,
                error_type,
                duration_ms,
            )
            with self.lock:
                if key not in self.routes:
                    return
                calls = self.seen_calls.setdefault(key, OrderedDict())
                if payload["call_id"] in calls:
                    return
                calls[payload["call_id"]] = True
                while len(calls) > 64:
                    calls.popitem(last=False)
                signature = (
                    payload["tool"] + payload["args_sha256"] + payload["result_sha256"]
                )
                old_sig, count = self.previous.get(key, (None, 0))
                count = count + 1 if old_sig == signature else 1
                self.previous[key] = (signature, count)
                context = self.contexts.get(key, {})
                should_probe = (
                    count == 3 and self.probe and context.get("file_tool")
                    and tool_name in ("read_file", "web_search")
                    and key not in self.probed
                )
                if should_probe:
                    self.probed.add(key)
            if should_probe:
                detail = self.probe(
                    tool_name,
                    args or {},
                    result,
                    context,
                    kwargs.get("task_id") or session_id,
                )
                if detail:
                    payload["probe"] = detail
            self.send(payload)
        except Exception as exc:
            # Observers cannot break the user's real tool execution. Failures
            # are bounded telemetry, never treated as successful learning.
            self.failures = min(self.failures + 1, 1_000_000)
            if self.failures == 1:
                logging.getLogger(__name__).warning(
                    "SCAR-X local tool observer unavailable: %s", type(exc).__name__
                )


def register_observer(ctx, send):
    observer = ToolExperienceObserver(
        send,
        lambda tool, args, result, context, task: readonly_probe(
            ctx, tool, args, result, context, task
        ),
    )
    ctx.register_hook("pre_api_request", observer.pre_api)
    ctx.register_hook("post_tool_call", observer.post_tool)
    return observer


def readonly_probe(ctx, tool, args, result, context, task_id):
    if not context.get("file_tool"):
        return None
    try:
        original = json.loads(result) if isinstance(result, str) else result
    except ValueError:
        return None
    name = None
    kind = None
    requested = None
    if (
        tool == "read_file"
        and isinstance(original, dict)
        and original.get("not_found") is True
    ):
        requested = args.get("path")
        if not isinstance(requested, str):
            return None
        name = posixpath.basename(requested.replace("\\", "/"))
        kind = "requested_path_binding"
    elif (
        tool == "web_search"
        and context.get("local_intent")
        and isinstance(original, dict)
        and isinstance(original.get("data"), dict)
        and original["data"].get("web") == []
    ):
        name = "README.md"
        kind = "local_source_check"
    if (
        not name
        or not re.fullmatch(r"[A-Za-z0-9_. -]{1,100}", name)
        or not name.lower().endswith((".md", ".txt"))
        or re.search(r"secret|password|credential|private|token|^\.", name, re.I)
    ):
        return None
    cwd = ctx.session_cwd()
    if not isinstance(cwd, str) or not cwd or len(cwd) > 2048:
        return None
    paths = posixpath if cwd.startswith("/") else ntpath
    target = paths.normpath(paths.join(cwd, name))
    base = paths.normpath(cwd)
    if paths.normcase(paths.dirname(target)) != paths.normcase(base):
        return None
    differs = requested is None or paths.normcase(
        paths.normpath(
            requested if paths.isabs(requested) else paths.join(cwd, requested)
        )
    ) != paths.normcase(target)
    if not differs:
        return None
    response = ctx.dispatch_tool(
        "read_file", {"path": target, "offset": 1, "limit": 12}, task_id=task_id
    )
    if not isinstance(response, str) or len(response) > 2_000_000:
        return None
    try:
        data = json.loads(response)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    read_ok = (
        not data.get("error")
        and not data.get("not_found")
        and not data.get("is_binary")
        and isinstance(data.get("content"), str)
    )
    return {
        "kind": kind,
        "source_name": name,
        "project_label": paths.basename(base)[:80],
        "original_path_differs": bool(differs),
        "read_ok": bool(read_ok),
        "reported_not_found": data.get("not_found") is True,
        "result_sha256": digest(data),
        "scope": "recorded_project_file_tool",
        "semantic_resolution_verified": False,
    }
