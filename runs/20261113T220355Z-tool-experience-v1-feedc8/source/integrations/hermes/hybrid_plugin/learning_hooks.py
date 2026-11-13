"""Published Hermes observation hooks; no raw private tool contents leave Hermes."""

from collections import OrderedDict
import hashlib
import json
import threading
import time
import logging
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
    if isinstance(value, dict):
        # Omit transport timing/identity fields, not arbitrary source data.
        value = {
            k: v
            for k, v in value.items()
            if k
            not in (
                "duration",
                "duration_ms",
                "elapsed",
                "elapsed_seconds",
                "timestamp",
                "tool_call_id",
                "pid",
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
    def __init__(self, send):
        self.send = send
        self.routes = OrderedDict()
        self.lock = threading.Lock()
        self.failures = 0

    def pre_api(self, model="", base_url="", session_id="", turn_id="", **kwargs):
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
            if accepted:
                self.routes[key] = time.monotonic()
            while len(self.routes) > 64:
                self.routes.popitem(last=False)

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
    observer = ToolExperienceObserver(send)
    ctx.register_hook("pre_api_request", observer.pre_api)
    ctx.register_hook("post_tool_call", observer.post_tool)
    return observer
