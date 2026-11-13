"""Normal Hermes tool integration; no separate conversational surface."""

import json
from pathlib import Path
import importlib.util

_tool_observer = None

SCHEMA = {
    "name": "scarx_video",
    "description": "Use when the user asks to look at their screen or the video they are watching. "
    "sources lists visible windows; watch captures the requested window and optional speaker audio locally; "
    "stop ends sharing; status reports analyzed observations. Never guess which of several windows the user means. "
    "Never use a microphone. Capture requires a recent explicit user request; start before playing or ask the user to replay.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["sources", "watch", "stop", "status"],
            },
            "hwnd": {
                "type": "integer",
                "description": "Window id from sources, required for watch",
            },
            "system_audio": {
                "type": "boolean",
                "description": "Speaker loopback only when the user requested listening to the video",
            },
            "stream_id": {"type": "string", "description": "Share identity to stop"},
        },
        "required": ["action"],
    },
}


def _backend():
    spec = importlib.util.spec_from_file_location(
        "_scarx_local_plugin_backend", Path(__file__).parent / "dashboard/plugin_api.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def handler(args, **kwargs):
    backend = _backend()
    action = args.get("action")
    session = (
        kwargs.get("session_id")
        or (
            _tool_observer.session_for_task(kwargs.get("task_id"))
            if _tool_observer
            else None
        )
        or "local"
    )
    if action == "sources":
        result = backend.forward("/api/screen/sources")
    elif action == "watch":
        result = backend.forward(
            "/api/screen/watch-request",
            {
                "hwnd": args.get("hwnd"),
                "system_audio": args.get("system_audio", False),
                "scarx_session_id": session,
            },
        )
    elif action == "stop":
        result = backend.forward("/api/screen/stop", {"id": args.get("stream_id")})
    elif action == "status":
        state = backend.forward("/api/status?after=2147483647")
        result = {
            "streams": state["streams"],
            "ready": state["ready"],
            "failure": state["failure"],
        }
    else:
        raise ValueError("Unknown video action")
    return json.dumps(result, ensure_ascii=False)


def register(ctx):
    global _tool_observer
    # Keep hook code inside the standalone plugin, use only public ctx hooks.
    spec = importlib.util.spec_from_file_location(
        "_scarx_learning_hooks", Path(__file__).with_name("learning_hooks.py")
    )
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    _tool_observer = hooks.register_observer(
        ctx, lambda payload: _backend().observe_tool(payload)
    )
    ctx.register_tool(
        name="scarx_video",
        toolset="scarx_video",
        schema=SCHEMA,
        handler=handler,
        emoji="👁",
    )
