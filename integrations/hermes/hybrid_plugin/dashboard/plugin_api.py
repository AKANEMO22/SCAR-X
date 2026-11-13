"""Hermes-authenticated, fixed-loopback adapter; no new service or framework."""

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler
from fastapi import APIRouter, HTTPException, Request as WebRequest

router = APIRouter()
_lock = threading.Lock()
_owned = None


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def settings():
    value = json.loads(
        Path(__file__).with_name("settings.json").read_text(encoding="utf-8")
    )
    if value.get("gateway") != "http://127.0.0.1:8775":
        raise HTTPException(409, "Unexpected SCAR-X endpoint")
    return value


def ensure_gateway():
    global _owned
    config = settings()
    try:
        with build_opener(NoRedirect()).open(
            config["gateway"] + "/api/status", timeout=2
        ) as response:
            value = json.loads(response.read(2_000_000))
        if value.get("model") != "scarx-hybrid":
            raise HTTPException(409, "Port8775 is used by a different service")
        # The SCAR-X monitor discovers/counts both Electron and a detached
        # Hermes Python backend; no second client-registration service.
        return
    except HTTPException:
        raise
    except (URLError, TimeoutError, OSError):
        pass
    with _lock:
        if _owned is not None and _owned.poll() is None:
            return
        # Never terminate/take over a foreign listener.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8775)) == 0:
                raise HTTPException(409, "Port8775 is occupied")
        script = Path(config["entrypoint"]).resolve()
        python = Path(config["python"]).resolve()
        directory = Path(config["data_dir"]).resolve()
        if not script.is_file() or not python.is_file():
            raise HTTPException(
                503, "SCAR-X installation is missing; run its installer"
            )
        log = (directory / "hermes-launch.log").open("ab")
        try:
            _owned = subprocess.Popen(
                [str(python), str(script), "--data-dir", str(directory)],
                cwd=script.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            log.close()


MAPPING = {
    "presence": "/api/presence",
    "observe": "/api/observe",
    "public-topic": "/api/public-topic",
    "goal": "/api/goal",
    "feedback": "/api/feedback",
    "save": "/api/save",
    "rollback": "/api/rollback",
    "learning": "/api/learning",
    "cognition": "/api/cognition",
    "chat": "/v1/chat/completions",
    "screen-start": "/api/screen/start",
    "screen-stop": "/api/screen/stop",
}


def forward(path, body=None):
    ensure_gateway()
    base = settings()["gateway"]
    headers = {"Content-Type": "application/json"}
    if body is not None:
        with build_opener(NoRedirect()).open(
            Request(base + "/api/status", headers={"X-SCARX-Backend": "hermes"}),
            timeout=3,
        ) as response:
            status = json.loads(response.read(2_000_000))
        headers["X-SCARX-Token"] = status["token"]
    encoded = (
        None
        if body is None
        else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
    )
    if encoded and len(encoded) > 6_000_000:
        raise HTTPException(413, "SCAR-X request exceeds bound")
    try:
        with build_opener(NoRedirect()).open(
            Request(base + path, data=encoded, headers=headers),
            timeout=120 if path.endswith("completions") else 15,
        ) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise HTTPException(502, "SCAR-X response exceeds bound")
        value = json.loads(raw)
        if isinstance(value, dict):
            value.pop("token", None)
        return value
    except HTTPError as exc:
        try:
            value = json.loads(exc.read(8000))
            error = value.get("error")
            message = error.get("message") if isinstance(error, dict) else str(error)
        except Exception:
            message = "SCAR-X rejected the request"
        raise HTTPException(
            exc.code if exc.code in (400, 403, 409, 413, 415, 422, 503, 507) else 502,
            message,
        )
    except (URLError, TimeoutError, OSError):
        raise HTTPException(
            503, "SCAR-X đang khởi động hoặc mất kết nối. Giao diện sẽ tự thử lại."
        )


@router.get("/status")
async def status(request: WebRequest):
    try:
        after = max(0, int(request.query_params.get("after", "0")))
    except ValueError:
        raise HTTPException(400, "Invalid cursor")
    return await asyncio.to_thread(forward, "/api/status?after=" + str(after))


@router.get("/screen-sources")
async def screen_sources():
    return await asyncio.to_thread(forward, "/api/screen/sources")


@router.post("/{action}")
async def action(action: str, request: WebRequest):
    if action not in MAPPING:
        raise HTTPException(404, "Unknown SCAR-X action")
    raw = await request.body()
    if len(raw) > 6_000_000:
        raise HTTPException(413, "Request too large")
    try:
        body = json.loads(raw)
    except ValueError:
        raise HTTPException(400, "Expected JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    return await asyncio.to_thread(forward, MAPPING[action], body)
