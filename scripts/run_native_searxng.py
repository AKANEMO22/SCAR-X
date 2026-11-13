"""Actual pinned SearXNG WSGI app in one native Windows process, loopback only."""

import json, os, sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "local/searxng"
installation = json.loads((base / "installation.json").read_text(encoding="utf-8"))
source = Path(installation["source"])
sys.path.insert(0, str(source))
os.environ["SEARXNG_SETTINGS_PATH"] = str(base / "settings.yml")
os.environ["SEARXNG_DEBUG"] = "false"
os.environ["SEARXNG_BIND_ADDRESS"] = "127.0.0.1"
os.environ["SEARXNG_PORT"] = "18778"
os.environ["SEARXNG_DATA_PATH"] = str(base / "data")
(base / "data").mkdir(exist_ok=True)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    from searx.webapp import app
    from waitress import serve

    def local_headers(environ, start_response):
        # Bound only to loopback; tell upstream bot detection the actual peer.
        if environ.get("REMOTE_ADDR") in ("127.0.0.1", "::1"):
            environ.setdefault("HTTP_X_REAL_IP", environ["REMOTE_ADDR"])
        return app(environ, start_response)

    serve(
        local_headers,
        host="127.0.0.1",
        port=18778,
        threads=4,
        connection_limit=24,
        channel_timeout=20,
        max_request_body_size=65536,
    )
