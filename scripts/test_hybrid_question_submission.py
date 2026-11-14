"""Real loopback feedback transport fixture, never live data/model/learning."""

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.store import Store


def main(source):
    source = source.resolve()
    if ROOT not in source.parents:
        raise ValueError("Test source must be in owned checkout")
    nonce = secrets.token_urlsafe(24)
    attempts, last_feedback = [], {}
    with tempfile.TemporaryDirectory(prefix="scarx-question-http-") as temporary:
        store = Store(temporary)
        monitor = SimpleNamespace(check=lambda: {"rss_bytes": 0},
                                  sample=lambda: {"rss_bytes": 0})
        engine = HybridEngine(store, SimpleNamespace(), monitor)
        # Deliberate UI state fixture, not a generated model question or label.
        engine.state["goals"]["fixture"] = {"id": "fixture", "status": "awaiting_human"}
        engine.state["question"] = {
            "id": "ui-transport-fixture", "goal_id": "fixture", "decision_id": None,
            "question": "Owned UI transport fixture; not neural training.",
            "status": "pending",
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send_json(self, body, code=200):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def allowed(self):
                return self.headers.get("X-UI-Fixture") == nonce

            def do_GET(self):
                if not self.allowed() or self.path != "/status":
                    return self.send_json({"error": "Wrong fixture scope"}, 403)
                self.send_json({
                    "question": engine.state["question"], "attempts": len(attempts),
                    "last_feedback": last_feedback, "updates": engine.model.updates,
                    "hidden_updates": engine.model.hidden_updates})

            def do_POST(self):
                if not self.allowed() or self.path != "/feedback":
                    return self.send_json({"error": "Wrong fixture scope"}, 403)
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                attempts.append(payload)
                if len(attempts) == 1:
                    return self.send_json({"error": "Controlled UI transport failure"}, 503)
                result = engine.feedback(**{
                    "identifier": payload["id"], "text": payload["text"],
                    **({"helpful": payload["helpful"]} if "helpful" in payload else {})})
                last_feedback.update(payload)
                self.send_json(result)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            node = Path(os.environ["LOCALAPPDATA"]) / "hermes/tools/node-26.12.1-win32-x64/node.exe"
            result = subprocess.run([str(node), "--test", "tests/hybrid_question_http.test.cjs"],
                cwd=ROOT, env=dict(os.environ,
                    SCARX_QUESTION_TEST_BASE=f"http://127.0.0.1:{server.server_port}",
                    SCARX_QUESTION_TEST_NONCE=nonce, SCARX_QUESTION_TEST_SOURCE=str(source)),
                timeout=30)
            return result.returncode
        finally:
            server.shutdown()
            server.server_close()
            worker.join(3)
            store.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=ROOT / "integrations/hermes/hybrid_plugin/plugin.js")
    raise SystemExit(main(p.parse_args().source))
