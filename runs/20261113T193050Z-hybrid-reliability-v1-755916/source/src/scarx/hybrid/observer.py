"""Passive observer usable during setup, before model inference is available."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlparse
import psutil
from .store import Store, encode

PAGE = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>SCAR-X Hybrid — hoạt động thực</title>
<style>body{background:#10151d;color:#dbe6f4;font:15px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}h1{font-size:25px}small{color:#a1b2c9}#status{padding:16px;background:#192533;border-radius:10px}article{padding:12px;border-bottom:1px solid #2a3647}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.actor{color:#69d6ae}a{color:#82c5ff}</style>
<h1>SCAR-X Hybrid · Trace trực tiếp</h1><p>Hoạt động kỹ thuật, đầu ra model và học neural được ghi riêng. Chỉ hiển thị sự kiện thực.</p>
<div id="status">Đang kết nối…</div><section id="events"></section><script>
let cursor=0;async function poll(){try{let r=await fetch('/api/status?after='+cursor);let s=await r.json();document.getElementById('status').textContent=s.phase+' · observer '+(s.observer_rss_bytes/1e6).toFixed(1)+' MB · trần toàn hệ 3000 MB · '+s.note;for(let e of s.events){cursor=e.id;let a=document.createElement('article');let h=document.createElement('div');h.className='actor';h.textContent=new Date(e.created*1000).toLocaleTimeString()+' · '+e.actor+' · '+e.kind;let p=document.createElement('pre');p.textContent=JSON.stringify(e.payload,null,2);a.append(h,p);document.getElementById('events').prepend(a)}while(document.getElementById('events').children.length>180)document.getElementById('events').lastChild.remove()}catch(e){document.getElementById('status').textContent=e.message}setTimeout(poll,1500)}poll();</script></html>"""


def serve(directory, port=8775):
    store = Store(directory)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = PAGE.encode()
                mime = "text/html; charset=utf-8"
            elif parsed.path == "/api/status":
                try:
                    after = max(0, int(parse_qs(parsed.query).get("after", ["0"])[0]))
                except ValueError:
                    self.send_error(400)
                    return
                phase = store.get("runtime", "status") or {
                    "phase": "setup",
                    "note": "Model chưa được nạp; không có phép đo chất lượng.",
                }
                body = encode(
                    dict(
                        phase,
                        events=store.events(after),
                        observer_rss_bytes=psutil.Process().memory_info().rss,
                    )
                ).encode()
                mime = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Observer http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--port", type=int, default=8775)
    args = p.parse_args()
    serve(args.data_dir, args.port)
