"""Read-only localhost observer for one registered research run; no model loaded."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse,json,os

PAGE='''<!doctype html><meta charset="utf-8"><title>SCAR-X · nghiên cứu thật</title>
<style>body{background:#111820;color:#e2e8ed;font:16px system-ui;margin:32px}h1{font-size:24px}small{color:#a6bdca}pre{white-space:pre-wrap;background:#18232f;padding:18px;border-radius:10px}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #34434f}#state{color:#71d6bf}code{color:#b0d9ff}</style>
<h1>SCAR-X · Kiểm chứng R8 đang chạy</h1><p id="state">Đang đọc log thật…</p>
<small>Trang chỉ quan sát. Đây là bài sự kiện neural hai nhãn, chưa phải chat/vision tổng quát. Không chạy Qwen; không hiển thị suy nghĩ bằng lời giả.</small>
<h2>Tiến độ và tài nguyên</h2><pre id="progress"></pre><h2>Cập nhật synapse gần nhất</h2><pre id="learning"></pre>
<h2>Đầu ra neural thật so với nhãn oracle</h2><table><thead><tr><th>Task / seed / arm</th><th>Sự kiện đầu vào</th><th>Neural</th><th>Đáp án</th></tr></thead><tbody id="answers"></tbody></table>
<script>function cell(row,t){let c=document.createElement('td');c.textContent=t;row.append(c)}async function refresh(){try{let p=await(await fetch('/progress.json',{cache:'no-store'})).json();document.getElementById('state').textContent=p.status+' · '+p.run;document.getElementById('progress').textContent=JSON.stringify(p,null,2);let e=await(await fetch('/events.json',{cache:'no-store'})).json();document.getElementById('learning').textContent=JSON.stringify(e.learning.slice(-5),null,2);let b=document.getElementById('answers');b.replaceChildren();for(let a of e.answers.slice(-16)){let r=document.createElement('tr');cell(r,a.task+' / '+a.seed+' / '+a.arm+' / '+a.budget);cell(r,JSON.stringify(a.input));cell(r,String(a.prediction));cell(r,String(a.target));b.append(r)}}catch(e){document.getElementById('state').textContent='Chưa đọc được log: '+e.message}}refresh();setInterval(refresh,1500)</script>'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--port',type=int,default=8767);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];run=Path(args.run).resolve()
    if run.parent!=root/'runs' or not (run/'manifest.json').is_file():raise ValueError('Registered owned run required')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=self.path.split('?',1)[0]
            if path=='/':body=PAGE.encode('utf-8');mime='text/html; charset=utf-8'
            elif path in ('/progress.json','/events.json'):
                source=run/path[1:]
                if not source.exists():self.send_error(404);return
                if source.stat().st_size>1_000_000:self.send_error(413);return
                body=source.read_bytes();mime='application/json; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    (run/'observer.json').write_text(json.dumps({'pid':os.getpid(),'port':args.port,'url':f'http://127.0.0.1:{args.port}',
        'run':run.name,'scope':'Read-only public research logs, no model or live SQLite loaded'},indent=2),encoding='utf-8')
    print(f'Observer http://127.0.0.1:{args.port}',flush=True);server.serve_forever()


if __name__=='__main__':main()
