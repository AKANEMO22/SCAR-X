"""Loopback-only web UI, stdlib HTTP, no cloud or frontend build chain."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
from urllib.parse import urlsplit,parse_qs
from ..runtime.limits import apply_memory_limit
from .state import Workbench
from .openai_backend import models,chat,BackendError


def make_server(workbench,port=8765):
    token=secrets.token_urlsafe(32)
    static=Path(__file__).parent/'web'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):
            if args and str(args[1:2])!='(\'200\',)': super().log_message(fmt,*args)

        def _host_ok(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

        def _send(self,data,status=200,kind='application/json; charset=utf-8',download=None):
            payload=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(status)
            for key,value in [('Content-Type',kind),('Content-Length',str(len(payload))),('Cache-Control','no-store'),
                ('X-Content-Type-Options','nosniff'),('Referrer-Policy','no-referrer'),
                ('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")]:
                self.send_header(key,value)
            if download: self.send_header('Content-Disposition',f'attachment; filename="{download}"')
            self.end_headers()
            try: self.wfile.write(payload)
            except ConnectionError: pass

        def do_GET(self):
            if not self._host_ok(): return self._send({'error':'Host không hợp lệ.'},403)
            parsed=urlsplit(self.path); path=parsed.path
            try:
                if path in ('/','/app.js','/style.css','/responsive.css'):
                    name='index.html' if path=='/' else path[1:]
                    kind={'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','style.css':'text/css; charset=utf-8','responsive.css':'text/css; charset=utf-8'}[name]
                    return self._send((static/name).read_bytes(),kind=kind)
                if path=='/api/status': return self._send({**workbench.status(),'token':token})
                if path=='/v1/models': return self._send(models(workbench))
                if path=='/v1/capabilities': return self._send({'model':'scarx-signal-v0.1','vision':False,'tool_calling':False,'learned_reasoning':False,'automatic_compute':True,'input':'structured user JSON'})
                if path=='/api/resources': return self._send(workbench.resources())
                if path=='/api/history': return self._send(workbench.history())
                if path=='/api/checkpoints': return self._send(workbench.checkpoints())
                if path=='/api/lab/status':return self._send(workbench.pilots.status())
                if path=='/api/lab/checkpoints':return self._send(workbench.pilots.checkpoints())
                if path=='/api/lab/examples':return self._send(workbench.pilots.examples())
                if path=='/api/memory': return self._send(workbench.memory(parse_qs(parsed.query).get('q',[''])[0]))
                if path.startswith('/api/export/'):
                    identifier=path.rsplit('/',1)[-1]
                    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier): raise ValueError('Checkpoint ID không hợp lệ.')
                    with workbench.lock: payload=workbench._payload(identifier)
                    return self._send(payload,kind='application/zip',download=f'scarx-{identifier}.zip')
                return self._send({'error':'Không tìm thấy đường dẫn.'},404)
            except (ValueError,KeyError) as exc: self._send({'error':str(exc)},400)
            except Exception as exc:
                print(f'GET failed: {type(exc).__name__}: {exc}',flush=True)
                self._send({'error':'Không đọc được dữ liệu. Xem terminal; trạng thái học chưa bị thay thế.'},500)

        def do_POST(self):
            allowed=(f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')
            backend=self.path=='/v1/chat/completions'
            # Drain a bounded body before an early403. Windows can otherwise
            # reset a socket with unread inbound bytes before the client gets
            # the error response. No JSON/model processing happens before auth.
            try:
                size=int(self.headers.get('Content-Length','0'))
                limit=1_000_000 if backend else (700_000 if self.path=='/api/lab/vision' else 65536)
                if not 0<size<=limit:return self._send({'error':'Request vượt giới hạn kích thước.'},413)
                self.connection.settimeout(3)
                body=self.rfile.read(size)
                if len(body)!=size:return self._send({'error':'Request chưa đầy đủ.'},400)
            except (ValueError,TimeoutError,OSError):return self._send({'error':'Không đọc được request trong giới hạn.'},408)
            if not self._host_ok() or self.headers.get('Origin',allowed[0]) not in allowed or (not backend and not secrets.compare_digest(self.headers.get('X-SCARX-Token',''),token)):
                return self._send({'error':'Yêu cầu không thuộc phiên localhost hiện tại.'},403)
            if self.headers.get('Content-Type','').split(';')[0]!='application/json': return self._send({'error':'Chỉ nhận application/json.'},415)
            try:
                data=json.loads(body,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('JSON không hữu hạn.')))
                if not isinstance(data,dict): raise ValueError('Request phải là JSON object.')
                if backend:
                    completion,stream=chat(workbench,data)
                    return self._send(stream,kind='text/event-stream; charset=utf-8') if stream is not None else self._send(completion)
                actions={'/api/predict':lambda:workbench.predict(data),'/api/teach':lambda:workbench.teach(data),
                         '/api/evaluate':workbench.evaluate,'/api/checkpoint':workbench.checkpoint,
                         '/api/restore':lambda:workbench.restore(data.get('checkpoint',''))}
                actions.update({'/api/lab/generate':lambda:workbench.pilots.language(data.get('prompt')),
                                '/api/lab/teach':lambda:workbench.pilots.teach_language(data),
                                '/api/lab/vision':lambda:workbench.pilots.vision(data),
                                '/api/lab/restore':lambda:workbench.pilots.restore(data.get('checkpoint'))})
                if self.path not in actions: return self._send({'error':'Không tìm thấy thao tác.'},404)
                return self._send(actions[self.path]())
            except BackendError as exc: self._send(exc.response(),exc.status)
            except (ValueError,KeyError,TypeError) as exc:
                self._send({'error':{'message':str(exc),'type':'invalid_request_error','code':'invalid_command','param':None}} if backend else {'error':str(exc)},400)
            except (MemoryError,TimeoutError) as exc: self._send({'error':str(exc),'rolled_back':True},507)
            except Exception as exc:
                print(f'POST failed: {type(exc).__name__}: {exc}',flush=True)
                self._send({'error':'Thao tác không hoàn tất. Đợt học lỗi được rollback; xem terminal để biết chi tiết.'},500)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    server.workbench=workbench
    return server


def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr,'reconfigure'): sys.stderr.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='SCAR-X V0.1 — bàn dạy neural core local')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--data-dir',default='local/mvp')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--import-checkpoint',type=Path,help='Import a SCAR-X full-session bundle as a new active version, preserving earlier checkpoints')
    args=parser.parse_args()
    if not 1<=args.port<=65535: parser.error('Port phải trong khoảng 1–65535.')
    limit=apply_memory_limit()
    workbench=Workbench(args.data_dir,args.seed); workbench.limit_status=limit
    if args.import_checkpoint:
        if args.import_checkpoint.stat().st_size>2_000_000: raise ValueError('Checkpoint lớn hơn 2 MB.')
        workbench.import_checkpoint(args.import_checkpoint.read_bytes())
    server=make_server(workbench,args.port)
    print(f'SCAR-X MVP V0.1: http://127.0.0.1:{server.server_port}\nData: {workbench.directory.resolve()}\nMemory limit: {limit}\nCtrl+C để dừng; mỗi đợt học thành công đã tự lưu.',flush=True)
    try: server.serve_forever(poll_interval=.3)
    except KeyboardInterrupt: pass
    finally:
        server.server_close(); workbench.close()


if __name__=='__main__': main()
