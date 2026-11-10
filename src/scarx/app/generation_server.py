"""Local HTTP surface for the single Generation 1 core."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
from urllib.parse import urlsplit, parse_qs
from .generation import Generation, MODEL
from .openai_backend import BackendError, format_completion
from ..runtime.limits import apply_memory_limit


def models():
    return {'object':'list','data':[{'id':MODEL,'object':'model','created':0,'owned_by':'local-scarx',
      'name':'SCAR-X · Thế hệ đầu','description':'One shared ALIF recurrent core, initially48 neurons with bounded adaptive allocation. Experimental dialogue and digits-only vision. No general chat/reasoning/tools claim.',
      'capabilities':{'vision':True,'vision_scope':'handwritten_digits_only','reasoning':False,'tool_calling':False,'shared_core':True}}]}


def chat(app, data):
    if data.get('model')!=MODEL:raise BackendError('Dùng model scarx-generation1. Các lựa chọn v0.1/v0.2 đã ngừng phục vụ.','model_not_found',404)
    if type(data.get('stream',False)) is not bool or data.get('n',1)!=1:raise BackendError('stream boolean, n=1.','invalid_parameter',400)
    if data.get('tool_choice') not in (None,'none','auto'):raise BackendError('Core chưa có policy thao tác công cụ.','tools_not_supported')
    messages=data.get('messages')
    if not isinstance(messages,list) or not 1<=len(messages)<=256 or any(not isinstance(m,dict) for m in messages):raise BackendError('messages cần1–256object.','invalid_messages',400)
    users=[m for m in messages if m.get('role')=='user']
    if not users:raise BackendError('Cần user message.','invalid_messages',400)
    content=users[-1].get('content'); prompt=''; images=[]
    if isinstance(content,str):prompt=content
    elif isinstance(content,list):
        if len(content)>16:raise BackendError('Quá nhiều content parts.','invalid_messages',400)
        for part in content:
            if not isinstance(part,dict):raise BackendError('Content part không hợp lệ.','invalid_messages',400)
            if part.get('type')=='text' and isinstance(part.get('text'),str):prompt+=part['text']+'\n'
            elif part.get('type')=='image_url':
                value=part.get('image_url');images.append(value.get('url') if isinstance(value,dict) else value)
            else:raise BackendError('Chỉ nhận text và image_url.','unsupported_input')
    else:raise BackendError('Content cầntext hoặccontentparts.','invalid_messages',400)
    if images:
        if len(images)!=1:raise BackendError('Chỉ nhận một ảnh chữ số.','unsupported_input')
        result=app.vision({'image':images[0],'prompt':prompt.strip(),'neuron_budget':data.get('scarx_neuron_budget'),'request_id':data.get('scarx_request_id')})
        return format_completion(json.dumps(result,ensure_ascii=False),data,MODEL,{'vision_scope':'handwritten_digits_only','shared_core':True,'language_generation':False})
    maximum=data.get('max_completion_tokens',data.get('max_tokens',24))
    if type(maximum) is not int or maximum<1:raise BackendError('max_tokens phải là số nguyên dương.','invalid_parameter',400)
    result=app.chat(prompt.strip(),min(32,maximum),data.get('scarx_neuron_budget'),data.get('scarx_request_id'),data.get('scarx_session_id','web'))
    if data.get('scarx_question_surface')=='learning_tab' and result.get('learning_question'):
        result={**result,'text':'[SCAR-X] Tôi chưa có câu trả lời đáng tin. Mục cần bạn hướng dẫn đã được chuyển sang tab Hỏi để học.',
          'source':'learning controller routing notice, not neural language generation'}
    return format_completion(result['text'],data,MODEL,{'shared_core':True,'quality_gate_passed':False,
       'checkpoint':result['checkpoint'],'stopped':result['stopped'],
       'workload':result['workload'],
       'learning_question':result.get('learning_question'),'generation_source':result.get('source'),
       'context_scope':'latest user text only; system/history not yet learned','thinking_trace':'local dashboard; measured events, not reasoning_content'})


def make_server(app, port=8765):
    token=secrets.token_urlsafe(32); static=Path(__file__).parent/'generation_web'
    from .capacity_lab import CapacityLab
    capacity_lab=CapacityLab(app)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def allowed(self):return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        def send(self,data,status=200,kind='application/json; charset=utf-8',download=False):
            payload=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            for key,value in [('Content-Type',kind),('Content-Length',str(len(payload))),('Cache-Control','no-store'),('X-Content-Type-Options','nosniff'),('Referrer-Policy','no-referrer'),
               ('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")]:self.send_header(key,value)
            if download:self.send_header('Content-Disposition','attachment; filename="SCAR-X-generation1-session.zip"')
            self.end_headers()
            try:self.wfile.write(payload)
            except (ConnectionError,OSError):pass
        def do_GET(self):
            if not self.allowed():return self.send({'error':'Host không hợp lệ'},403)
            url=urlsplit(self.path)
            try:
                if url.path in ('/','/app.js','/research.js','/research.css','/style.css'):
                    name='index.html' if url.path=='/' else url.path[1:]
                    kind={'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','research.js':'text/javascript; charset=utf-8','research.css':'text/css; charset=utf-8','style.css':'text/css; charset=utf-8'}[name]
                    return self.send((static/name).read_bytes(),kind=kind)
                if url.path=='/api/status':return self.send({**app.status(int(parse_qs(url.query).get('after',['0'])[0])),'token':token})
                if url.path=='/api/checkpoints':return self.send(app.checkpoints())
                if url.path=='/api/research':
                    from ..runtime.experiment_status import dashboard
                    root=Path(__file__).resolve().parents[3]
                    with app.lock:
                        live={'active':app.model.active_count,'allocated':app.model.core.config.hidden,'maximum':app.model.max_neurons,
                          'workload':app.workload,'model':MODEL,'state':app.state}
                    from .observatory import control_answer_archive,vision_answer_archive,visual_word_answer_archive,eligibility_answer_archive,direct_word_answer_archive,direct_visual_answer_archive
                    research=dashboard(root);selected=next((j for j in research['jobs'] if j['status']=='running'),next(iter(research['jobs']),{}))
                    archive=direct_visual_answer_archive(root,selected.get('run_id')) or direct_word_answer_archive(root,selected.get('run_id')) or eligibility_answer_archive(root,selected.get('run_id')) or visual_word_answer_archive(root,selected.get('run_id')) or vision_answer_archive(root,selected.get('run_id')) or control_answer_archive(root,selected.get('run_id'))
                    return self.send({**research,'capacity_demo':capacity_lab.status(),'live_capacity':live,'answer_archive':archive})
                if url.path=='/api/research/report':
                    from ..runtime.experiment_status import report
                    return self.send(report(Path(__file__).resolve().parents[3],parse_qs(url.query).get('run_id',[''])[0]),kind='text/plain; charset=utf-8')
                if url.path=='/api/trace':
                    from .observatory import feed
                    return self.send(feed(app,Path(__file__).resolve().parents[3]))
                if url.path=='/api/export':return self.send(app.export(),kind='application/zip',download=True)
                if url.path=='/api/examples':
                    path=Path(__file__).resolve().parents[3]/'artifacts/pilots/vision-examples.json'
                    return self.send(json.loads(path.read_text(encoding='utf-8')) if path.exists() else [])
                if url.path=='/v1/models':return self.send(models())
                if url.path=='/v1/workload':
                    with app.lock:load=app.workload
                    return self.send({'model':MODEL,'workload':load})
                if url.path=='/v1/questions':return self.send(app.pending_question(parse_qs(url.query).get('session_id',['web'])[0]))
                if url.path=='/v1/learning/feed':
                    from .learning_channel import feed
                    return self.send(feed(app))
                if url.path=='/v1/capabilities':return self.send({'model':MODEL,'cores':1,'vision':True,'vision_scope':'handwritten_digits_only','tool_calling':False,'learned_reasoning':False,'idle_neural_processing':True,'guarded_online_learning':True,'idle_verified_replay':True,'unverified_self_training':False,'proactive_questions':True,'general_chat':False,'ram_target_bytes':2_000_000_000})
                return self.send({'error':'Không tìm thấy'},404)
            except (ValueError,TypeError):return self.send({'error':'Tham số không hợp lệ'},400)
            except Exception as exc:
                print(f'GET: {type(exc).__name__}: {exc}',flush=True);return self.send({'error':'Không đọc được dữ liệu'},500)
        def do_POST(self):
            backend=self.path in ('/v1/chat/completions','/v1/feedback')
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=(1_000_000 if backend or self.path=='/api/vision' else 65536):return self.send({'error':'Request quá lớn hoặc rỗng'},413)
                self.connection.settimeout(3);raw=self.rfile.read(size)
                if len(raw)!=size:return self.send({'error':'Request chưa đầy đủ'},400)
            except (ValueError,TimeoutError,OSError):return self.send({'error':'Không đọc được request'},408)
            origins=(f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')
            if not self.allowed() or self.headers.get('Origin',origins[0]) not in origins or (not backend and not secrets.compare_digest(self.headers.get('X-SCARX-Token',''),token)):
                return self.send({'error':'Request không thuộc phiên localhost'},403)
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send({'error':'Cầnapplication/json'},415)
            try:
                data=json.loads(raw,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
                if not isinstance(data,dict):raise ValueError('CầnJSONobject')
                if self.path=='/v1/feedback':return self.send(app.answer_question(data))
                if self.path=='/v1/learning/message':
                    from .learning_channel import post
                    return self.send(post(app,data))
                if backend:
                    result,stream=chat(app,data)
                    return self.send(stream,kind='text/event-stream; charset=utf-8') if stream is not None else self.send(result)
                actions={'/api/chat':lambda:app.chat(data.get('prompt'),data.get('max_words',24),data.get('neuron_budget'),data.get('request_id')),
                         '/api/vision':lambda:app.vision(data),'/api/teach':lambda:app.teach(data),
                         '/api/idle':lambda:app.configure(data),'/api/checkpoint':app.checkpoint,
                         '/api/learning':lambda:app.configure_learning(data),'/api/feedback':lambda:app.answer_question(data),
                         '/api/capacity-demo':lambda:capacity_lab.start(data.get('origin','release')),
                         '/api/capacity-demo/stop':lambda:(capacity_lab.stop.set() or {'stop_requested':True}),
                         '/api/restore':lambda:app.restore(data.get('checkpoint'))}
                if self.path=='/api/observe-answer':
                    from .observatory import observe_answer
                    return self.send(observe_answer(app,Path(__file__).resolve().parents[3],data))
                if self.path=='/api/project-probe':
                    from .project_supervision import example
                    prompt,target,sha=example(data)
                    return self.send(app.chat(prompt,record_experience=False,observation={'target':target,'case_id':f'project-{data["family"]}-{data.get("example",0)}',
                        'input_scope':'Actual project public prompt only; reference target is not neural input or an automatic label','source_sha256':sha}))
                if self.path not in actions:return self.send({'error':'Không tìm thấy thao tác'},404)
                return self.send(actions[self.path]())
            except BackendError as exc:return self.send(exc.response(),exc.status)
            except (ValueError,TypeError,KeyError) as exc:return self.send({'error':{'message':str(exc),'type':'invalid_request_error','code':'invalid_input'}},400)
            except (MemoryError,TimeoutError) as exc:return self.send({'error':str(exc),'rolled_back':True},507)
            except Exception as exc:
                print(f'POST: {type(exc).__name__}: {exc}',flush=True);return self.send({'error':'Thao tác lỗi; đợt học lỗi giữ state trước đó.'},500)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True;server.workbench=app;server.capacity_lab=capacity_lab
    original_close=server.server_close
    def close_server():capacity_lab.close();original_close()
    server.server_close=close_server
    return server


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='SCAR-X — Thế hệ đầu, một core chung')
    parser.add_argument('--port',type=int,default=8765);parser.add_argument('--data-dir',default='local/generation1')
    parser.add_argument('--no-idle',action='store_true');parser.add_argument('--import-checkpoint',type=Path)
    args=parser.parse_args()
    if not 1<=args.port<=65535:parser.error('Port1–65535')
    limit=apply_memory_limit();app=Generation(args.data_dir);app.limit_status=limit
    try:
        if args.import_checkpoint:
            if args.import_checkpoint.stat().st_size>3_000_000:raise ValueError('Session quá lớn')
            with app.attention():
                before=app._pack()
                try:app._unpack(args.import_checkpoint.read_bytes());app._commit('import')
                except BaseException:app._unpack(before);raise
        if args.no_idle:app.configure({'enabled':False})
        server=make_server(app,args.port)
        print(f'SCAR-X · Thế hệ đầu: http://127.0.0.1:{args.port}\nModel: {MODEL}\nData: {app.directory.resolve()}\nMemory cap: {limit}',flush=True)
        try:server.serve_forever(poll_interval=.3)
        except KeyboardInterrupt:pass
        finally:server.server_close()
    finally:app.close()


if __name__=='__main__':main()
