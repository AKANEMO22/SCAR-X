"""Same native inputs on actual CPU/GPU profiles, one owned model at a time."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import base64,hashlib,json,subprocess,sys,threading,time,uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import parse_qs,urlparse
from urllib.request import Request,urlopen
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from scarx.hybrid.config import HybridConfig,load_runtime_config
from scarx.hybrid.store import Store,encode
from scarx.hybrid.resources import ResourceMonitor,windows_tree_commit_limit
from scarx.hybrid.backends import NativeBackends
from scarx.hybrid.qualification import native_parity,verify_native_storage
from scarx.hybrid.search_service import SearchService
from hybrid_context_benchmark import consume
import hybrid_service

PAGE='''<!doctype html><meta charset=utf-8><title>SCAR-X CPU / GPU</title><style>body{background:#111820;color:#e0edff;font:15px system-ui;margin:32px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-bottom:1px solid #456;padding:12px}b{color:#8ed}</style><h1>SCAR-X · Đo CPU / GPU thực</h1><p>Trọng số/precision/context giống nhau · không huấn luyện · chat tự khôi phục sau đo</p><div id=s></div><section id=e></section><script>let c=0;async function p(){try{let s=await(await fetch('/api/status?after='+c)).json();document.getElementById('s').textContent=s.note;for(let e of s.events){c=e.id;let a=document.createElement('article'),h=document.createElement('b'),v=document.createElement('pre');h.textContent=e.kind;v.textContent=JSON.stringify(e.payload,null,2);a.append(h,v);document.getElementById('e').prepend(a)}while(document.getElementById('e').children.length>60)document.getElementById('e').lastChild.remove()}catch(e){}setTimeout(p,1500)}p()</script>'''


def main():
    rid=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-cpu-gpu-speed-v1-'+uuid.uuid4().hex[:6]
    run=ROOT/'runs'/rid;run.mkdir()
    data=ROOT/'local/hybrid-v1'
    image=data/'samples/scene-first.png'
    log='\n'.join(f'Record {i}: parcel {i%23} is on shelf {i%7}.' for i in range(80))
    cases=[{'id':'vi-generation','messages':[{'role':'user','content':'Viết bằng tiếng Việt 8 lời khuyên ngắn để kiểm tra một phần mềm trước khi phát hành. Đánh số từng dòng, giải thích mỗi lời khuyên bằng một câu.'}]},
           {'id':'long-log','messages':[{'role':'user','content':'Read this log and briefly explain the last three records.\n'+log}]},
           {'id':'vision','messages':[{'role':'user','content':[{'type':'text','text':'Describe the visible shapes and their colors in English in three sentences. Do not invent text.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()}}]}]}]
    (run/'inputs.json').write_text(encode(cases),encoding='utf8')
    sources=[ROOT/'experiments/hybrid-cpu-gpu-speed-v1.md',Path(__file__),*list((ROOT/'src/scarx/hybrid').glob('*.py'))]
    manifest={'protocol':'experiments/hybrid-cpu-gpu-speed-v1.md','inputs_sha256':hashlib.sha256((run/'inputs.json').read_bytes()).hexdigest(),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'repeats':3,'max_tokens':128,'cache_prompt':False,'CPU_cap':4_500_000_000,'GPU_cap':None,'precision':'unchanged native BF16/F32 + BF16 KV','locked':time.time()}
    (run/'manifest.json').write_text(encode(manifest),encoding='utf8')
    report={'run':rid,'arms':[],'failures':[],'scope':'Deployment profile speed DEV, not a skill/retention benchmark'}
    server=store=monitor=search=backend=None
    state={'note':'Starting benchmark','phase':'CPU/GPU benchmark','ready':False,'model':'scarx-hybrid','pid':os.getpid()}
    try:
        hybrid_service.stop()
        store=Store(data);monitor=ResourceMonitor(4_500_000_000,include_hermes=True)
        lock=threading.RLock()
        def sample():
            with lock:return monitor.sample()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def send(self,data,status=200,mime='application/json'):
                body=data.encode() if isinstance(data,str) else encode(data).encode()
                self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def do_GET(self):
                if self.path=='/':return self.send(PAGE,mime='text/html; charset=utf-8')
                if self.path.startswith('/api/status'):
                    after=int(parse_qs(urlparse(self.path).query).get('after',['0'])[0]);return self.send(state|{'resources':sample(),'events':store.events(after)})
                return self.send({'error':'benchmark maintenance'},503)
            def do_POST(self):return self.send({'error':'CPU/GPU benchmark running; chat resumes automatically'},503)
        server=ThreadingHTTPServer(('127.0.0.1',8775),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
        job=windows_tree_commit_limit(None);report['job']=job
        def event(kind,payload):
            store.event('cpu_gpu_benchmark.'+kind,{'run':rid,**payload},'orchestrator');print(encode({'kind':kind,**payload}),flush=True)
        event('started',manifest)
        integrity=verify_native_storage(data);report['integrity']=integrity
        search=SearchService(store,monitor,True);search.start()
        deadline=time.monotonic()+600
        for mode in ('cpu','gpu'):
            config=HybridConfig(execution_mode=mode,searxng_enabled=True)
            monitor.limit=config.effective_ram_limit;monitor.exceeded=False;monitor.first_exceed=None
            monitor.peak_rss=monitor.peak_commit=monitor.peak_ram=monitor.peak_dedicated_gpu=0
            backend=NativeBackends(data,store,monitor,config)
            stop=threading.Event()
            def watch():
                while not stop.wait(.05):
                    r=sample()
                    if r['exceeded'] and backend and backend.vlm and backend.vlm.poll() is None:backend.vlm.terminate()
            watcher=threading.Thread(target=watch,daemon=True);watcher.start()
            arm={'mode':mode,'records':[]};report['arms'].append(arm)
            try:
                state['note']='Loading '+mode;start=time.monotonic();backend.start();arm['load_seconds']=time.monotonic()-start
                arm['parity']=native_parity(backend,data)
                event('mode_ready',{'mode':mode,'load_seconds':arm['load_seconds'],'native_parity':arm['parity']['status']})
                for case in cases:
                    for repeat in range(3):
                        if time.monotonic()>deadline:raise TimeoutError('Registered600s overall deadline')
                        state['note']=f"{mode} · {case['id']} · repeat{repeat+1}"
                        event('case_started',{'mode':mode,'case':case['id'],'repeat':repeat+1})
                        body={'model':'scarx-hybrid','messages':case['messages'],'max_tokens':128,'seed':260113,'temperature':0.0,'cache_prompt':False,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
                        start=time.monotonic();timer=threading.Timer(90,lambda:backend.vlm.terminate() if backend.vlm and backend.vlm.poll() is None else None);timer.start()
                        try:
                            with urlopen(Request(backend.vlm_url+'/v1/chat/completions',data=encode(body).encode(),headers=backend.headers()),timeout=90) as response:
                                header_seconds=time.monotonic()-start;result=consume(response)
                        finally:timer.cancel();timer.join()
                        (run/f'{mode}-{case["id"]}-{repeat}.raw.json').write_text(encode(result),encoding='utf8')
                        row={k:v for k,v in result.items() if k not in ('chunks',)}
                        row.update(case=case['id'],repeat=repeat+1,seconds=time.monotonic()-start,ttft_seconds=header_seconds+(result['first_after_headers_seconds'] or 0),characters=len(result['answer']),resources=sample())
                        arm['records'].append(row);monitor.check()
                        event('answer',{'mode':mode,**row})
                arm['status']='COMPLETED'
            except Exception as exc:
                arm.update(status='FAILED',error=type(exc).__name__+': '+str(exc))
            finally:
                stop.set();watcher.join(2);arm['resources']=sample();backend.close();backend=None
                (run/f'{mode}.json').write_text(encode(arm),encoding='utf8')
                event('mode_finished',{'mode':mode,'status':arm['status'],'resources':arm['resources']})
        for arm in report['arms']:
            arm['by_case']={}
            for case in cases:
                rows=[r for r in arm['records'] if r['case']==case['id']]
                if rows:arm['by_case'][case['id']]={'median_seconds':statistics.median(r['seconds'] for r in rows),'median_TTFT':statistics.median(r['ttft_seconds'] for r in rows),'median_decode_tps':statistics.median(r['timings'].get('predicted_per_second',0) for r in rows),'median_prefill_tps':statistics.median(r['timings'].get('prompt_per_second',0) for r in rows),'repeats':len(rows)}
        if len(report['arms'])==2:
            cpu,gpu=report['arms'];report['speedups']={k:cpu['by_case'][k]['median_seconds']/gpu['by_case'][k]['median_seconds'] for k in cpu['by_case'] if k in gpu['by_case']}
        event('finished',{'speedups':report.get('speedups'),'status':[a['status'] for a in report['arms']]})
    except Exception as exc:report['failures'].append(type(exc).__name__+': '+str(exc))
    finally:
        if backend:backend.close()
        if search:search.close()
        if server:server.shutdown();server.server_close()
        if monitor:monitor.close()
        if store:store.close()
        # The user's explicit final preference is GPU; preserve the same data.
        (data/'config.json').write_text(json.dumps({'execution_mode':'gpu','gpu_device':0,'searxng_enabled':True},indent=2),encoding='utf8')
        try:report['restored']=hybrid_service.start()
        except Exception as exc:report['failures'].append('restore: '+str(exc))
        (run/'receipt.json').write_text(encode(report),encoding='utf8');print(encode({'run':rid,'speedups':report.get('speedups'),'failures':report['failures'],'restored':report.get('restored')}),flush=True)


if __name__=='__main__':main()
