"""Pinned Qwen on same public conversation prefixes; no configuration changes."""
from pathlib import Path
import sys,os,json,socket,time,subprocess,urllib.request,urllib.error
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import psutil
from scarx.neural.language import tokenize
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

cap=apply_memory_limit(4_500_000_000);source=json.loads((root/'artifacts/event-dialogue-latest.json').read_text(encoding='utf-8'))
examples=[json.loads(l) for l in (root/source['data']/'experiences.jsonl').read_text(encoding='utf-8').splitlines()];examples=[r for r in examples if r['split']=='development']
reference=json.loads((root/'artifacts/qwen4b-reference-download.json').read_text(encoding='utf-8'));assert reference['available'] and reference['sha256_verified']
run=new_run(root,'qwen-dialogue-reference-v1',{'protocol':'experiments/qwen-dialogue-reference-v1.md','source_run':source['run'],'max_tokens':64,'context':1024,'threads':4,'temperature':0},data_description='24public curated Antigravity development dialogue prefixes; no private user messages')
job=root/'local/qwen-dialogue-jobs'/run.name;job.mkdir(parents=True)
exe=Path(os.environ['LOCALAPPDATA'])/'hermes/runtimes/llamacpp/b10679/cpu/llama-server.exe'
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
command=[str(exe),'-m',reference['local_path'],'--host','127.0.0.1','--port',str(port),'-c','1024','-t','4','-tb','4','-b','64','-ub','64','-ngl','0','-np','1','--no-mmap','--no-webui']
process=None;outputs=[];error=None;began=time.monotonic()
with ResourceMonitor(limit=4_500_000_000,interval=.1) as monitor,(job/'stdout.log').open('wb') as out,(job/'stderr.log').open('wb') as err:
 try:
    process=subprocess.Popen(command,cwd=job,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if os.name=='nt':psutil.Process(process.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    base='http://127.0.0.1:'+str(port)
    while True:
        try:json.load(urllib.request.urlopen(base+'/health',timeout=1));break
        except (OSError,urllib.error.HTTPError):
            if process.poll() is not None:raise RuntimeError('Reference startup exited')
            if time.monotonic()-began>90:raise TimeoutError('Startup90s')
            time.sleep(.2)
    for row in examples:
        if time.monotonic()-began>180:raise TimeoutError('Registered180s budget')
        payload={'model':'qwen4b-reference','messages':[{'role':m['role'],'content':m['text']} for m in row['messages']],
          'max_tokens':64,'temperature':0,'seed':919,'stream':False}
        request=urllib.request.Request(base+'/v1/chat/completions',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        response=json.load(urllib.request.urlopen(request,timeout=30));choice=response['choices'][0];text=choice['message'].get('content') or ''
        item={'id':row['id'],'messages':row['messages'],'one_reference_target':row['target'],'generated':text,'exact_to_single_target':tokenize(text)==tokenize(row['target']),
          'finish_reason':choice.get('finish_reason'),'usage':response.get('usage')};outputs.append(item)
        with (run/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
        monitor.check()
 except Exception as exc:error=type(exc).__name__+': '+str(exc)[:160]
 finally:
    if process and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
 result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'data':source['data'],'completed':len(outputs),'expected':24,
   'outputs':outputs,'exact_to_single_target':sum(r['exact_to_single_target'] for r in outputs)/len(outputs) if outputs else None,
   'truncated':sum(r['finish_reason']=='length' for r in outputs),'error':error,'server_stopped':process.poll() is not None if process else True,
   'resources':monitor.report(),'cap':cap,'matched_memory_or_pretraining':False,'rubric_review':'pending; source-known exploratory only',
   'hermes_configuration_changed':False,'scarx_uses_qwen_proxy':False,'general_chat_equivalence_proven':False}
 write_json(run/'result.json',result);write_json(root/'artifacts/qwen-dialogue-reference-latest.json',result)
 (run/'report.md').write_text('# Qwen dialogue reference\n\nSame24public context prefixes, no added system prompt/tools/retrieval. Exact matching to one stylistic target is not a sufficient conversational quality metric.\n\n'+json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'run':str(run),'completed':len(outputs),'server_stopped':result['server_stopped'],'error':error}),flush=True)
