"""One bounded CPU reference run; never modifies Hermes/SCAR-X configuration."""
from pathlib import Path
import os,sys,json,socket,subprocess,time,urllib.request,urllib.error
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import psutil
from scarx.evaluation.knowledge_capacity import dataset,probe,fraction
from scarx.neural.language import tokenize
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor

if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
cap=apply_memory_limit(4_500_000_000)
run=new_run(root,'qwen4b-numeric-reference',{'protocol':'experiments/knowledge-load-retention-v1-amendment.md','reference':'Qwen3-4B-Instruct-2507-Q4_K_M','max_seconds':360,'max_new_tokens':12,'threads':4,'context':1024,'temperature':0},data_description='Same unsigned arithmetic cases. Additional output-format instruction applied identically in this secondary cross-model comparison. No training/selection uses these prompts.')
receipt=json.loads((root/'artifacts/qwen4b-reference-download.json').read_text(encoding='utf-8'));assert receipt['available'] and receipt['sha256_verified']
rows=dataset(nonnegative_only=True)[:160]
for row in rows:row['paraphrase']+='\nChỉ trả lời đáp án số, không giải thích.'
write_json(run/'comparison-data.json',rows)
executable=Path(os.environ['LOCALAPPDATA'])/'hermes/runtimes/llamacpp/b10679/cpu/llama-server.exe'
assert executable.is_file()
job=root/'local/qwen4b-reference-jobs'/run.name;job.mkdir(parents=True)
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
command=[str(executable),'-m',receipt['local_path'],'--host','127.0.0.1','--port',str(port),'-c','1024','-t','4','-tb','4','-b','64','-ub','64','-ngl','0','-np','1','--no-mmap','--no-webui']
outputs=[];error=None;began=time.monotonic();process=None
with ResourceMonitor(limit=4_500_000_000,interval=.1) as monitor, (job/'stdout.log').open('wb') as out,(job/'stderr.log').open('wb') as err:
 try:
    process=subprocess.Popen(command,cwd=job,stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if os.name=='nt':psutil.Process(process.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    endpoint='http://127.0.0.1:'+str(port)
    while True:
        try:
            json.load(urllib.request.urlopen(endpoint+'/health',timeout=1));break
        except (OSError,urllib.error.HTTPError):
            if process.poll() is not None:raise RuntimeError('Reference server exited during startup; inspect local logs')
            if time.monotonic()-began>90:raise TimeoutError('Reference startup90s')
            time.sleep(.2)
    for index,row in enumerate(rows):
        if time.monotonic()-began>360:raise TimeoutError('Registered reference360s budget')
        payload={'model':'qwen4b-reference','messages':[{'role':'user','content':row['paraphrase']}],
          'max_tokens':12,'temperature':0,'seed':919,'stream':False}
        request=urllib.request.Request(endpoint+'/v1/chat/completions',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        t0=time.monotonic();response=json.load(urllib.request.urlopen(request,timeout=30));choice=response['choices'][0];text=choice['message'].get('content') or ''
        outputs.append({'id':row['id'],'split':row['split'],'target':row['target'],'generated':text,'correct':tokenize(text)==tokenize(row['target']),
          'finish_reason':choice.get('finish_reason'),'seconds':time.monotonic()-t0,'usage':response.get('usage')})
        with (run/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(outputs[-1],ensure_ascii=False)+'\n')
        if (index+1)%32==0:print(json.dumps({'completed':index+1,'exact':fraction(outputs),'elapsed':time.monotonic()-began}),flush=True)
        monitor.check()
 except Exception as exc:
    error=type(exc).__name__+': '+str(exc)[:160]
 finally:
    if process and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
 result={'run':run.relative_to(root).as_posix(),'reference_download':{k:v for k,v in receipt.items() if k!='local_path'},'outputs':outputs,
   'exact':fraction(outputs),'curriculum_exact':fraction([x for x in outputs if x['split']=='curriculum']),
   'unseen_exact':fraction([x for x in outputs if x['split']=='unseen']),'completed':len(outputs),'expected':160,
   'truncated':sum(x['finish_reason']=='length' for x in outputs),'error':error,'server_stopped':bool(process and process.poll() is not None),
   'resources':monitor.report(),'hard_cap':cap,'matched_scarx_memory_budget':False,'elapsed_seconds':time.monotonic()-began,
   'general_knowledge_equivalence':False,'reference_online_learning_tested':False,'hermes_config_changed':False}
 write_json(run/'result.json',result);write_json(root/'artifacts/qwen4b-comparison-latest.json',result)
 (run/'report.md').write_text('# Qwen4B numeric reference\n\nFrozen quantized model on unsigned arithmetic only. Same raw question/format instruction is available for secondary SCAR-X probes; no claim about general intelligence or Qwen online learning.\n\n'+json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2)+'\n',encoding='utf-8')
 print('RESULT '+str(run),flush=True)
