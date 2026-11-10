"""One bounded public-brief CLI generation; no private transcripts or weights."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys,os,time,json,hashlib
root=Path(__file__).resolve().parents[1];cli=Path.home()/'.gemini/bin/agy.exe'
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
job=root/'local/antigravity-event-dialogue-jobs'/stamp;job.mkdir(parents=True)
output=root/'data/candidates'/('antigravity-events-'+stamp);output.mkdir(parents=True)
brief=(root/'data/contracts/ANTIGRAVITY_DIALOGUE_EVENTS_V1.md').read_text(encoding='utf-8')
turn={'type':'object','properties':{'role':{'type':'string','enum':['user','assistant']},'text':{'type':'string'}},'required':['role','text']}
dialog={'type':'object','properties':{'id':{'type':'string'},'language':{'type':'string','enum':['vi','en']},'split':{'type':'string','enum':['train','development']},'scenario_id':{'type':'string'},'template_id':{'type':'string'},'turns':{'type':'array','minItems':6,'maxItems':6,'items':turn}},'required':['id','language','split','scenario_id','template_id','turns']}
schema={'type':'object','properties':{'dialogs':{'type':'array','minItems':40,'maxItems':40,'items':dialog},'notes':{'type':'string'}},'required':['dialogs','notes']}
schema_path=job/'schema.json';schema_path.write_text(json.dumps(schema),encoding='utf-8')
prompt=brief+'\nReturn schema JSON only. Do not inspect workspace files or use any tool. Split all40 exactly as specified. Use input text that unambiguously supports any memory-related answer.'
(job/'prompt.txt').write_text(prompt,encoding='utf-8')
args=[str(cli),'--print',prompt,'--model','gemini-3.8-flash-high','--output-format','stream-json','--json-schema',str(schema_path),'--disable-slash-commands','--print-timeout','8m']
started=time.monotonic()
with (job/'events.jsonl').open('wb') as out,(job/'stderr.log').open('wb') as err:
    proc=subprocess.Popen(args,cwd=job,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    print(json.dumps({'pid':proc.pid,'status':'generating','job':str(job)}),flush=True)
    try:code=proc.wait(timeout=510)
    except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10);raise TimeoutError('Registered510s CLI limit')
events=[]
for line in (job/'events.jsonl').read_text(encoding='utf-8-sig').splitlines():
    try:events.append(json.loads(line))
    except ValueError:continue
results=[e.get('result',e) for e in events if e.get('event')=='result' or e.get('status') in ('SUCCESS','ERROR')]
if not results:raise RuntimeError('No structured result; raw job retained')
result=results[-1]
if code or result.get('status')!='SUCCESS':raise RuntimeError('CLI generation failed; raw job retained')
value=result.get('structured_output')
if value is None:value=json.loads(result['response'])
target=output/'dialogue.json';target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
tool_steps=[{'name':e['step_update'].get('tool_name'),'state':e['step_update'].get('state')} for e in events if e.get('step_update',{}).get('step_type')=='tool']
manifest={'created_utc':stamp,'source':'synthetic:antigravity','model':'gemini-3.8-flash-high','conversation_id':result.get('conversation_id'),
 'usage':result.get('usage'),'seconds':time.monotonic()-started,'tool_steps':tool_steps,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
 'data_contract':'data/contracts/ANTIGRAVITY_DIALOGUE_EVENTS_V1.md','independently_verified':False,'trained':False}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
(root/'artifacts/antigravity-events-latest.json').write_text(json.dumps({'candidate_directory':output.relative_to(root).as_posix(),'validation_pending':True,'trained':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'candidate':str(output),'seconds':manifest['seconds']}),flush=True)
