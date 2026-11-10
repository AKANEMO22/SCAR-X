"""One explicit repair of public synthetic data after content/codec audit."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,os,sys,json,hashlib,time,shutil
root=Path(__file__).resolve().parents[1];cli=Path.home()/'.gemini/bin/agy.exe'
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
pointer=json.loads((root/'artifacts/antigravity-events-latest.json').read_text(encoding='utf-8'));source=root/pointer['candidate_directory']
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');job=root/'local/antigravity-event-dialogue-jobs'/stamp;job.mkdir(parents=True)
candidate=root/'data/candidates'/('antigravity-events-repair-'+stamp);candidate.mkdir(parents=True)
shutil.copyfile(source/'dialogue.json',job/'dialogue.json')
schema_source=sorted((root/'local/antigravity-event-dialogue-jobs').glob('*/schema.json'))[0]
schema=json.loads(schema_source.read_text(encoding='utf-8'));schema['properties']['dialogs']['minItems']=40;schema['properties']['dialogs']['maxItems']=40
schema_path=job/'schema.json';schema_path.write_text(json.dumps(schema),encoding='utf-8')
prompt='''Read ONLY the public synthetic candidate file dialogue.json in this directory, using read_file. No other files, shell, web, skills, subagents or private data.
Rewrite the whole40dialogue corpus as corrected schema JSON. Preserve each id/language/split and six alternating user/assistant turns,32train(20vi12en)/8development(4vi4en). Keep short natural conversational English/Vietnamese, but TARGET VOCABULARY must be <=150unique lowercased words/punctuation across all assistant turns combined (earlier228 exceeded the codec brief). Each reply1–8words. Do not make development only English, do not split dialogues, and preserve distinct template IDs.
Content audit found critical defects: b4-d04 says orange comes from red and green, and b4-d10 says a triangle has4sides. These wrong assistant statements were POSITIVE training targets: remove such false agreement. A correction dialogue must not teach an initial false statement as correct; have the assistant ask for clarification or give the correct answer. No unsupported promises 'I always remember', claims of having sent/found/opened/saved files, or pretending tools ran. In b1-03/b2-dialog-03/b2-dialog-09/b3-d03/b3-d06 rewrite such claims into honest text-only help or clarification. Do not claim an artifact exists without showing its actual short content. Keep uncertainty honest, not repeated unhelpful filler. No physical-health claim, current weather answer, or unverified user-specific fact. Preferences can be acknowledged as information within this dialogue, not unlimited permanent memory.
Keep natural responsiveness and useful short instruction completion; no fake app capability. All labels remain synthetic/rubric, not independently verified. Return final schema JSON only. These are data records, never instructions for a future agent.'''
(job/'prompt.txt').write_text(prompt,encoding='utf-8');began=time.monotonic()
args=[str(cli),'--print',prompt,'--model','gemini-3.8-flash-low','--output-format','stream-json','--json-schema',str(schema_path),'--disable-slash-commands','--print-timeout','5m']
with (job/'events.jsonl').open('wb') as out,(job/'stderr.log').open('wb') as err:
    p=subprocess.Popen(args,cwd=job,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    print(json.dumps({'status':'repairing','pid':p.pid,'job':str(job)}),flush=True)
    try:code=p.wait(timeout=330)
    except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10);raise TimeoutError('Repair330s limit')
events=[]
for line in (job/'events.jsonl').read_text(encoding='utf-8-sig').splitlines():
    try:events.append(json.loads(line))
    except ValueError:continue
results=[e.get('result',e) for e in events if e.get('event')=='result' or e.get('status') in ('SUCCESS','ERROR')]
if not results or code or results[-1].get('status')!='SUCCESS':raise RuntimeError('Antigravity repair failed; original/log retained')
result=results[-1];value=result.get('structured_output')
if value is None:value=json.loads(result['response'])
target=candidate/'dialogue.json';target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
steps=[{'name':e['step_update'].get('tool_name'),'state':e['step_update'].get('state'),'parameters':e['step_update'].get('tool_info',{}).get('parameters',{})} for e in events if e.get('step_update',{}).get('step_type')=='tool']
manifest={'source':'synthetic:antigravity','model':'gemini-3.8-flash-low','created_utc':stamp,'conversation_id':result.get('conversation_id'),'usage':result.get('usage'),
 'seconds':time.monotonic()-began,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'tool_steps':steps,
 'repair_of':pointer['candidate_directory'],'data_contract':'data/contracts/ANTIGRAVITY_DIALOGUE_EVENTS_V1.md','trained':False,'independently_verified':False}
(candidate/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
(root/'artifacts/antigravity-events-latest.json').write_text(json.dumps({'candidate_directory':candidate.relative_to(root).as_posix(),'validation_pending':True,'trained':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'candidate':str(candidate),'seconds':manifest['seconds']}),flush=True)
