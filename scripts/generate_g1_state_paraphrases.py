"""Authorized Antigravity CLI public data generation; no model training/import."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os,time,sys,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress

INTENTS=('greeting_vi','greeting_en','identity','gratitude','farewell','activity','capabilities','teaching')


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cli=Path.home()/'.gemini/bin/agy.exe'
    if not cli.is_file():raise FileNotFoundError('Installed Antigravity CLI missing')
    run=new_run(root,'antigravity-g1-state-paraphrases',{'display_title':'Antigravity · paraphrases hội thoại công khai để kiểm state',
        'observability_kind':'data_generation','model_scope':'Synthetic public prompts only; no learned weights or factual certification',
        'teacher':'gemini-3.8-flash-low','items':48,'new_training':False},
        data_description='Public project intent descriptions only; no user chat/history/files transmitted',data_files=['data/contracts/ANTIGRAVITY_G1_STATE_PARAPHRASES.md'])
    job=root/'local/antigravity-g1-state'/run.name;job.mkdir(parents=True)
    candidate=root/'data/candidates'/run.name;candidate.mkdir(parents=True)
    schema={'type':'object','properties':{'items':{'type':'array','minItems':48,'maxItems':48,'items':{'type':'object',
        'properties':{'id':{'type':'string'},'intent':{'type':'string','enum':list(INTENTS)},'language':{'type':'string','enum':['vi','en']},'prompt':{'type':'string'}},
        'required':['id','intent','language','prompt']}}},'required':['items']}
    sp=job/'schema.json';sp.write_text(json.dumps(schema),encoding='utf-8')
    prompt=(root/'data/contracts/ANTIGRAVITY_G1_STATE_PARAPHRASES.md').read_text(encoding='utf-8')
    (job/'prompt.txt').write_text(prompt,encoding='utf-8')
    args=[str(cli),'--print',prompt,'--model','gemini-3.8-flash-low','--output-format','stream-json','--json-schema',str(sp),'--disable-slash-commands','--print-timeout','4m']
    began=time.perf_counter();events=job/'events.jsonl'
    with events.open('wb') as out,(job/'stderr.log').open('wb') as err:
        proc=subprocess.Popen(args,cwd=job,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        print(json.dumps({'run':run.relative_to(root).as_posix(),'pid':proc.pid,'status':'generating_public_prompts'}),flush=True)
        while proc.poll() is None:
            progress(run,'Antigravity đang sinh prompts; chưa học vào neuron',0,48,metrics={'elapsed_seconds':time.perf_counter()-began,'cli_process_id':proc.pid,'neural_training':False})
            if time.perf_counter()-began>270:proc.terminate();proc.wait(timeout=10);raise TimeoutError('Antigravity270s limit')
            time.sleep(2)
    parsed=[]
    for line in events.read_text(encoding='utf-8-sig').splitlines():
        try:parsed.append(json.loads(line))
        except ValueError:continue
    results=[e.get('result',e) for e in parsed if e.get('event')=='result' or e.get('status') in ('SUCCESS','ERROR')]
    if proc.returncode or not results or results[-1].get('status')!='SUCCESS':raise ValueError('Antigravity failed; private raw log retained, no data import')
    result=results[-1];value=result.get('structured_output')
    if value is None:value=json.loads(result['response'])
    tool_steps=[e['step_update'].get('tool_name') for e in parsed if e.get('step_update',{}).get('step_type')=='tool']
    if tool_steps:raise ValueError('Unexpected teacher tool use; retain output quarantined')
    items=value.get('items');assert isinstance(items,list) and len(items)==48
    if any(not isinstance(r,dict) or set(r)!=set(('id','intent','language','prompt')) or r['intent'] not in INTENTS or r['language'] not in ('vi','en') or not isinstance(r['prompt'],str) or not 1<=len(r['prompt'])<=180 for r in items):raise ValueError('Invalid public prompt schema')
    if any(sum(r['intent']==intent for r in items)!=6 for intent in INTENTS):raise ValueError('Intent counts do not match')
    if len(set(r['id'] for r in items))!=48 or len(set(r['prompt'].strip().casefold() for r in items))!=48:raise ValueError('Duplicate generated prompt/ID')
    raw=candidate/'prompts.json';write_json(raw,value)
    receipt={'run':run.relative_to(root).as_posix(),'candidate_directory':candidate.relative_to(root).as_posix(),
        'decision':'PROMPTS_READY_FOR_SEMANTIC_REVIEW','teacher':'gemini-3.8-flash-low','items':48,'trained':False,'validation_pending':True,
        'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'usage':result.get('usage'),'seconds':time.perf_counter()-began,
        'provenance':'actual installed Antigravity CLI; public intent brief only','tool_steps':tool_steps}
    write_json(candidate/'manifest.json',receipt);write_json(run/'result.json',receipt);write_json(root/'artifacts/antigravity-g1-state-latest.json',receipt)
    (run/'report.md').write_text('# Public Antigravity paraphrase generation\n\nNo private prompt, teacher tool use, model training or factual verification. Synthetic prompts require review.\n\n'+json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':str(run),'items':48,'trained':False,'validation_pending':True}),flush=True)


if __name__=='__main__':main()
