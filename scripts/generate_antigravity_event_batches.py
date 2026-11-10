"""Bounded retry after verified CLI timeout; four small public-text-only batches."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,os,json,time,hashlib,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
root=Path(__file__).resolve().parents[1];cli=Path.home()/'.gemini/bin/agy.exe'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
job=root/'local/antigravity-event-dialogue-jobs'/stamp;job.mkdir(parents=True)
candidate=root/'data/candidates'/('antigravity-events-'+stamp);candidate.mkdir(parents=True)
brief=(root/'data/contracts/ANTIGRAVITY_DIALOGUE_EVENTS_V1.md').read_text(encoding='utf-8')
turn={'type':'object','properties':{'role':{'type':'string','enum':['user','assistant']},'text':{'type':'string'}},'required':['role','text']}
dialog={'type':'object','properties':{'id':{'type':'string'},'language':{'type':'string','enum':['vi','en']},'split':{'type':'string','enum':['train','development']},'scenario_id':{'type':'string'},'template_id':{'type':'string'},'turns':{'type':'array','minItems':6,'maxItems':6,'items':turn}},'required':['id','language','split','scenario_id','template_id','turns']}
schema={'type':'object','properties':{'dialogs':{'type':'array','minItems':10,'maxItems':10,'items':dialog},'notes':{'type':'string'}},'required':['dialogs','notes']}
schema_path=job/'schema.json';schema_path.write_text(json.dumps(schema),encoding='utf-8')


def generate(index):
    vi=6 if index<2 else 4;en=8-vi
    prompt=(brief+f'\nThis call is batch{index+1}/4. Override ONLY the overall count: return exactly10dialogs for this batch,8train({vi}vi+{en}en),2development(1vi+1en). Prefix all id/template_id values b{index+1}-. '
      'Development wording templates must differ from train. Keep the combined assistant vocabulary simple/common, around60unique words in this batch, each reply1–8words. '
      'Do not label facts verified. No real user data. Do not use any tools/files/web/shell/skills/subagents. Return schema JSON only.')
    (job/f'batch-{index+1}-prompt.txt').write_text(prompt,encoding='utf-8')
    args=[str(cli),'--print',prompt,'--model','gemini-3.8-flash-low','--output-format','stream-json','--json-schema',str(schema_path),'--disable-slash-commands','--print-timeout','4m']
    began=time.monotonic();events_path=job/f'batch-{index+1}.events.jsonl'
    with events_path.open('wb') as out,(job/f'batch-{index+1}.stderr.log').open('wb') as err:
        proc=subprocess.Popen(args,cwd=job,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        print(json.dumps({'batch':index+1,'pid':proc.pid,'status':'generating'}),flush=True)
        try:code=proc.wait(timeout=270)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10);raise TimeoutError('Batch270s budget')
    events=[]
    for line in events_path.read_text(encoding='utf-8-sig').splitlines():
        try:events.append(json.loads(line))
        except ValueError:continue
    results=[e.get('result',e) for e in events if e.get('event')=='result' or e.get('status') in ('SUCCESS','ERROR')]
    if not results or code or results[-1].get('status')!='SUCCESS':
        error={'batch':index+1,'success':False,'seconds':time.monotonic()-began,'status':results[-1].get('status') if results else 'missing_result'}
        (candidate/f'batch-{index+1}-failure.json').write_text(json.dumps(error,indent=2)+'\n',encoding='utf-8');return error
    result=results[-1];value=result.get('structured_output')
    if value is None:value=json.loads(result['response'])
    path=candidate/f'batch-{index+1}.json';path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    steps=[{'name':e['step_update'].get('tool_name'),'state':e['step_update'].get('state')} for e in events if e.get('step_update',{}).get('step_type')=='tool']
    return {'batch':index+1,'success':True,'conversation_id':result.get('conversation_id'),'usage':result.get('usage'),
      'seconds':time.monotonic()-began,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tool_steps':steps}


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    sys.path.insert(0,str(root/'src'))
    from scarx.evaluation.runs import new_run,write_json
    from scarx.runtime.experiment_status import progress
    run=new_run(root,'antigravity-event-data-generation',{'observability_kind':'data_generation','display_title':'Antigravity tạo curriculum hội thoại',
      'batches':4,'teacher':'gemini-3.8-flash-low','neural_training':False},data_description='Public synthetic brief only; raw CLI logs local, no private dialogue or imported weights')
    progress(run,'Antigravity đang sinh dữ liệu; chưa học vào neuron',0,4)
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(generate,i) for i in range(4)]):
            records.append(result.result());print(json.dumps(records[-1]),flush=True)
            progress(run,'Đã nhận lô dữ liệu '+str(records[-1]['batch']),len(records),4,
              metrics={'batch_success':records[-1]['success'],'seconds':records[-1]['seconds'],'neural_training':False})
    records.sort(key=lambda x:x['batch'])
    manifest={'source':'synthetic:antigravity','model':'gemini-3.8-flash-low','created_utc':stamp,'batches':records,'trained':False,'prior_cli_timeout':'20261109T092308Z; original job retained','data_contract':'data/contracts/ANTIGRAVITY_DIALOGUE_EVENTS_V1.md'}
    if all(x['success'] for x in records):
        dialogs=[]
        for i in range(1,5):dialogs+=json.loads((candidate/f'batch-{i}.json').read_text(encoding='utf-8'))['dialogs']
        path=candidate/'dialogue.json';path.write_text(json.dumps({'dialogs':dialogs,'notes':'Four separately generated small batches; validate cross-batch balance/leakage before use.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        manifest['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        (root/'artifacts/antigravity-events-latest.json').write_text(json.dumps({'candidate_directory':candidate.relative_to(root).as_posix(),'validation_pending':True,'trained':False},indent=2)+'\n',encoding='utf-8')
    (candidate/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    outcome={'run':run.relative_to(root).as_posix(),'successful_batches':sum(x['success'] for x in records),'expected_batches':4,
      'candidate_directory':candidate.relative_to(root).as_posix(),'trained':False,'validation_pending':True,
      'decision':'DATA_READY_FOR_VALIDATION' if all(x['success'] for x in records) else 'DATA_GENERATION_INCOMPLETE'}
    write_json(run/'result.json',outcome);write_json(root/'artifacts/antigravity-generation-latest.json',outcome)
    (run/'report.md').write_text('# Antigravity curriculum generation\n\n'+json.dumps(outcome,indent=2)+'\n\nGenerated synthetic data is not independently verified and has not been learned by the neural core.\n',encoding='utf-8')
    print(json.dumps({'candidate':str(candidate),'successful_batches':sum(x['success'] for x in records),'expected_batches':4}),flush=True)
