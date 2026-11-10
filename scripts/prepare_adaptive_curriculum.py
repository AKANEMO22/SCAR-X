"""Authorized AGY wording generation only. Never imports/teaches a neural model."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,tempfile,time,unicodedata
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.work_trace import activity

CHAT=('greet','identity','thanks','bye','learning','capability')
VISION=('single_color','single_shape','object_count','left_color')


def normalized(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def validate(items):
    if not isinstance(items,list) or len(items)!=128:raise ValueError('Exactly128rows')
    expected={f'{mod}_{family}_{lang}_{i:02d}' for mod,families,n in
              [('chat',CHAT,8),('vision',VISION,4)] for family in families
              for lang in ('vi','en') for i in range(1,n+1)}
    seen=set()
    for row in items:
        if not isinstance(row,dict) or set(row)!={'id','modality','family','language','prompt'}:raise ValueError('Row schema')
        if row['modality'] not in ('chat','vision') or row['language'] not in ('vi','en') or row['family'] not in (CHAT if row['modality']=='chat' else VISION):raise ValueError('Modality/family/language')
        if not isinstance(row['id'],str) or row['id'] not in expected or row['id'] in seen:raise ValueError('Row ID')
        prefix=f"{row['modality']}_{row['family']}_{row['language']}_"
        if not row['id'].startswith(prefix):raise ValueError('ID semantic metadata')
        text=row['prompt']
        if not isinstance(text,str) or not 1<=len(text)<=180 or not 3<=len(text.split())<=24 or any(ord(c)<32 for c in text):raise ValueError('Prompt budget')
        seen.add(row['id'])
    if seen!=expected:raise ValueError('Unbalanced IDs')
    return sorted(items,key=lambda r:r['id'])


def old_public_wording():
    """Only attributed project public candidates, never local/session DB/history."""
    result=set();paths=[]
    for name in ('direct-word-data-reviewed.json','vision-questions-reviewed.json','g1-state-prompts-reviewed.json','g1-curriculum-reviewed.json'):
        receipt=root/'artifacts'/name
        if not receipt.exists():continue
        meta=json.loads(receipt.read_text(encoding='utf-8'))
        for key in ('data','reviewed','path'):
            if not isinstance(meta.get(key),str):continue
            path=(root/meta[key]).resolve()
            if path.is_relative_to((root/'data/candidates').resolve()) and path.is_file() and path.stat().st_size<1_000_000:paths.append(path)
    def visit(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('prompt','question') and isinstance(v,str):result.add(normalized(v))
                elif isinstance(v,(dict,list)):visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    for path in set(paths):visit(json.loads(path.read_text(encoding='utf-8')))
    return result,[p.relative_to(root).as_posix() for p in sorted(set(paths))]


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cli=Path.home()/'.gemini/bin/agy.exe'
    if not cli.is_file():raise FileNotFoundError('Installed authorized AGYCLI missing')
    contract='data/contracts/ANTIGRAVITY_ADAPTIVE_CURRICULUM_V1.md'
    run=new_run(root,'adaptive-curriculum-data-v1',{'protocol':'experiments/adaptive-curriculum-data-v1.md',
        'display_title':'AGY ·128wording chat/vision mới · chỉ chuẩn bị data',
        'observability_kind':'data_generation','model_scope':'Publiccontract only. No SCAR-X training/inference, no newlearner. Development candidates require all-row review.',
        'teacher':'gemini-3.8-flash-low','items':128,'new_training':False},data_description='Publicwording contract; no privatechat/images/DB/weights/answers.',data_files=[contract])
    private=root/'local/adaptive-curriculum'/run.name;private.mkdir(parents=True)
    candidate=root/'data/candidates'/run.name;candidate.mkdir(parents=True)
    sandbox=Path(tempfile.mkdtemp(prefix='scarx-agy-wording-')).resolve()
    fields={k:{'type':'string'} for k in ('id','modality','family','language','prompt')}
    schema={'type':'object','properties':{'items':{'type':'array','minItems':128,'maxItems':128,'items':{'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}}},'required':['items'],'additionalProperties':False}
    schema_path=sandbox/'schema.json';write_json(schema_path,schema)
    args=[str(cli),'--print',(root/contract).read_text(encoding='utf-8'),'--model','gemini-3.8-flash-low','--output-format','stream-json','--json-schema',str(schema_path),'--disable-slash-commands','--print-timeout','4m']
    start=time.perf_counter();process=None;error=None;result=None
    (run/'report.md').write_text('# AGY developmentwording\n\nActualCLI call pending; no neurallearning.\n',encoding='utf-8')
    activity(run,'AntigravityCLI tạo wording chat/vision','128candidate công khai; không đáp án teacher/ảnh riêng/weights. Review toàn bộ trước dùng; không code hoặc chạy learner U.',status='started',evidence={'contract':contract,'new_training':False})
    events=private/'events.jsonl'
    with ResourceMonitor(limit=1_000_000_000,interval=.1) as monitor,events.open('wb') as out,(private/'stderr.log').open('wb') as err:
        try:
            process=subprocess.Popen(args,cwd=sandbox,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            print(json.dumps({'run':run.relative_to(root).as_posix(),'cli_pid':process.pid,'status':'public_wording_generation'}),flush=True)
            while process.poll() is None:
                monitor.check();progress(run,'AGYCLI đang tạo data thật, chưa học',0,128,metrics={'cli_pid':process.pid,'seconds':time.perf_counter()-start,'new_training':False})
                if time.perf_counter()-start>270:raise TimeoutError('270s teacherbudget')
                time.sleep(1)
            if events.stat().st_size>20_000_000:raise ValueError('Oversize teacherlog')
            parsed=[]
            for line in events.read_text(encoding='utf-8-sig').splitlines():
                try:parsed.append(json.loads(line))
                except ValueError:continue
            results=[e.get('result',e) for e in parsed if e.get('event')=='result' or e.get('status') in ('SUCCESS','ERROR')]
            if process.returncode or not results or results[-1].get('status')!='SUCCESS':raise ValueError('CLI did not return SUCCESS')
            response=results[-1];value=response.get('structured_output')
            if value is None:value=json.loads(response['response'])
            steps=[e['step_update'] for e in parsed if e.get('step_update',{}).get('step_type')=='tool']
            if any(s.get('tool_name')!='finish' or s.get('tool_info',{}).get('parameters')!=value for s in steps):raise ValueError('Unexpected toolsteps; candidate quarantined')
            items=validate(value.get('items'));old,sources=old_public_wording();seen=set();quarantine=[];clean=[]
            for row in items:
                key=normalized(row['prompt']);reason='duplicate_in_candidate' if key in seen else 'same_public_wording' if key in old else None
                seen.add(key)
                (quarantine if reason else clean).append({**row,**({'reason':reason} if reason else {})})
            path=candidate/'candidate.json';write_json(path,{'items':clean,'quarantine':quarantine,'semantic_review_pending':True,'automatic_learning_allowed':False,'targets_generated_by_teacher':False,'scope':'Developmentwording only, not unseenfinaltest.'})
            result={'run':run.relative_to(root).as_posix(),'decision':'ADAPTIVE_WORDING_REVIEW_PENDING','data':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'items':128,'candidate_rows':len(clean),'quarantined_rows':len(quarantine),'compared_public_sources':sources,'native_finish_count':len(steps),'teacher':'gemini-3.8-flash-low','semantic_review_pending':True}
        except Exception as exc:error=type(exc).__name__+': '+str(exc)[:220]
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        if result is None:result={'run':run.relative_to(root).as_posix(),'decision':'ADAPTIVE_WORDING_GENERATION_FAILED','candidate_rows':0}
        result.update(error=error,seconds=time.perf_counter()-start,resources=monitor.report(),private_content_sent=False,main_training=False,new_learning_rule_implemented=False,automatic_learning_allowed=False)
    (run/'report.md').write_text('# AGY actualdevelopmentwording\n\nNo maintraining, no newlearner, candidate review required.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_json(run/'result.json',result);write_json(root/'artifacts/adaptive-curriculum-data.json',result)
    activity(run,'Đã nhận data AGY' if error is None else 'CLI tạo data chưa đạt',result['decision']+'; không train/import model.',status='completed' if error is None else 'failed',evidence={k:result.get(k) for k in ('data','sha256','candidate_rows','quarantined_rows','seconds','error')})
    print(json.dumps({k:result.get(k) for k in ('run','decision','candidate_rows','quarantined_rows','seconds','error')},ensure_ascii=False))
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
