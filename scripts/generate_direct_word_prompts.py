"""Actual authorized installed AGYCLI; public wording only, no teacheranswers."""
from pathlib import Path
import sys,json,subprocess,os,time,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.direct_word_data import FAMILIES,validate
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.work_trace import activity
from scarx.runtime.experiment_status import progress
from scarx.runtime.resources import ResourceMonitor


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cli=Path.home()/'.gemini/bin/agy.exe';assert cli.is_file();contract='data/contracts/ANTIGRAVITY_DIRECT_WORD_BRIDGE_V1.md'
    run=new_run(root,'antigravity-direct-word-prompts-v1',{'protocol':'experiments/direct-word-bridge-v1.md','display_title':'AntigravityCLI ·128publicchatprompts, chưahọc',
        'observability_kind':'data_generation','model_scope':'Public8stylepromptwording only; semanticreview required, no privateinputs/answers/neuraltraining.',
        'teacher':'gemini-3.8-flash-low','items':128,'new_training':False},data_description='Only projectpubliccontract, no privateuserchat/history/files/images/weights.',data_files=[contract])
    private=root/'local/antigravity-direct-word'/run.name;private.mkdir(parents=True);candidate=root/'data/candidates'/run.name;candidate.mkdir(parents=True)
    schema={'type':'object','properties':{'items':{'type':'array','minItems':128,'maxItems':128,'items':{'type':'object','properties':{'id':{'type':'string'},'family':{'type':'string','enum':list(FAMILIES)},'language':{'type':'string','enum':['vi','en']},'prompt':{'type':'string'}},'required':['id','family','language','prompt'],'additionalProperties':False}}},'required':['items'],'additionalProperties':False}
    schema_path=private/'schema.json';schema_path.write_text(json.dumps(schema),encoding='utf-8');prompt=(root/contract).read_text(encoding='utf-8')
    args=[str(cli),'--print',prompt,'--model','gemini-3.8-flash-low','--output-format','stream-json','--json-schema',str(schema_path),'--disable-slash-commands','--print-timeout','4m']
    (run/'report.md').write_text('# Actual AGY promptgeneration\n\nGeneration in progress; publiccontract only. No neuraltraining/teacheranswers; candidate requiresmanualreview.\n',encoding='utf-8')
    activity(run,'Đang gọiAntigravityCLI tạo128prompt mới','Public4intent×2language, chỉwording. Nhãnstyle dựán riêng; không privatehistory/teachertruth hoặc modeltraining.',status='started',evidence={'contract':contract,'expected_items':128})
    events=private/'events.jsonl';process=None;began=time.perf_counter()
    with ResourceMonitor(interval=.1) as monitor,events.open('wb') as out,(private/'stderr.log').open('wb') as err:
        try:
            process=subprocess.Popen(args,cwd=private,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            print(json.dumps({'run':run.relative_to(root).as_posix(),'cli_pid':process.pid,'status':'generating_public_wording'}),flush=True)
            while process.poll() is None:
                progress(run,'AGYCLI tạo dữ liệu thật, chưa học vào neuron',0,128,metrics={'seconds':time.perf_counter()-began,'cli_pid':process.pid,'new_training':False});monitor.check()
                if time.perf_counter()-began>270:raise TimeoutError('Registered270s teacherlimit')
                time.sleep(1)
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        if events.stat().st_size>20_000_000:raise ValueError('Oversized teacherstream quarantined')
        parsed=[]
        for line in events.read_text(encoding='utf-8-sig').splitlines():
            try:parsed.append(json.loads(line))
            except ValueError:continue
        results=[e.get('result',e) for e in parsed if e.get('event')=='result' or e.get('status') in ('SUCCESS','ERROR')]
        if process.returncode or not results or results[-1].get('status')!='SUCCESS':raise ValueError('Teacherfailed; retainedprivateoutput, no import')
        result=results[-1];value=result.get('structured_output')
        if value is None:value=json.loads(result['response'])
        steps=[e['step_update'] for e in parsed if e.get('step_update',{}).get('step_type')=='tool']
        if any(s.get('tool_name')!='finish' or s.get('tool_info',{}).get('parameters')!=value for s in steps):raise ValueError('Unexpectedteacher toolstep; quarantine')
        items=validate(value.get('items'));path=candidate/'prompts.json'
        write_json(path,{'items':items,'automatic_learning_allowed':False,'semantic_review_pending':True,'teacher_answers_used':False})
        receipt={'run':run.relative_to(root).as_posix(),'data':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'decision':'DIRECT_WORD_PROMPTS_REQUIRE_SEMANTIC_REVIEW','items':128,'teacher':'gemini-3.8-flash-low','native_finish_count':len(steps),
            'seconds':time.perf_counter()-began,'usage':result.get('usage'),'automatic_learning_allowed':False,'validation_pending':True,'trained':False,
            'private_content_sent':False,'teacher_answers_used':False,'resources':monitor.report()}
        (run/'report.md').write_text('# Actual AGY wordpromptgeneration\n\nCandidate only, manualreview required. No targetanswers/privatecontent/neuraltraining.\n\n'+json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',receipt);write_json(root/'artifacts/antigravity-direct-word-prompts.json',receipt)
        activity(run,'AGYCLI đã tạo128prompt công khai',receipt['decision']+'; giữcandidate/rawlogs, chưaimportmodel.',evidence={'data':receipt['data'],'sha256':receipt['sha256'],'seconds':receipt['seconds'],'items':128})
        print(json.dumps({k:receipt[k] for k in ('run','data','decision','items','seconds')},ensure_ascii=False))


if __name__=='__main__':main()
