"""Actual authorized CLI call: public vision-question wording, no answers."""
from pathlib import Path
import sys,json,subprocess,os,time,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity

KINDS=('single_color','single_shape','object_count','left_color')


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cli=Path.home()/'.gemini/bin/agy.exe'
    if not cli.is_file():raise FileNotFoundError('Installed Antigravity CLI missing')
    contract='data/contracts/ANTIGRAVITY_VISION_QUESTIONS_V1.md'
    run=new_run(root,'antigravity-vision-questions-v1',{'protocol':'experiments/vision-question-path-audit-v1.md',
        'display_title':'AntigravityCLI · 32câu hỏi ảnh công khai · chưatraining',
        'observability_kind':'data_generation','model_scope':'Questionwording only. No privatecontent/answers/tools, proposal30 stillpending.',
        'teacher':'gemini-3.8-flash-low','items':32,'new_training':False},
        data_description='Public4visualattribute questiontypes only; no images, userchat, privatefiles or teacheranswers.',data_files=[contract])
    directory=root/'local/antigravity-vision-questions'/run.name;directory.mkdir(parents=True)
    candidate=root/'data/candidates'/run.name;candidate.mkdir(parents=True)
    schema={'type':'object','properties':{'items':{'type':'array','minItems':32,'maxItems':32,'items':{'type':'object',
        'properties':{'id':{'type':'string'},'kind':{'type':'string','enum':list(KINDS)},'language':{'type':'string','enum':['vi','en']},'question':{'type':'string'}},
        'required':['id','kind','language','question'],'additionalProperties':False}}},'required':['items'],'additionalProperties':False}
    sp=directory/'schema.json';sp.write_text(json.dumps(schema),encoding='utf-8');prompt=(root/contract).read_text(encoding='utf-8')
    args=[str(cli),'--print',prompt,'--model','gemini-3.8-flash-low','--output-format','stream-json','--json-schema',str(sp),'--disable-slash-commands','--print-timeout','4m']
    began=time.perf_counter();events=directory/'events.jsonl';process=None
    with events.open('wb') as stdout,(directory/'stderr.log').open('wb') as stderr:
        try:
            process=subprocess.Popen(args,cwd=directory,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            print(json.dumps({'run':run.relative_to(root).as_posix(),'pid':process.pid,'status':'generating_public_question_text'}),flush=True)
            while process.poll() is None:
                progress(run,'CLI sinh câu hỏi công khai; KHÔNG HUẤN LUYỆN',0,32,metrics={'elapsed_seconds':time.perf_counter()-began,'cli_pid':process.pid,'new_learning_signal_approved':False})
                if time.perf_counter()-began>270:raise TimeoutError('Registered270s CLI generationcap')
                time.sleep(2)
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    parsed=[]
    for line in events.read_text(encoding='utf-8-sig').splitlines():
        try:parsed.append(json.loads(line))
        except ValueError:continue
    results=[e.get('result',e) for e in parsed if e.get('event')=='result' or e.get('status') in ('SUCCESS','ERROR')]
    if process.returncode or not results or results[-1].get('status')!='SUCCESS':raise ValueError('CLI failed; rawoutput retained, no import')
    result=results[-1];value=result.get('structured_output')
    if value is None:value=json.loads(result['response'])
    steps=[e['step_update'] for e in parsed if e.get('step_update',{}).get('step_type')=='tool']
    if any(s.get('tool_name')!='finish' or s.get('tool_info',{}).get('parameters')!=value for s in steps):raise ValueError('Unexpected teacher toolstep; quarantine')
    items=value.get('items')
    if not isinstance(items,list) or len(items)!=32:raise ValueError('Question count')
    if any(not isinstance(r,dict) or set(r)!={'id','kind','language','question'} or r['kind'] not in KINDS or r['language'] not in ('vi','en') or not isinstance(r['question'],str) or not 1<=len(r['question'])<=180 for r in items):raise ValueError('Question schema')
    if len(set(r['id'] for r in items))!=32 or len(set(r['question'].strip().casefold() for r in items))!=32:raise ValueError('Duplicate IDs/questions')
    if any(sum(r['kind']==kind and r['language']==language for r in items)!=4 for kind in KINDS for language in ('vi','en')):raise ValueError('Unbalanced kinds/languages')
    write_json(candidate/'questions.json',{'items':items,'automatic_learning_allowed':False,'semantic_review_pending':True})
    receipt={'run':run.relative_to(root).as_posix(),'data':(candidate/'questions.json').relative_to(root).as_posix(),
        'decision':'VISION_QUESTIONS_READY_FOR_SEMANTIC_REVIEW','teacher':'gemini-3.8-flash-low','items':32,
        'sha256':hashlib.sha256((candidate/'questions.json').read_bytes()).hexdigest(),'seconds':time.perf_counter()-began,
        'native_finish_count':len(steps),'usage':result.get('usage'),'trained':False,'validation_pending':True,
        'automatic_learning_allowed':False,'private_content_sent':False,'teacher_answers_used':False,'proposal30_approved':False}
    (run/'report.md').write_text('# Actual Antigravity visual-question wording\n\nPubliccontract only; no sceneanswers or privateinputs; candidate requires semanticreview. No modeltraining or proposal30approval.\n\n'+json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    write_json(run/'result.json',receipt);write_json(root/'artifacts/antigravity-vision-questions-latest.json',receipt)
    activity(run,'Đã sinh32câu hỏi ảnh quaCLI thật','Candidatewording cầnsemanticreview; đáp án sẽ từrenderer, không teacher. Không training hoặc mở tínhiệuđangchờduyệt.',evidence={'items':32,'candidate_sha256':receipt['sha256'],'seconds':receipt['seconds'],'native_finish_count':len(steps)})
    print(json.dumps({'run':receipt['run'],'items':32,'trained':False,'semantic_review_pending':True}))


if __name__=='__main__':main()
