"""One premeasurement language-invalid AGY row correction, immutable sources."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    source=json.loads((root/'artifacts/antigravity-direct-word-prompts.json').read_text(encoding='utf-8'));raw=(root/source['data']).read_bytes();assert hashlib.sha256(raw).hexdigest()==source['sha256']
    old=next(r for r in json.loads(raw)['items'] if r['id']=='vi_greet_11');assert old['prompt']=='Hello bạn trợ lý!'
    run=new_run(root,'antigravity-direct-word-language-repair-v1',{'protocol':'experiments/direct-word-bridge-v1-language-repair.md',
        'display_title':'AGYCLI · sửa1câu sai ngônngữ trướctraining','observability_kind':'data_generation','model_scope':'One publiclanguage-schema repair only, no neuralquality/training/privateinput.','items':1},
        data_description='Public8stylecontract+one mixedlanguage row, no targetanswers/modelscore/privatecontext.',data_files=[source['data'],'experiments/direct-word-bridge-v1-language-repair.md'])
    directory=root/'local/antigravity-direct-word-repair'/run.name;directory.mkdir(parents=True);candidate=root/'data/candidates'/run.name;candidate.mkdir(parents=True)
    prompt='Return exactly one JSON item with id vi_greet_11, family vi_greet, language vi, prompt a natural greeting addressed to an assistant entirely in Vietnamese. The previous prompt Hello bạn trợ lý! violates language=vi because Hello is English. Use a different all-Vietnamese greeting, no English words, no answer, facts, tools, files or externalsearch. 1..180characters. This is synthetic publicwording only.'
    schema={'type':'object','properties':{k:{'type':'string'} for k in ('id','family','language','prompt')},'required':['id','family','language','prompt'],'additionalProperties':False}
    sp=directory/'schema.json';sp.write_text(json.dumps(schema),encoding='utf-8');(run/'report.md').write_text('# One language repair\n\nPremeasurement AGY repair in progress; originalrow quarantined, no modeltraining.\n',encoding='utf-8')
    activity(run,'Cách ly1câu mixedlanguage, sửa trước học','Đã đọc128prompts; chỉvi_greet_11 sai languagecontract. Không thay candidategốc/khácrow hoặc lựa chọn theo modelscore.',status='started',evidence={'source_sha256':source['sha256'],'quarantined_id':'vi_greet_11'})
    process=None;began=time.perf_counter();events=directory/'events.jsonl'
    with events.open('wb') as out,(directory/'stderr.log').open('wb') as err:
        try:
            process=subprocess.Popen([str(Path.home()/'.gemini/bin/agy.exe'),'--print',prompt,'--model','gemini-3.8-flash-low','--output-format','stream-json','--json-schema',str(sp),'--disable-slash-commands','--print-timeout','4m'],cwd=directory,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            while process.poll() is None:
                progress(run,'Sửa1prompt publiclanguage, chưatraining',0,1,metrics={'seconds':time.perf_counter()-began})
                if time.perf_counter()-began>270:raise TimeoutError('Registered270s repaircap')
                time.sleep(1)
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
    if process.returncode or not results or results[-1].get('status')!='SUCCESS':raise ValueError('Repairfailed, quarantine')
    result=results[-1];value=result.get('structured_output')
    if value is None:value=json.loads(result['response'])
    steps=[e['step_update'] for e in parsed if e.get('step_update',{}).get('step_type')=='tool']
    if any(s.get('tool_name')!='finish' or s.get('tool_info',{}).get('parameters')!=value for s in steps):raise ValueError('Unexpectedteacher toolstep')
    if set(value)!={'id','family','language','prompt'} or value['id']!='vi_greet_11' or value['family']!='vi_greet' or value['language']!='vi' or not 1<=len(value['prompt'])<=180:raise ValueError('Invalid repairrow')
    p=candidate/'repair.json';write_json(p,{'original':old,'replacement':value,'semantic_review_pending':True,'automatic_learning_allowed':False})
    receipt={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'data':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'decision':'ONE_LANGUAGE_REPAIR_REQUIRES_REVIEW','items':1,'new_training':False,'private_content_sent':False,'teacher_answers_used':False,'seconds':time.perf_counter()-began,'usage':result.get('usage')}
    (run/'report.md').write_text('# One language repair\n\nOriginalcandidate retained, replacement requires review before anytraining.\n\n'+json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    write_json(run/'result.json',receipt);write_json(root/'artifacts/antigravity-direct-word-repair.json',receipt);print(json.dumps({'run':receipt['run'],'replacement':value},ensure_ascii=True))


if __name__=='__main__':main()
