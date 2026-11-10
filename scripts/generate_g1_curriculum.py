"""Two authorized small public-text Antigravity batches; no learner import."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.neural.language import tokenize
from generate_g1_state_paraphrases import INTENTS


def key(text):
    words=tokenize(text)
    while words and words[-1] in ('.','!','?'):words.pop()
    return tuple(words)


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cli=Path.home()/'.gemini/bin/agy.exe'
    if not cli.is_file():raise FileNotFoundError('Installed Antigravity CLI missing')
    run=new_run(root,'antigravity-g1-curriculum-v1',{'display_title':'Antigravity · curriculum128train/32development · chưa học',
        'observability_kind':'data_generation','batches':2,'items':160,'teacher':'gemini-3.8-flash-low',
        'model_scope':'Public phrasing curriculum only; no personalinput or neuraltraining'},
        data_description='Public8intent brief and previously public prompts only, no userprivate model/feedback',data_files=['data/contracts/ANTIGRAVITY_G1_CURRICULUM_V1.md'])
    job=root/'local/antigravity-g1-curriculum'/run.name;job.mkdir(parents=True)
    candidate=root/'data/candidates'/run.name;candidate.mkdir(parents=True)
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
    forbidden=[p for f in corpus['families'] for s in ('train','development','acceptance') for p in f[s]]
    source=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'))
    forbidden += [r['prompt'] for r in json.loads((root/source['data']).read_text(encoding='utf-8'))['items']]
    schema={'type':'object','properties':{'items':{'type':'array','minItems':80,'maxItems':80,'items':{'type':'object',
        'properties':{'id':{'type':'string'},'intent':{'type':'string','enum':list(INTENTS)},'language':{'type':'string','enum':['vi','en']},
            'split':{'type':'string','enum':['train','development']},'prompt':{'type':'string'}},'required':['id','intent','language','split','prompt']}}},'required':['items']}
    sp=job/'schema.json';sp.write_text(json.dumps(schema),encoding='utf-8');records=[];items=[];began=time.perf_counter()
    for batch in range(2):
        prefix=f'c{batch+1}-';prompt=(root/'data/contracts/ANTIGRAVITY_G1_CURRICULUM_V1.md').read_text(encoding='utf-8')
        prompt+='\nBatch prefix: '+prefix+'\nForbidden public prompts (do not duplicate terminal-punctuation variants):\n'+json.dumps(forbidden,ensure_ascii=False)
        (job/f'batch{batch+1}-prompt.txt').write_text(prompt,encoding='utf-8')
        args=[str(cli),'--print',prompt,'--model','gemini-3.8-flash-low','--output-format','stream-json','--json-schema',str(sp),'--disable-slash-commands','--print-timeout','4m']
        started=time.perf_counter();events=job/f'batch{batch+1}.jsonl'
        with events.open('wb') as out,(job/f'batch{batch+1}.stderr.log').open('wb') as err:
            proc=subprocess.Popen(args,cwd=job,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            while proc.poll() is None:
                progress(run,f'Antigravity đang sinh lô{batch+1}/2; chưa học',batch*80,160,metrics={'cli_process_id':proc.pid,'elapsed_seconds':time.perf_counter()-started,'neural_training':False})
                if time.perf_counter()-started>270:proc.terminate();proc.wait(timeout=10);raise TimeoutError('CLI batch270s deadline')
                time.sleep(2)
        parsed=[]
        for line in events.read_text(encoding='utf-8-sig').splitlines():
            try:parsed.append(json.loads(line))
            except ValueError:continue
        results=[e.get('result',e) for e in parsed if e.get('event')=='result' or e.get('status') in ('SUCCESS','ERROR')]
        if proc.returncode or not results or results[-1].get('status')!='SUCCESS':raise ValueError('Antigravity batch failed; no data import')
        result=results[-1];value=result.get('structured_output')
        if value is None:value=json.loads(result['response'])
        tool_steps=[e['step_update'] for e in parsed if e.get('step_update',{}).get('step_type')=='tool']
        tools=[step.get('tool_name') for step in tool_steps]
        # JSON-schema finalization may itself be surfaced as a native `finish`
        # tool. Accept only an exact copy of the final structured payload.
        if any(s.get('tool_name')!='finish' or s.get('tool_info',{}).get('parameters')!=value for s in tool_steps):raise ValueError('Unexpected teacher tools; quarantine output')
        raw=candidate/f'batch{batch+1}.json';write_json(raw,value)
        data=value.get('items')
        if not isinstance(data,list) or len(data)!=80:raise ValueError('Expected80items')
        for intent in INTENTS:
            if sum(r.get('intent')==intent and r.get('split')=='train' for r in data)!=8 or sum(r.get('intent')==intent and r.get('split')=='development' for r in data)!=2:raise ValueError('Unexpected intent/split count')
        if any(set(r)!=set(('id','intent','language','split','prompt')) or not r['id'].startswith(prefix) or r['language'] not in ('vi','en') or not isinstance(r['prompt'],str) or not 1<=len(r['prompt'])<=180 for r in data):raise ValueError('Invalid prompt metadata')
        exposure={key(p) for p in forbidden};duplicates=[r['id'] for r in data if key(r['prompt']) in exposure]
        internal=len(data)-len(set(key(r['prompt']) for r in data))
        records.append({'batch':batch+1,'items':len(data),'seconds':time.perf_counter()-started,'usage':result.get('usage'),
            'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'forbidden_surface_duplicates':duplicates,'internal_surface_duplicates':internal,'tool_steps':tools})
        items+=data;forbidden += [r['prompt'] for r in data]
        progress(run,f'Đã sinh lô{batch+1}; cần review, chưa học',len(items),160,metrics={'duplicates_to_review':len(duplicates)+internal,'neural_training':False})
    write_json(candidate/'curriculum.json',{'items':items,'automatic_learning_allowed':False,'review_pending':True})
    receipt={'run':run.relative_to(root).as_posix(),'candidate_directory':candidate.relative_to(root).as_posix(),'decision':'CURRICULUM_GENERATED_REVIEW_PENDING',
        'batches':records,'items':len(items),'train':sum(r['split']=='train' for r in items),'development':sum(r['split']=='development' for r in items),
        'source':'actual installed Antigravity CLI; public prompts only','new_training':False,'automatic_learning_allowed':False,
        'validation_pending':True,'sha256':hashlib.sha256((candidate/'curriculum.json').read_bytes()).hexdigest(),'seconds':time.perf_counter()-began}
    write_json(candidate/'manifest.json',receipt);write_json(run/'result.json',receipt);write_json(root/'artifacts/antigravity-g1-curriculum-latest.json',receipt)
    (run/'report.md').write_text('# Bounded public G1 phrasing curriculum\n\nActual CLI generation, synthetic prompts require fullsemantic/splitreview; no automatic training or factualproof.\n\n'+json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':str(run),'items':len(items),'validation_pending':True,'trained':False}),flush=True)


if __name__=='__main__':main()
