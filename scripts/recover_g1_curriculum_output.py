"""Recover actual CLI output after false-positive native schema-finish rejection."""
from pathlib import Path
import sys,json,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from generate_g1_curriculum import key,INTENTS


def main():
    source='20261109T163500Z-antigravity-g1-curriculum-v1-4c2593';job=root/'local/antigravity-g1-curriculum'/source
    run=new_run(root,'g1-curriculum-output-recovery',{'display_title':'Khôi phục output Antigravity · native JSON finish, không sinh lại',
        'observability_kind':'data_generation','source_run':'runs/'+source,'new_teacher_calls':0,
        'model_scope':'Actual prior CLI output only; strict schema/finish equality, no neuraltraining'},
        data_description='Public actual generated prompts retained in private CLI logs; no new external request')
    candidate=root/'data/candidates'/run.name;candidate.mkdir(parents=True)
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
    forbidden={key(p) for f in corpus['families'] for split in ('train','development','acceptance') for p in f[split]}
    info=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'))
    forbidden|={key(r['prompt']) for r in json.loads((root/info['data']).read_text(encoding='utf-8'))['items']}
    items=[];records=[]
    for index in (1,2):
        events=[];path=job/f'batch{index}.jsonl'
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            try:events.append(json.loads(line))
            except ValueError:continue
        results=[e.get('result',e) for e in events if e.get('event')=='result' or e.get('status') in ('SUCCESS','ERROR')]
        assert results and results[-1]['status']=='SUCCESS';result=results[-1];value=result.get('structured_output')
        if value is None:value=json.loads(result['response'])
        steps=[e['step_update'] for e in events if e.get('step_update',{}).get('step_type')=='tool']
        assert all(s.get('tool_name')=='finish' and s.get('tool_info',{}).get('parameters')==value for s in steps)
        data=value['items'];assert len(data)==80
        assert all(set(r)==set(('id','intent','language','split','prompt')) and r['intent'] in INTENTS and r['language'] in ('vi','en') and r['split'] in ('train','development') and isinstance(r['prompt'],str) and 1<=len(r['prompt'])<=180 for r in data)
        for intent in INTENTS:
            assert sum(r['intent']==intent and r['split']=='train' for r in data)==8
            assert sum(r['intent']==intent and r['split']=='development' for r in data)==2
        duplicates=[r['id'] for r in data if key(r['prompt']) in forbidden];internal=len(data)-len(set(key(r['prompt']) for r in data))
        records.append({'batch':index,'items':len(data),'usage':result.get('usage'),'native_schema_finalizers':len(steps),
            'forbidden_surface_duplicates':duplicates,'internal_surface_duplicates':internal,
            'original_event_log_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        write_json(candidate/f'batch{index}.json',value);items+=data;forbidden|={key(r['prompt']) for r in data}
    assert len(set(r['id'] for r in items))==160
    write_json(candidate/'curriculum.json',{'items':items,'automatic_learning_allowed':False,'review_pending':True})
    receipt={'run':run.relative_to(root).as_posix(),'source_generation_run':'runs/'+source,'candidate_directory':candidate.relative_to(root).as_posix(),
        'decision':'CURRICULUM_RECOVERED_REVIEW_PENDING','batches':records,'items':160,'train':128,'development':32,
        'source':'actual Antigravity gemini-3.8-flash-low CLI, output recovered without new teacher call',
        'native_finish_matches_final_payload':True,'original_failure_retained':True,'new_teacher_calls':0,
        'new_training':False,'validation_pending':True,'automatic_learning_allowed':False,
        'sha256':hashlib.sha256((candidate/'curriculum.json').read_bytes()).hexdigest()}
    write_json(candidate/'manifest.json',receipt);write_json(run/'result.json',receipt);write_json(root/'artifacts/antigravity-g1-curriculum-latest.json',receipt)
    (run/'report.md').write_text('# Recover actual Antigravity curriculum output\n\nOriginal run stopped before import because a native JSON-schemafinish event was treated as an externaltool. Allfinish payloads exactly match finalstructuredoutput; no newCLI call or neuraltraining. Originaljournal retained.\n\n'+json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':str(run),'items':160,'batches':records,'review_pending':True}))


if __name__=='__main__':main()
