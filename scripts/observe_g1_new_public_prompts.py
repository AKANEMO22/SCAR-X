"""Real live inference on reviewed48prompts, no lesson/memory labels; autoarchive comparison."""
from pathlib import Path
import sys,json,urllib.request
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress


def main():
    base='http://127.0.0.1:8765'
    def get(path):return json.load(urllib.request.urlopen(base+path,timeout=20))
    before=get('/api/status');token=before['token']
    source=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'))
    cases=json.loads((root/source['data']).read_text(encoding='utf-8'))['items']
    run=new_run(root,'g1-live-new-prompts-observation',{'display_title':'Live SCAR-X · 48câu mới và Qwen đã đo · inference thật',
        'source_dataset':source['data'],'model_scope':'Actual serving core, continuous state. No curriculumweights imported or labels learned.',
        'cases':48,'new_training':False,'script_scope':'No parity/equivalence claim; compare actual observed answers with timestamped reference'})
    outputs=[]
    for index,case in enumerate(cases):
        request=urllib.request.Request(base+'/api/observe-answer',json.dumps({'dataset':'g1_state','case_id':case['id']}).encode(),
            headers={'Content-Type':'application/json','X-SCARX-Token':token})
        result=json.load(urllib.request.urlopen(request,timeout=20))
        outputs.append({'id':case['id'],'prompt':case['prompt'],'generated':result.get('neural_draft',result['text']),
            'workload':result['workload'],'checkpoint':result['checkpoint']})
        progress(run,'Inference thực · '+case['id'],index+1,48,metrics={'characters_generated':len(outputs[-1]['generated']),'active':result['workload']['active_neurons']})
    after=get('/api/status')
    # Public input, private-model raw answers remain local rather than in package.
    private=root/'local/g1-live-observations'/run.name;private.mkdir(parents=True);write_json(private/'outputs.json',outputs)
    result={'run':run.relative_to(root).as_posix(),'decision':'ACTUAL_LIVE_INFERENCE_OBSERVED_ONLY','completed':len(outputs),'expected':48,
        'updates_before':before['updates'],'updates_after':after['updates'],'new_training':False,
        'live_model_id':after['model'],'cores_loaded':after['cores_loaded'],'normal_chat_or_vision_proven':False,
        'input_scope':'SCAR-X continuous recurrent state, Qwen archived single-prompt context; not a matched-state/training benchmark',
        'private_raw_answers_saved_locally':True,'resources':after['resources']}
    (run/'report.md').write_text('# Actual live inference on48reviewed public utterances\n\nRaw private-model answers shown locally/ontrace, no target labels learned; archivedQwen matched only by exactpublicprompt. No new capability certification.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    write_json(run/'result.json',result);write_json(root/'artifacts/g1-live-new-prompts-latest.json',result)
    print(json.dumps({'run':str(run),'completed':len(outputs),'updates_before':before['updates'],'updates_after':after['updates']}))


if __name__=='__main__':main()
