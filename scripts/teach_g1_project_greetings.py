"""Two explicit source-aware live teaching examples, existing controller/guard."""
from pathlib import Path
import sys,json,time,urllib.request,uuid
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress


def main():
    base='http://127.0.0.1:8765'
    def get(path):return json.load(urllib.request.urlopen(base+path,timeout=20))
    token=get('/api/status')['token']
    def post(path,data):
        request=urllib.request.Request(base+path,json.dumps(data).encode(),headers={'Content-Type':'application/json','X-SCARX-Token':token})
        return json.load(urllib.request.urlopen(request,timeout=20))
    run=new_run(root,'g1-project-supervision-acceptance-v1',{'protocol':'experiments/g1-project-supervision-acceptance-v1.md',
        'display_title':'G1 đang dùng · 2ví dụ chào hỏi có nguồn và guard',
        'model_scope':'Actual existing live learner, project supervisor labels; no failed-clone import or normalization enabled',
        'examples':2,'new_hidden_rule':False,'attempt_limit_per_lesson':32})
    source=get('/api/status');results=[]
    for family in (0,1):
        before=post('/api/project-probe',{'family':family,'example':0})
        response=post('/v1/learning/message',{'kind':'project_lesson','client_id':'project-'+uuid.uuid4().hex,'family':family,'example':0})
        deadline=time.monotonic()+30
        while get('/api/status')['learning']['queued_lessons'] and time.monotonic()<deadline:time.sleep(.5)
        after=post('/api/project-probe',{'family':family,'example':0});status=get('/api/status')
        row={'family':family,'before_actual_neural':before.get('neural_draft',before['text']),
            'after_actual_neural':after.get('neural_draft',after['text']),'lesson_id':response.get('lesson_id'),
            'source':response.get('source'),'source_sha256':response.get('source_sha256'),
            'checkpoint_before':response.get('checkpoint_before'),'skipped':response.get('skipped',False),
            'queue_drained':status['learning']['queued_lessons']==0,'updates_after':status['updates'],
            'workload':after['workload'],'note':'Only an explicit supervised trainingpoint, not general language or retained skill proof'}
        results.append(row)
        progress(run,'Đã kiểm ví dụ '+str(family),len(results),2,metrics={'queue_drained':row['queue_drained'],'actual_updates':status['updates']})
    end=get('/api/status');result={'run':run.relative_to(root).as_posix(),'decision':'EXPLICIT_TEACHING_OBSERVED_ONLY','examples':results,
        'updates_before':source['updates'],'updates_after':end['updates'],'learning_before':source['learning'],'learning_after':end['learning'],
        'failed_curriculum_imported':False,'surface_normalization_enabled':False,'normal_chat_or_vision_proven':False,'retention_confirmed':False,
        'live_model_id':end['model'],'cores_loaded':end['cores_loaded'],'resources':end['resources'],'private_history_preserved':True}
    (run/'report.md').write_text('# Source-aware explicit live teaching\n\nTwo actual trainingexamples via existing guard/controller; no modelreplacement, failed-cloneimport, response lookup or qualityclaim.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    write_json(run/'result.json',result);write_json(root/'artifacts/g1-project-supervision-latest.json',result)
    print(json.dumps({'run':str(run),'examples':results,'updates_before':source['updates'],'updates_after':end['updates']}))


if __name__=='__main__':main()
