"""Actual serving inference + one verified local lesson, published as it happens."""
from pathlib import Path
import json,sys,time,urllib.request,uuid
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    base='http://127.0.0.1:8765'
    def get(path):return json.load(urllib.request.urlopen(base+path,timeout=25))
    token=get('/api/status')['token']
    def post(path,data):
        request=urllib.request.Request(base+path,json.dumps(data).encode(),headers={'Content-Type':'application/json','X-SCARX-Token':token})
        return json.load(urllib.request.urlopen(request,timeout=25))
    run=new_run(root,'live-observatory-acceptance-v1',{
        'display_title':'Inference thực + học số học trên lõi đang dùng',
        'observability_kind':'serving_measurement','protocol':'experiments/live-observatory-acceptance-v1.md',
        'model_scope':'Lõi đang dùng; checkpoint trước lượt; targets không đưa vào inference',
        'development_cases':24,'lesson':'7 + 8 → 15','training_attempt_limit':32,'wait_limit_seconds':60})
    checkpoint=post('/api/checkpoint',{})['checkpoint'];outputs=[]
    reference=json.loads((root/'artifacts/qwen-dialogue-reference-latest.json').read_text(encoding='utf-8'))
    for index,case in enumerate(reference['outputs']):
        result=post('/api/observe-answer',{'case_id':case['id']})
        outputs.append({'id':case['id'],'generated':result.get('neural_draft',result['text']),
            'one_reference_target':case['one_reference_target'],'workload':result['workload'],'checkpoint':result['checkpoint']})
        progress(run,'Sinh câu trả lời thực trên lõi đang dùng',index+1,24,
            metrics={'case_id':case['id'],'cases_finished':len(outputs)},task=case['messages'][-1]['text'])
        write_json(run/'answers.json',outputs)
    before=get('/api/trace');began=time.time()
    post('/v1/learning/message',{'kind':'lesson','client_id':'obs-'+uuid.uuid4().hex,'prompt':'7 + 8','answer':'15'})
    deadline=time.monotonic()+60
    while True:
        trace=get('/api/trace');status=get('/api/status')
        learning=[r for r in trace['learning'] if r['time']>=began]
        neural=next((r for r in reversed(status['trace']) if 'spikes' in r),None)
        progress(run,'Quan sát cập nhật synapse và guard thật',len(learning),32,
            metrics={'learning_events':len(learning),'queued_lessons':trace['controller']['queued'],
                'accepted_since_start':trace['controller']['accepted']-before['controller']['accepted'],
                'rejected_since_start':trace['controller']['rejected']-before['controller']['rejected']},
            neural=neural,
            task='7 + 8 → 15')
        if not trace['controller']['queued'] or time.monotonic()>=deadline:break
        time.sleep(1)
    result={'run':str(run.relative_to(root)).replace('\\','/'),'decision':'MEASURED_VISIBILITY_ONLY',
        'checkpoint_before':checkpoint,'checkpoint_after':status['checkpoint'],'answers':outputs,
        'learning_events':learning,'controller_before':before['controller'],'controller_after':trace['controller'],
        'queue_drained':not trace['controller']['queued'],'normal_chat_or_vision_proven':False,
        'resources':status.get('resources',{}),'qwen_reference':reference['run'],
        'input_scope':'SCAR-X latest user text vs archived Qwen full prefix; unequal context; no quality-equivalence claim'}
    (run/'report.md').write_text('# Real serving observability acceptance\n\n24 actual inferences and one explicit verified integer lesson. See result.json for every output, loss, weight delta and real guard outcome. No claim of language/representation success.\n',encoding='utf-8')
    write_json(run/'result.json',result);write_json(root/'artifacts/live-observatory-latest.json',result)
    print(json.dumps({'run':result['run'],'answers':len(outputs),'learning_events':len(learning),'queued':trace['controller']['queued']},ensure_ascii=False))


if __name__=='__main__':main()
