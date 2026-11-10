"""Actual current/release model inference under controlled prior states; no learning."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import io,json,sys,time,zipfile,hashlib,urllib.request
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.unified import UnifiedCore
from scarx.neural.language import tokenize
from scarx.app.workload import plan_workload
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor


def prepare(model,condition):
    if condition=='canonicalreset':model.core.reset_state();model.last_active[:]=model.ticks
    elif condition=='silent_idle':
        model.select_capacity(48)
        for _ in range(100):model.tick()
    elif condition=='public_idle_probes':
        model.select_capacity(48)
        prompts=('chào bạn','hi','bạn là ai','cảm ơn')
        for index in range(8):
            prompt=prompts[index%4];model.boundary();context=model.context(prompt);previous=1
            for position in range(4):
                token=int(model.word_step(previous,context,position).argmax())
                if token==0:break
                previous=token
    elif condition!='savedstate':raise ValueError('Unknown statecondition')


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    info=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'))
    raw=(root/info['data']).read_bytes();assert hashlib.sha256(raw).hexdigest()==info['sha256']
    cases=json.loads(raw)['items'];corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
    for index,family in enumerate(corpus['families']):
        for j,p in enumerate(family['train']):cases.append({'id':f'known-{index}-{j}','prompt':p,'target':family['answer'],'intent':str(index),'source':'project authored training control','same_as_old_fixture':True})
    run=new_run(root,'g1-conversational-state-audit-v1',{'protocol':'experiments/g1-conversational-state-audit-v1.md',
        'display_title':'G1 hội thoại thật · trạng thái rảnh hay weights gây lỗi?',
        'model_scope':'Disposable release/current snapshots, no training/live state reset; personalweights/rawoutputs private',
        'new_training':False,'cases':len(cases),'states':['savedstate','silent_idle','public_idle_probes','canonicalreset']},
        data_description='Reviewed public Antigravity prompts + authored known controls; no private dialogue sent to teachers',data_files=[info['data']])
    release=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'))
    payloads={'public_release':(root/release['checkpoint']).read_bytes()}
    bundle=urllib.request.urlopen('http://127.0.0.1:8765/api/export',timeout=10).read()
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:payloads['current_private']=archive.read('model.zip')
    private=root/'local/g1-state-audit'/run.name;private.mkdir(parents=True)
    (private/'current-model.zip').write_bytes(payloads['current_private'])
    initial_live=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/status'))['updates']
    output=[];summary={};total=len(cases)*4*2
    with ResourceMonitor(interval=.1) as monitor:
        for scope,payload in payloads.items():
            start=time.perf_counter();sha=hashlib.sha256(payload).hexdigest();scoped=[]
            for condition in ('savedstate','silent_idle','public_idle_probes','canonicalreset'):
                for case in cases:
                    model=UnifiedCore.loads(payload);fingerprint=model.core.fingerprint();updates=model.core.updates
                    prepare(model,condition);load=plan_workload(model,case['prompt']);model.select_capacity(load['requested_neurons'])
                    began_inference=time.perf_counter();tick=model.ticks;response=model.generate(case['prompt'],24)
                    assert model.core.fingerprint()==fingerprint and model.core.updates==updates
                    row={'scope':scope,'state_condition':condition,'case_id':case['id'],'intent':case['intent'],
                        'prompt':case['prompt'],'target':case['target'],'generated':response['text'],
                        'exact_to_one_style_target':tokenize(response['text'])==tokenize(case['target']),
                        'known_or_duplicate':case['same_as_old_fixture'],'source':case['source'],
                        'seconds':time.perf_counter()-began_inference,'neural_ticks':model.ticks-tick,
                        'active':model.active_count,'allocated':model.core.config.hidden,'weights_unchanged':True,
                        'checkpoint_sha256':sha,'qwen':None,'new_training':False}
                    scoped.append(row);output.append(row)
                    append(run,'answer',{**row,'scope':f'{scope} · {condition} · inference clone, no live mutation',
                        'input_scope':'Latest public user text only; actual unchanged UnifiedCore weights. One style target is diagnostic, not a truth label.',
                        'checkpoint':'private snapshot in local/' if scope=='current_private' else release['checkpoint']})
                monitor.check();progress(run,f'{scope} · {condition}',len(output),total,
                    metrics={'strict_style_exact':float(np.mean([r['exact_to_one_style_target'] for r in scoped if r['state_condition']==condition])),
                        'scope':scope,'weights_updated':False},
                    neural={'computed':model.active_count,'allocated':model.core.config.hidden,'tick':model.ticks,'spikes':model.core.spikes.astype(int).tolist(),'voltage':np.round(model.core.voltage,4).tolist()})
                if time.perf_counter()-start>120 or time.perf_counter()-began>300:raise TimeoutError('State audit deadline')
            summary[scope]={}
            for condition in ('savedstate','silent_idle','public_idle_probes','canonicalreset'):
                rs=[r for r in scoped if r['state_condition']==condition]
                reset={r['case_id']:r for r in scoped if r['state_condition']=='canonicalreset'}
                summary[scope][condition]={'strict_style_exact':float(np.mean([r['exact_to_one_style_target'] for r in rs])),
                    'disagreement_vs_reset':float(np.mean([r['generated']!=reset[r['case_id']]['generated'] for r in rs])),
                    'known_control_exact':float(np.mean([r['exact_to_one_style_target'] for r in rs if r['case_id'].startswith('known-')])),
                    'antigravity_exact':float(np.mean([r['exact_to_one_style_target'] for r in rs if not r['case_id'].startswith('known-')]))}
            write_json((private if scope=='current_private' else run)/(scope+'-outputs.json'),scoped)
        result={'run':run.relative_to(root).as_posix(),'data':info['data'],'decision':'G1_STATE_DIAGNOSTIC_ONLY',
            'summary':summary,'cases':len(cases),'inferences':len(output),'weights_unchanged':True,
            'private_model_sha256':hashlib.sha256(payloads['current_private']).hexdigest(),
            'serving_updates_before':initial_live,'serving_updates_after':json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/status'))['updates'],
            'resources':monitor.report(),'cap':cap,'deployed':False,'new_training':False,'normal_chat_or_vision_proven':False,
            'limits':['One reference phrase/known8intent taxonomy, not general language or semantic correctness.','Current/release scopes differ learnedweights and allocation, not matched architecturecontrols.',
                'Private model/rawoutputs remain local only; no private context sent to teacher.']}
        (run/'report.md').write_text('# G1 conversational state diagnostic\n\nActual model-generated outputs, no weights updated, no model substitution. Public cases only.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-conversational-state-latest.json',result)
        print(json.dumps({'run':str(run),'summary':summary,'inferences':len(output),'weights_unchanged':True}))


if __name__=='__main__':main()
