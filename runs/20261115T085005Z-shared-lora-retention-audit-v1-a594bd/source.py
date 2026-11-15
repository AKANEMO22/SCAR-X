"""Independent20-stage metric/label/trace audit and saved-effect replay, no model."""
import argparse,copy,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import random,sys,uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.tooltalk_sandbox import Sandbox,load_executor
from scarx.hybrid.store import encode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic-comparator-stop',action='store_true',help='Rescore a complete20stage trajectory from a run stopped before the RAG comparator; never certify the whole protocol')
    args=parser.parse_args()
    original=Path((ROOT/'local/current-shared-lora-retention20.txt').read_text(encoding='utf-8').strip())
    result=json.loads((original/'result.json').read_text(encoding='utf-8'));manifest=json.loads((original/'manifest.json').read_text(encoding='utf-8'))
    data=json.loads((original/'locked-data.json').read_text(encoding='utf-8'));by_id={r['id']:r for r in data}
    assert hashlib.sha256((original/'locked-data.json').read_bytes()).hexdigest()==manifest['data_sha256']
    diagnostic=args.diagnostic_comparator_stop
    if diagnostic:
        assert result['completed'] is False and result['verdict']=='SHARED_ADAPTER_RETENTION20_STOPPED_NOT_PASS'
        assert result['failures']==['ValueError: No truncation; token bound512']
    else:assert result['completed']
    assert len(result['stages'])==20 and len(result['updates'])==480
    train_ids={r['id'] for r in data if r['split']=='TRAIN'};test_ids={r['id'] for r in data if r['split']=='TEST'}
    assert len(train_ids)==68 and len(test_ids)==40 and not(train_ids&test_ids)
    assert all(r['row'] in train_ids and np.isfinite(r['loss']) for r in result['updates'])
    assert len([r for r in result['updates'] if r['replay']])==240
    assert [r['step'] for r in result['updates']]==list(range(1,481))
    scores=[];api_scores=[]
    for stage in [result['baseline']]+result['stages']:
        assert len(stage['predictions'])==40 and {p['id'] for p in stage['predictions']}==test_ids
        counted={}
        for api in ('AddAlarm','AddReminder','CompleteReminder','DeleteAlarm'):
            matches=[]
            for prediction in stage['predictions']:
                if prediction['api']!=api:continue
                row=by_id[prediction['id']]
                try:correct=json.loads(prediction['actual_answer'])==json.loads(row['target'])
                except ValueError:correct=False
                assert correct==prediction['exact_correct']
                matches.append(correct)
            counted[api]=sum(matches)/len(matches)
            assert counted[api]==stage['by_API'][api]
        accuracy=sum(p['exact_correct'] for p in stage['predictions'])/40
        assert accuracy==stage['accuracy'];scores.append(accuracy);api_scores.append(counted)
    forgetting={api:100*(max(s[api] for s in api_scores)-api_scores[-1][api]) for api in api_scores[-1]}
    if not diagnostic:assert forgetting==result['forgetting_pp']
    mean=sum(forgetting.values())/4;worst=max(forgetting.values());acquisition=max(scores[1:])
    if not diagnostic:assert abs(mean-result['mean_forgetting_pp'])<1e-12 and worst==result['worst_API_forgetting_pp']
    joint=acquisition>=.65 and mean<=3 and worst<=5
    if not diagnostic:assert joint==result['joint_acquisition_retention_gate']
    # Independent replay for final and frozen-base-RAG outputs against exact source state.
    intake=json.loads((ROOT/'runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json').read_text(encoding='utf-8'))
    upstream=Path(intake['local_snapshot']);cls=load_executor(upstream,intake);sandbox=Sandbox(cls,upstream/'data/databases')
    multi=ROOT/'runs'/manifest['multi_API_data'];multirows={r['id']:r for r in json.loads((multi/'locked-data.json').read_text(encoding='utf-8'))}
    alarmref=json.loads((upstream/'data/easy/AddAlarm-easy.json').read_text(encoding='utf-8'))
    reminderref=json.loads((upstream/'data/easy/AddReminder-easy.json').read_text(encoding='utf-8'))
    replay=[];tool_calls=0
    groups=[('final_adapter',result['stages'][-1])]
    if not diagnostic:groups.append(('frozen_base_RAG',result['frozen_base_RAG_final']))
    for label,group in groups:
        for prediction in group['predictions']:
            row=by_id[prediction['id']];api=row['api'];expected=json.loads(row['target']);owner=None
            if api=='AddAlarm':
                sandbox.initialize(alarmref['metadata'],alarmref['user']);owner=alarmref['user']['username']
            else:
                sandbox.restore((multi/(row['source_row']+'-source.snapshot')).read_bytes());owner=reminderref['user']['username']
            before=copy.deepcopy(sandbox.executor.databases);effect=False
            try:
                request=json.loads(prediction['actual_answer'])
                if set(request)=={'api_name','parameters'} and request['api_name']==api:
                    actual=sandbox.dispatch(request,origin='independent_saved_retention_prediction');tool_calls+=1
                    after=copy.deepcopy(sandbox.executor.databases);valid=actual['exception'] is None
                    reconstructed=copy.deepcopy(after)
                    if api in ('AddAlarm','AddReminder'):
                        db='Alarm' if api=='AddAlarm' else 'Reminder';old=before[db].get(owner,{});new=after[db].get(owner,{})
                        added=set(new)-set(old)
                        if valid and len(added)==1:
                            ident=next(iter(added));record=new[ident]
                            valid=all(record.get(k)==v for k,v in expected['parameters'].items())
                            del reconstructed[db][owner][ident]
                            if owner not in before[db]:del reconstructed[db][owner]
                            effect=valid and reconstructed==before
                    elif api=='CompleteReminder':
                        ident=expected['parameters']['reminder_id']
                        if valid and after['Reminder'][owner].get(ident,{}).get('status')=='complete':
                            reconstructed['Reminder'][owner][ident]['status']=before['Reminder'][owner][ident]['status']
                            effect=reconstructed==before
                    elif api=='DeleteAlarm':
                        ident=expected['parameters']['alarm_id']
                        if valid and ident not in after['Alarm'][owner]:
                            reconstructed['Alarm'][owner][ident]=copy.deepcopy(before['Alarm'][owner][ident]);effect=reconstructed==before
            except (ValueError,KeyError,TypeError):pass
            assert bool(effect)==prediction['exact_correct']
            replay.append({'mode':label,'id':row['id'],'effect':bool(effect),'api':api})
    final_by={r['id']:r['effect'] for r in replay if r['mode']=='final_adapter'}
    rag_by={r['id']:r['effect'] for r in replay if r['mode']=='frozen_base_RAG'}
    confidence=None
    if not diagnostic:
        api_deltas=[]
        for api in api_scores[-1]:
            ids=[i for i in test_ids if by_id[i]['api']==api];api_deltas.append(np.mean([int(final_by[i])-int(rag_by[i]) for i in ids]))
        rng=np.random.default_rng(260118);a=np.array(api_deltas);boot=a[rng.integers(0,4,size=(10000,4))].mean(axis=1)
        confidence=(100*np.quantile(boot,[.025,.975])).tolist()
    ledger=[json.loads(line) for line in (original/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(r['kind']=='optimizer' for r in ledger)==480
    assert all(r['id'] in train_ids for r in ledger if r['kind'] in ('optimizer','training_forward'))
    artifact=ROOT/(result.get('final_adapter') or result['last_saved_adapter']);checkpoints=sorted(artifact.parent.glob('stage-*'));assert len(checkpoints)==20
    for path in checkpoints:
        state=json.loads((path/'controller-state.json').read_text(encoding='utf-8'))
        assert state['TRAIN_only'] and len(state['replay_ids'])<=64 and set(state['replay_ids'])<=train_ids
        assert hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()==state['adapter_sha256']
        def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,list) else value
        random.Random().setstate(tuples(state['python_rng']))
    control={phase:{f:sum(r['exact_correct'] for r in result['old_general_'+phase] if r['family']==f) for f in ('math','JSON','source')} for phase in ('baseline','final')}
    report={'verdict':('RETENTION20_DATA_DIAGNOSTIC_FROM_COMPARATOR_STOP_NOT_WHOLE_PROTOCOL_PASS' if diagnostic
                       else 'INDEPENDENT_SHARED_ADAPTER_RETENTION20_METRICS_AND_EFFECT_REPLAY_MATCH'),
        'diagnostic_only':diagnostic,'original_whole_protocol_completed':result['completed'],
        'original_run':original.name,'optimizer_steps':480,'stages':20,'unique_TRAIN_rows':68,'known_TEST_rows':40,
        'baseline_by_API':api_scores[0],'final_by_API':api_scores[-1],'forgetting_pp':forgetting,
        'mean_forgetting_pp':mean,'worst_API_forgetting_pp':worst,'acquisition':acquisition,'joint_acquisition_retention_gate':joint,
        'final_adapter_effect_accuracy':sum(final_by.values())/40,
        'frozen_base_RAG_effect_accuracy':sum(rag_by.values())/40 if rag_by else None,
        'paired_API_cluster_gain_CI95_pp':confidence,
        'error_API_clusters':4,'single_seed_nonfresh_TEST_scope':True,'old_general_controls':control,
        'population_oldskill_noninferiority_or_lifelong_retention_proven':False,
        'source_effect_replays':tool_calls,'new_model_or_learning_calls':0,'checkpoint_replay_ids_and_python_rng_validated':True,
        'ALIF_learning_or_SNN_advantage':False,'production_qualified':False,'curves':scores,
        'original_result_sha256':hashlib.sha256((original/'result.json').read_bytes()).hexdigest(),'replayed':replay}
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-shared-lora-retention-audit-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    (run/'result.json').write_text(encode(report),encoding='utf-8');(run/'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT/'local/current-shared-lora-retention-audit.txt').write_text(str(run),encoding='utf-8')
    print(encode({'run':str(run),**{k:v for k,v in report.items() if k!='replayed'}}),flush=True)


if __name__=='__main__':main()
