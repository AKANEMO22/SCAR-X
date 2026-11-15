"""Data-only/native-output effect audit for stopped multi-API qualification.

Actual owned source simulator replay; no new native inference or SNN updates.
"""
import copy,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import sys,uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.tooltalk_sandbox import Sandbox,load_executor
from scarx.hybrid.store import encode


def independent_effect(before,after,row,owner):
    api=row['api'];args=row['correct_request']['parameters']
    reconstructed=copy.deepcopy(after)
    if api=='AddReminder':
        old=before['Reminder'].get(owner,{});new=after['Reminder'].get(owner,{})
        identifiers=set(new)-set(old)
        if len(identifiers)!=1:return False
        ident=next(iter(identifiers));record=new[ident]
        if (record.get('task')!=args['task'] or record.get('due_date')!=args['due_date']
                or record.get('status')!='pending' or record.get('reminder_id')!=ident):return False
        del reconstructed['Reminder'][owner][ident]
    elif api=='CompleteReminder':
        ident=args['reminder_id']
        if reconstructed['Reminder'][owner].get(ident,{}).get('status')!='complete':return False
        reconstructed['Reminder'][owner][ident]['status']=before['Reminder'][owner][ident]['status']
    elif api=='DeleteAlarm':
        ident=args['alarm_id']
        if ident in reconstructed['Alarm'][owner]:return False
        reconstructed['Alarm'][owner][ident]=copy.deepcopy(before['Alarm'][owner][ident])
    else:raise ValueError('Unknown source API')
    return reconstructed==before


def main():
    source=Path((ROOT/'local/current-p1b-multi-api-v2.txt').read_text(encoding='utf-8').strip())
    result=json.loads((source/'result.json').read_text(encoding='utf-8'));data=json.loads((source/'locked-data.json').read_text(encoding='utf-8'))
    assert result['native_calls']==84 and not result['failures'] and result['training_data_qualified'] is False
    assert len(data)==72 and len(result['cases'])==12 and all(r['split']=='DEV' for r in result['cases'])
    intake=json.loads((ROOT/'runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json').read_text(encoding='utf-8'))
    upstream=Path(intake['local_snapshot']);cls=load_executor(upstream,intake);sandbox=Sandbox(cls,upstream/'data/databases')
    owner=json.loads((upstream/'data/easy/AddReminder-easy.json').read_text(encoding='utf-8'))['user']['username']
    rows={r['id']:r for r in data};tool_calls=0;proofs=[]
    for row in data:
        # Reconstruct the actual source prefix, not an insertion-order hash
        # over restored JSON (the snapshot codec deliberately sorts keys).
        reference=json.loads((upstream/'data/easy/AddReminder-easy.json').read_text(encoding='utf-8'))
        sandbox.initialize(reference['metadata'],reference['user'])
        index=int(row['id'].rsplit('-',1)[1]);offset={'TRAIN':0,'DEV':1000,'TEST':2000}[row['split']]
        setup_api='AddAlarm' if row['api']=='DeleteAlarm' else 'AddReminder'
        sandbox.executor.get_init_tool(setup_api).random.seed(700000+offset+index)
        setup={'api_name':setup_api,'parameters':{'time':'02:01:00'} if setup_api=='AddAlarm' else {'task':'owned anchor '+str(index)}}
        setup_result=sandbox.dispatch(setup,origin='independent_source_setup_replay');tool_calls+=1
        assert setup_result['exception'] is None
        before=copy.deepcopy(sandbox.executor.databases)
        assert hashlib.sha256(encode(before).encode()).hexdigest()==row['before_state_sha256']
        error=sandbox.dispatch(copy.deepcopy(row['original_request']),origin='independent_actual_error_replay');tool_calls+=1
        assert error['exception']==row['actual_error']['exception'] and sandbox.executor.databases==before
        reconstructed=copy.deepcopy(sandbox.executor.databases)
        sandbox.restore((source/(row['id']+'-source.snapshot')).read_bytes())
        assert sandbox.executor.databases==reconstructed
        actual=sandbox.dispatch(copy.deepcopy(row['correct_request']),origin='independent_teacher_effect_replay');tool_calls+=1
        assert actual['exception'] is None and independent_effect(before,sandbox.executor.databases,row,owner)
    matches=[]
    for case in result['cases']:
        row=rows[case['id']]
        for action,outcome in case['outcomes'].items():
            sandbox.restore((source/(row['id']+'-source.snapshot')).read_bytes());before=copy.deepcopy(sandbox.executor.databases)
            good=False;actual=None
            try:
                parsed=json.loads(outcome['raw_answer'])
                if action not in ('clarify','defer','finish') and set(parsed)=={'api_name','parameters'} and parsed['api_name']==row['api']:
                    actual=sandbox.dispatch(parsed,origin='independent_saved_multiAPI_reply');tool_calls+=1
                    good=actual['exception'] is None and independent_effect(before,sandbox.executor.databases,row,owner)
            except (ValueError,TypeError,KeyError):pass
            assert bool(good)==outcome['verified_task_effect']
            matches.append({'id':row['id'],'api':row['api'],'action':action,'verified_effect':bool(good)})
    actions=('show_schema','inspect_status','retry','clarify','defer','finish')
    counts={a:sum(r['outcomes'][a]['verified_task_effect'] for r in result['cases']) for a in actions+('RAG_experience',)}
    oracle=sum(any(r['outcomes'][a]['verified_task_effect'] for a in actions) for r in result['cases'])
    union=sum(any(v['verified_task_effect'] for v in r['outcomes'].values()) for r in result['cases'])
    assert oracle==6 and max(counts.values())==6 and union==7
    # Includes the unimplemented retrieval-action proposal; still below the
    # unchanged65% acquisition criterion on DEV. No need to train it to learn this bound.
    report={'verdict':'MULTI_API_EXACT_EFFECTS_AND_USEFULNESS_STOP_INDEPENDENTLY_VERIFIED','original_run':source.name,
        'verified_teacher_corrections':72,'saved_native_replies_replayed':84,'actual_owned_tool_replays':tool_calls,
        'new_model_or_learning_calls':0,'DEV_count':12,'registered_control_counts':counts,
        'six_action_oracle_correct':oracle,'six_action_margin_vs_best_control_pp':0,
        'seven_action_union_including_RAG_correct':union,'seven_action_DEV_ceiling_percent':100*union/12,
        'unchanged_acquisition_minimum_percent':65,'seven_action_expansion_would_still_fail_acquisition_bound':True,
        'TEST_native_outputs_generated':False,'SNN_training_executed':False,'new_retention_measurements':False,
        'no_MORE_BANKS_or_epoch_search_from_this_failed_family':True,'scientific_or_production_PASS':False,
        'before_hash_verified_on_reconstructed_source_order_not_sorted_snapshot_encoding':True,
        'raw_result_SHA256':hashlib.sha256((source/'result.json').read_bytes()).hexdigest(),'rows':matches}
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-multi-api-data-audit-v2-'+uuid.uuid4().hex[:6]);run.mkdir()
    (run/'result.json').write_text(encode(report),encoding='utf-8');(run/'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT/'local/current-multi-api-v2-audit.txt').write_text(str(run),encoding='utf-8')
    print(encode({'run':str(run),**{k:v for k,v in report.items() if k!='rows'}}),flush=True)


if __name__=='__main__':main()
