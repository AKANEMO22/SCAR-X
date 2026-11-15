"""Replay separately corrected RAG outputs, no inference and no superiority claim."""
import copy,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import sys,uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.tooltalk_sandbox import Sandbox,load_executor
from scarx.hybrid.store import encode


def main():
    source=Path((ROOT/'local/current-retention-comparator-view.txt').read_text(encoding='utf-8').strip())
    result=json.loads((source/'result.json').read_text(encoding='utf-8'));assert result['completed'] and len(result['predictions'])==40
    original=ROOT/'runs'/result['original_run'];manifest=json.loads((original/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((original/'result.json').read_bytes()).hexdigest()==result['original_result_sha256']
    data={r['id']:r for r in json.loads((original/'locked-data.json').read_text(encoding='utf-8'))}
    prompts={r['id']:r for r in json.loads((source/'locked-corrected-prompts.json').read_text(encoding='utf-8'))}
    intake=json.loads((ROOT/'runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json').read_text(encoding='utf-8'));upstream=Path(intake['local_snapshot'])
    cls=load_executor(upstream,intake);sandbox=Sandbox(cls,upstream/'data/databases')
    multi=ROOT/'runs'/manifest['multi_API_data'];aref=json.loads((upstream/'data/easy/AddAlarm-easy.json').read_text(encoding='utf-8'));mref=json.loads((upstream/'data/easy/AddReminder-easy.json').read_text(encoding='utf-8'))
    records=[];calls=0;copied=0
    for prediction in result['predictions']:
        row=data[prediction['id']];api=row['api'];expected=json.loads(row['target']);source_ids=[r['id'] for r in data.values() if r['split']=='TRAIN']
        if api=='AddAlarm':sandbox.initialize(aref['metadata'],aref['user']);owner=aref['user']['username']
        else:sandbox.restore((multi/(row['source_row']+'-source.snapshot')).read_bytes());owner=mref['user']['username']
        before=copy.deepcopy(sandbox.executor.databases);effect=False;copied_example=False;actual=None
        try:
            request=json.loads(prediction['actual_answer'])
            for candidate in data.values():
                if candidate['split']=='TRAIN' and json.loads(candidate['target'])==request:copied_example=True
            if set(request)=={'api_name','parameters'} and request['api_name']==api:
                actual=sandbox.dispatch(request,origin='independent_corrected_comparator_reply');calls+=1
                after=copy.deepcopy(sandbox.executor.databases);rebuilt=copy.deepcopy(after)
                if actual['exception'] is None:
                    if api in ('AddAlarm','AddReminder'):
                        db='Alarm' if api=='AddAlarm' else 'Reminder';old=before[db].get(owner,{});new=after[db].get(owner,{})
                        added=set(new)-set(old)
                        if len(added)==1:
                            ident=next(iter(added));valid=all(new[ident].get(k)==v for k,v in expected['parameters'].items())
                            del rebuilt[db][owner][ident]
                            if owner not in before[db]:del rebuilt[db][owner]
                            effect=valid and rebuilt==before
                    elif api=='CompleteReminder':
                        ident=expected['parameters']['reminder_id']
                        if after['Reminder'][owner].get(ident,{}).get('status')=='complete':
                            rebuilt['Reminder'][owner][ident]['status']=before['Reminder'][owner][ident]['status'];effect=rebuilt==before
                    else:
                        ident=expected['parameters']['alarm_id']
                        if ident not in after['Alarm'][owner]:rebuilt['Alarm'][owner][ident]=before['Alarm'][owner][ident];effect=rebuilt==before
        except (ValueError,KeyError,TypeError):pass
        assert bool(effect)==prediction['exact_correct'];copied+=copied_example
        records.append({'id':row['id'],'api':api,'effect':bool(effect),'exact_TRAIN_correction_copied':copied_example,'actual_replay_result':actual})
    ledger=[json.loads(l) for l in (source/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(ledger)==42 and sum(r['kind']=='frozen_base_RAG' for r in ledger)==40
    assert not any(r.get('kind') in ('optimizer','training_forward') for r in ledger)
    report={'verdict':'CORRECTED_RAG_PROMPT_REPLAY_VERIFIED_BUT_NOT_A_STRONG_BASELINE_SUPERIORITY_TEST',
        'source_run':source.name,'actual_source_tool_replays':calls,'new_model_or_training_calls':0,
        'correct_effects':sum(r['effect'] for r in records),'count':40,'exact_TRAIN_answers_copied':copied,
        'RAG_examples_after_current_request_is_a_prompt_binding_confound':True,
        'observed_failure':'Base model often copies example values/IDs; compact prompt not established as strongest RAG',
        'LoRA_vs_general_RAG_superiority_claim_allowed':False,
        'RAG_comparator_view_changed_and_measured_separately':True,
        'original_retention_whole_protocol_stays_STOPPED_NOT_PASS':True,
        'total_original_plus_correction_top_level_calls':1344+42,'no_additional_training':True,
        'final_adapter_artifact_reload_exact_in_correction':result['adapter_artifact_reload_logits_exact'],
        'maximum_prompt_plus_target_tokens':result['maximum_prompt_plus_target_tokens'],
        'population_retention_or_production_PASS':False,'records':records}
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-retention-comparator-audit-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    (run/'result.json').write_text(encode(report),encoding='utf-8');(run/'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT/'local/current-retention-comparator-audit.txt').write_text(str(run),encoding='utf-8')
    print(encode({'run':str(run),**{k:v for k,v in report.items() if k!='records'}}),flush=True)


if __name__=='__main__':main()
