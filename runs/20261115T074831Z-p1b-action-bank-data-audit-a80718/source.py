"""Independent source/executor replay for held-out actual model repair outputs.

No model calls/learning. Replays only owned AddAlarm simulator states; checks
expected task time and no other effect without trusting producer success flags.
"""
from datetime import datetime,timezone
import copy,hashlib,json
from pathlib import Path
import sys,time,uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.tooltalk_sandbox import Sandbox,load_executor
from scarx.hybrid.store import encode


def main():
    table=Path((ROOT/'local/current-p1b-training-outcomes.txt').read_text(encoding='utf-8').strip())
    parent=Path((ROOT/'local/current-p1b-action-usefulness.txt').read_text(encoding='utf-8').strip())
    data_dir=Path((ROOT/'local/current-qualified-repair-data.txt').read_text(encoding='utf-8').strip())
    data={r['id']:r for r in json.loads((data_dir/'verified-data.json').read_text(encoding='utf-8'))}
    intake=json.loads((ROOT/'runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json').read_text(encoding='utf-8'))
    upstream=Path(intake['local_snapshot']);cls=load_executor(upstream,intake);sandbox=Sandbox(cls,upstream/'data/databases')
    reference=json.loads((upstream/'data/easy/AddAlarm-easy.json').read_text(encoding='utf-8'));owner=reference['user']['username']
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-p1b-action-bank-data-audit-'+uuid.uuid4().hex[:6]);run.mkdir()
    report={'source_tables':[parent.name,table.name],'rows':[],'mismatches':[],'actual_owned_tool_replays':0,'new_model_or_learning_calls':0}
    per_split={}
    for source in (parent,table):
        original=json.loads((source/'result.json').read_text(encoding='utf-8'))
        for case in original['cases']:
            row=data[case['id']]
            for action,record in case['outcomes'].items():
                sandbox.initialize(reference['metadata'],reference['user']);before=copy.deepcopy(sandbox.executor.databases)
                initial=sandbox.dispatch(copy.deepcopy(row['original_actual_result']['request']),origin='independent_saved_error_replay')
                report['actual_owned_tool_replays']+=1;assert initial['exception'] is not None
                expected=False;actual=None
                try:
                    request=json.loads(record['raw_model_answer'])
                    if action not in ('clarify','defer','finish') and set(request)=={'api_name','parameters'} and request['api_name']=='AddAlarm':
                        actual=sandbox.dispatch(request,origin='independent_saved_model_output_replay');report['actual_owned_tool_replays']+=1
                        after=copy.deepcopy(sandbox.executor.databases)
                        old=before.get('Alarm',{}).get(owner,{});new=after.get('Alarm',{}).get(owner,{})
                        added=set(new)-set(old)
                        if len(added)==1 and actual['exception'] is None:
                            identifier=next(iter(added));alarm=new[identifier]
                            reconstructed=copy.deepcopy(after);del reconstructed['Alarm'][owner][identifier]
                            if owner not in before.get('Alarm',{}):del reconstructed['Alarm'][owner]
                            expected=(alarm.get('time')==row['requested_time'] and alarm.get('alarm_id')==identifier and reconstructed==before)
                except (ValueError,KeyError,TypeError):pass
                if bool(expected)!=record['verified_task_effect']:report['mismatches'].append({'id':row['id'],'action':action,'recalculated':bool(expected),'producer':record['verified_task_effect']})
                report['rows'].append({'id':row['id'],'split':row['split'],'action':action,'effect':bool(expected),
                    'actual_replay_result':actual,'raw_model_answer_sha256':hashlib.sha256(record['raw_model_answer'].encode()).hexdigest()})
    assert len(report['rows'])==392 and not report['mismatches']
    for split in ('TRAIN','DEV','TEST'):
        ids=[r['id'] for r in data.values() if r['split']==split];success={r['id']+':'+r['action']:r['effect'] for r in report['rows']}
        rule=sum(success[i+':show_schema'] for i in ids);rag=sum(success[i+':RAG_experience'] for i in ids)
        oracle=sum(any(success[i+':'+a] for a in ('show_schema','inspect_status','retry','clarify','defer','finish')) for i in ids)
        per_split[split]={'N':len(ids),'fixed_rule_correct':rule,'RAG_correct':rag,'finite_action_oracle_correct':oracle,
            'maximum_gain_vs_strongest_pp':100*(oracle-max(rule,rag))/len(ids)}
    report.update(verdict='ACTUAL_REPAIR_EFFECT_REPLAY_MATCHES_ACTION_BANK_STOP_GATE',by_split=per_split,
        P1B_advantage_gate_possible_on_frozen_action_bank=False,SNN_training_or_retention_executed=False,
        raw_state_replay_is_not_new_model_quality_evidence=True)
    (run/'result.json').write_text(encode(report),encoding='utf-8');(run/'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT/'local/current-p1b-action-bank-audit.txt').write_text(str(run),encoding='utf-8')
    print(encode({'run':str(run),**{k:v for k,v in report.items() if k!='rows'}}),flush=True)


if __name__=='__main__':main()
