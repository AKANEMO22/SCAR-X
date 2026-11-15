"""Actual native0.8B+ToolTalk counterfactual action table; stop at ceiling.

DEV only, no SNN updates. Whole held-out TEST remains outside this qualifier.
"""
from datetime import datetime,timezone
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.tooltalk_sandbox import Sandbox,load_executor,verify_single_alarm_transition
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.store import Store,encode

ACTIONS=('show_schema','inspect_status','retry','clarify','defer','finish')


def main():
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-p1b-action-usefulness-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    with (ROOT/'local/p1b-action-usefulness-v1-execution.json').open('x',encoding='utf-8') as f:json.dump({'run':str(run)},f)
    data_dir=Path((ROOT/'local/current-qualified-repair-data.txt').read_text(encoding='utf-8').strip())
    allrows=json.loads((data_dir/'verified-data.json').read_text(encoding='utf-8'))
    dev=[r for r in allrows if r['split']=='DEV'];train=[r for r in allrows if r['split']=='TRAIN']
    assert len(dev)==8 and len(train)==32
    sources=[Path(__file__),ROOT/'experiments/hybrid-p1b-error-learning-v1.md',ROOT/'src/scarx/evaluation/tooltalk_sandbox.py',ROOT/'src/scarx/evaluation/shared_native.py']
    hashes={}
    for p in sources:
        rel=p.relative_to(ROOT);target=run/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes());hashes[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (run/'manifest.json').write_text(encode({'source_git_sha':__import__('os').environ.get('SCARX_P1B_SOURCE_SHA'),
        'source_sha256':hashes,'data_run':data_dir.name,'data_sha256':hashlib.sha256((data_dir/'verified-data.json').read_bytes()).hexdigest(),
        'split':'DEV only','DEV_ids':[r['id'] for r in dev],'actions':ACTIONS,'max_native_calls':56,'max_seconds':900,
        'baseline_rule_before_outputs':'if actual error then show_schema else finish',
        'oracle_control_margin_stop_threshold_pp':5,'no_training_before_gate':True}),encoding='utf-8')
    native=SharedNative(ROOT);trace=Store(ROOT/'local/hybrid-v1');started=time.time();calls=0
    report={'run_id':run.name,'cases':[],'failures':[],'neural_updates':0,'dataset_qualified_for_training':False,
            'retention_measured':False,'scientific_or_production_PASS':False}
    try:
        intake=json.loads((ROOT/'runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json').read_text(encoding='utf-8'))
        upstream=Path(intake['local_snapshot']);cls=load_executor(upstream,intake);sandbox=Sandbox(cls,upstream/'data/databases')
        reference=json.loads((upstream/'data/easy/AddAlarm-easy.json').read_text(encoding='utf-8'));owner=reference['user']['username']
        for row in dev:
            sandbox.initialize(reference['metadata'],reference['user']);before=copy.deepcopy(sandbox.executor.databases)
            initial=sandbox.dispatch(copy.deepcopy(row['original_actual_result']['request']),origin='P1B_actual_fault')
            assert initial['exception'];snapshot=sandbox.snapshot()
            record={'id':row['id'],'user_request':row['user_request'],'initial_actual_error':initial['exception'],'outcomes':{}}
            for action in ACTIONS+('RAG_experience',):
                if time.time()-started>900 or calls>=56:raise TimeoutError('Qualifier native/time budget')
                sandbox.restore(snapshot)
                messages=copy.deepcopy(row['messages'])
                if action in ACTIONS:
                    macro=sandbox.recover(action,origin='counterfactual action-table measurement')
                    macro_public=copy.deepcopy(macro)
                    if isinstance(macro_public.get('result'),dict):macro_public['result'].get('request',{}).get('parameters',{}).pop('session_token',None)
                    messages.append({'role':'user','content':'Selected recovery observation: '+encode(macro_public)+
                        ('\nThis recovery is terminal. Return {"status":"deferred"} without claiming a successful tool repair.' if action in ('clarify','defer','finish') else '\nReturn the corrected AddAlarm JSON only.')})
                else:
                    # Same actual error evidence; fixed deterministic lexical retrieval
                    # from TRAIN only. No DEV answer or private future outcome is used.
                    words=set(row['original_actual_result']['exception'].lower().split())
                    neighbors=sorted(train,key=lambda r:-len(words&set(r['original_actual_result']['exception'].lower().split())))[:2]
                    messages.append({'role':'user','content':'Previous verified TRAIN experiences: '+encode([
                        {'error':r['original_actual_result']['exception'],'user_request':r['user_request'],'successful_repair':r['target']} for r in neighbors])+'. Repair the current request, not the example time.'})
                calls+=1
                with (run/'attempts.jsonl').open('a',encoding='utf-8') as f:f.write(encode({'attempt':calls,'case':row['id'],'action':action,'epoch':time.time()})+'\n')
                output=native.generate({'messages':messages,'max_tokens':96})
                text=output['choices'][0]['message'].get('content') or ''
                actual=None;valid=False
                try:
                    parsed=json.loads(text)
                    if action not in ('clarify','defer','finish') and set(parsed)=={'api_name','parameters'} and parsed['api_name']=='AddAlarm':
                        actual=sandbox.dispatch(parsed,origin='actual_frozen_VLM_repair')
                        proof=verify_single_alarm_transition(before,sandbox.executor.databases,owner,row['requested_time'])
                        valid=proof['goal_verified'] and actual['exception'] is None
                except (ValueError,TypeError,KeyError):pass
                outcome={'raw_model_answer':text,'actual_tool_execution':actual,'verified_task_effect':bool(valid),
                         'native_input_sha256':hashlib.sha256(encode(messages).encode()).hexdigest()}
                record['outcomes'][action]=outcome
                trace.event('work_trace.activity',{'kind':'actual_P1B_action_qualification','title':'P1B DEV '+row['id']+' '+action,
                    'run_id':run.name,'question':row['user_request'],'actual_answer':text,
                    'teacher_expected':row['target'],'actual_repair_effect':bool(valid),'neural_updates':0},'orchestrator')
            report['cases'].append(record)
        rule=sum(r['outcomes']['show_schema']['verified_task_effect'] for r in report['cases'])/8
        rag=sum(r['outcomes']['RAG_experience']['verified_task_effect'] for r in report['cases'])/8
        oracle=sum(any(r['outcomes'][a]['verified_task_effect'] for a in ACTIONS) for r in report['cases'])/8
        report.update(verdict='ACTION_USEFULNESS_QUALIFIED' if oracle-max(rule,rag)>=.05 and max(rule,rag)<1 else 'STOP_INSUFFICIENT_ORACLE_CONTROL_MARGIN_NO_SNN_TRAINING',
            fixed_rule_success=rule,RAG_verified_experience_success=rag,finite_action_oracle=oracle,
            oracle_minus_strongest_control_pp=100*(oracle-max(rule,rag)),
            dataset_qualified_for_training=oracle-max(rule,rag)>=.05 and max(rule,rag)<1,
            native_calls=calls,elapsed_seconds=time.time()-started,TEST_used=False)
    except Exception as exc:
        report['failures'].append(type(exc).__name__+': '+str(exc));report['verdict']='QUALIFIER_STOPPED_NOT_PASS'
    finally:
        (run/'result.json').write_text(encode(report),encoding='utf-8')
        (ROOT/'local/current-p1b-action-usefulness.txt').write_text(str(run),encoding='utf-8')
        trace.event('work_trace.activity',{'title':'P1B usefulness terminal','run_id':run.name,**{k:v for k,v in report.items() if k!='cases'}},'orchestrator');trace.close()
    print(encode({'run':str(run),**{k:v for k,v in report.items() if k!='cases'}}),flush=True)


if __name__=='__main__':main()
