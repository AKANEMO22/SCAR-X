"""Independent data/state rescore: no checker import, tool replay or model calls."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]


def serial(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value):return hashlib.sha256(serial(value)).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def audit(run):
    run=run.resolve()
    if run.parent!=ROOT/'runs' or '-outcome-contract-p0-v1-' not in run.name:raise ValueError('Not an owned P0 run')
    manifest=read(run/'manifest.json');result=read(run/'result.json');locked=read(run/'locked-cases.json')
    assert hashlib.sha256((run/'locked-cases.json').read_bytes()).hexdigest()==manifest['locked_cases_sha256']
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==digest
    assert len(locked)==14 and len({r['name'][:2] for r in locked})==12
    assert len(result['cases'])==14 and result['gate_pass']
    comparisons=[];total_calls=0;ordered={}
    permitted_result={'ok','error','source','text','progress','status','prepared','changed_rows','pending_field','sent_to_real_person'}
    for definition in locked:
        name=definition['name'];directory=run/'fixtures'/name
        trace=read(directory/'trace.json');contract=read(directory/'contract.json');outcome=read(directory/'outcome.json')
        snapshots={p.stem:read(p) for p in (directory/'snapshots').glob('*.json')}
        assert all(sha(s)==key for key,s in snapshots.items())
        assert contract['kind']==definition['kind'] and contract['required_order']==definition['required_order']
        assert trace['task_id']==contract['task_id'] and trace['scope']==contract['scope']=='owned:'+name
        before=snapshots[trace['initial_snapshot']];after=snapshots[trace['final_snapshot']]
        previous=trace['initial_snapshot'];last=0.;prefix=[];signatures=[]
        assert len(trace['events'])==len(definition['actions'])
        for index,(event,(tool,args)) in enumerate(zip(trace['events'],definition['actions'])):
            assert event['tool']==event['proposed_action']==tool and event['arguments']==args
            assert event['selector']=='fixed_fixture' and event['executed_action']==('deny' if tool=='forbidden_update' else tool)
            assert event['index']==index and event['task_id']==trace['task_id'] and event['scope']==trace['scope']
            a,b,c=(event[k] for k in ('decision_cutoff','started_epoch','returned_epoch'))
            assert all(math.isfinite(x) for x in (a,b,c,event['duration_ms'])) and last<=a<=b<=c
            assert event['duration_ms']>=0 and sha(event['result'])==event['result_sha256']
            assert event['before_snapshot']==previous
            assert event['after_snapshot'] in snapshots
            assert event['observations_before_decision']==prefix
            view_text=serial(prefix).decode();assert name not in view_text and trace['task_id'] not in view_text
            prefix.append({'tool':tool,'arguments':args,
                           'result':{k:v for k,v in event['result'].items() if k in permitted_result},
                           'duration_ms':event['duration_ms']})
            unchanged=event['before_snapshot']==event['after_snapshot']
            signatures.append((serial([tool,args,event['result']]),unchanged))
            previous=event['after_snapshot'];last=c
        assert previous==trace['final_snapshot']
        # Check final snapshots against the retained real workspace and DB,
        # instead of trusting producer outcome/status or a 'verified' flag.
        project=directory/'project'
        actual_files={p.name:{'text':p.read_bytes().decode(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in project.glob('*.txt')}
        assert actual_files==after['files']
        with sqlite3.connect((project/'state.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
            records=[dict(zip(('id','value','status'),r)) for r in db.execute('SELECT id,value,status FROM records ORDER BY id')]
            assert records==after['records']
            assert db.execute('SELECT done FROM preparation').fetchone()[0]==after['prepared']
            assert dict(zip(('progress','status'),db.execute('SELECT progress,status FROM job').fetchone()))==after['job']
            assert {'alarm_time':db.execute('SELECT alarm_time FROM request').fetchone()[0]}==after['request']
            assert [r[1] for r in db.execute('PRAGMA table_info(request)')]==after['request_schema']
            assert [r[0] for r in db.execute('SELECT field FROM clarifications ORDER BY rowid')]==after['clarifications']
        effect=False;clarification=False
        if name[:2] in ('01','02','09','11'):
            for event in trace['events']:
                state=snapshots[event['before_snapshot']]
                if event['tool']=='read_file' and event['arguments']=={'path':'requested.txt'} and event['result'].get('ok') is True:
                    actual=state['files']['requested.txt']
                    assert event['result']['text']==actual['text'] and hashlib.sha256(actual['text'].encode()).hexdigest()==actual['sha256']
                    effect=actual['sha256']==contract['value']
            allowed=set()
        elif name[:2] in ('03','04','05','06','07','08'):
            old={r['id']:r for r in before['records']};new={r['id']:r for r in after['records']}
            effect=new.get('target')=={'id':'target','value':'new','status':'committed'} and new.get('other')==old['other'] and set(new)==set(old)
            allowed={'records','prepared'} if name.startswith('07') else {'records'}
        elif name.startswith('10'):
            effect=after['job']=={'progress':3,'status':'completed'};allowed={'job'}
        elif name.startswith('12'):
            clarification=(before['request']['alarm_time'] is None and 'alarm_time' in before['request_schema']
                           and after['clarifications']==['alarm_time']
                           and all(e['tool']=='ask_field' and e['arguments']=={'field':'alarm_time'} for e in trace['events']))
            allowed={'clarifications'}
        else:raise AssertionError('Unexpected fixture name')
        groups=[(key,len(list(values))) for key,values in itertools.groupby(signatures)]
        longest=max([length for (signature,stable),length in groups if stable]+[0])
        redundant=sum(length-1 for (signature,stable),length in groups if stable)
        terminal=bool(groups and groups[-1][0][1] and groups[-1][1]>=3)
        forbidden=sum(e['proposed_action']=='forbidden_update' for e in trace['events'])
        order=True
        if name.startswith('07'):
            tools=[e['tool'] for e in trace['events']];order=tools.index('prepare')<tools.index('commit')
            ordered[name]=(after,sorted(tools))
        extra=[k for k in before if before[k]!=after[k] and k not in allowed]
        success=effect and order and forbidden==0 and longest<3 and not extra
        status='completed' if success else 'constraint_violation' if effect else 'clarification_supported' if clarification else 'unverified'
        recalculated={'status':status,'task_effect_satisfied':effect,'dependency_order_satisfied':order,
            'forbidden_attempts':forbidden,'had_nonprogress_loop':longest>=3,
            'repeated_nonprogress_calls':redundant,'longest_nonprogress_streak':longest,
            'terminal_stuck':terminal,'clarification_supported':clarification,'unexpected_state_changes':extra,
            'whole_task_completion':bool(success),'tool_calls':len(trace['events']),
            'tool_duration_ms':sum(e['duration_ms'] for e in trace['events']),
            'selected_by':'fixed_fixture','neural_credit_eligible':False,'learning_authorized':False,'factual_root_cause_verified':False}
        assert outcome==recalculated
        assert all(outcome.get(k)==v for k,v in definition['expected'].items())
        assert next(r['actual'] for r in result['cases'] if r['name']==name)==outcome
        total_calls+=len(trace['events']);comparisons.append({'name':name,'matches_independent_state_rescore':True})
    assert ordered['07-order-reference']==ordered['07-order-reversed']
    live=read(run/'live-state-audit.json');assert live['equal'] and live['before']==live['after']==manifest['live_before']
    assert result['neural_forward_or_learning_calls']==result['native_VLM_ASR_calls']==0
    assert result['elapsed_seconds']<=manifest['max_seconds'] and result['peak_process_RSS_bytes']<=manifest['max_RSS_bytes']
    assert not result['blocked_attempts'] and not result['neural_packages_loaded']
    assert total_calls==result['tool_calls']
    assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<=manifest['max_artifact_bytes']
    return {'verdict':'INDEPENDENT_EXACT_STATE_ORDER_SOURCE_COST_RESCORE_MATCHES_12_FIXTURES',
        'fixtures':12,'traces':14,'actual_owned_tool_calls':total_calls,'comparisons':comparisons,
        'order_pair_has_identical_final_state_and_bag_of_actions':True,
        'live_saved_neural_state_RNG_replay_equal':True,'new_tool_or_model_calls_in_audit':0,
        'same_orchestrator_separate_implementation_not_external_peer_approval':True,
        'scope':'Engineering contract fixtures; not a neural accuracy/generalization/retention result',
        'research_or_production_PASS':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=audit(args.run)
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))
