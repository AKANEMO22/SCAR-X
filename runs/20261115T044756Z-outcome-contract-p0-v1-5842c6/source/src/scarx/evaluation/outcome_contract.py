"""Research-only exact state/effect contracts. No neural imports or execution.

Inspired by state/milestone evaluation; not copied from ToolSandbox. No soft
text similarity, model judgement, asserted 'verified' flag or learning labels.
"""
import hashlib
import json
import math

FORMAT='scarx-owned-outcome-contract-v1'


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def observed_prefix(events,cutoff):
    if type(cutoff) not in (int,float) or not math.isfinite(cutoff):
        raise ValueError('Invalid decision cutoff')
    fields=('ok','error','source','text','progress','status','prepared','changed_rows',
            'pending_field','sent_to_real_person')
    # Raw OS errors can contain the owned fixture directory/case name. Keep
    # them in the audit trace, not as an identity shortcut in a policy view.
    return [{'tool':e['tool'],'arguments':e['arguments'],
             'result':{k:e['result'][k] for k in fields if k in e['result']},
             'duration_ms':e['duration_ms']} for e in events if e['returned_epoch']<=cutoff]


def validate(trace,snapshots,contract):
    fields={'format','task_id','scope','kind','source','value','required_order','missing_field'}
    if (not isinstance(contract,dict) or set(contract)!=fields or contract['format']!=FORMAT
            or not isinstance(contract['required_order'],list)
            or len(set(contract['required_order']))!=len(contract['required_order'])):
        raise ValueError('Invalid contract identity/schema')
    if trace['format']!=FORMAT or trace['task_id']!=contract['task_id'] or trace['scope']!=contract['scope']:
        raise ValueError('Task/scope binding mismatch')
    for key,value in snapshots.items():
        if key!=digest(value):raise ValueError('Snapshot digest mismatch')
        for entry in value['files'].values():
            if hashlib.sha256(entry['text'].encode('utf-8')).hexdigest()!=entry['sha256']:
                raise ValueError('File content/digest mismatch')
    prior=trace['initial_snapshot'];last_time=0.0
    for i,event in enumerate(trace['events']):
        if event['index']!=i or event['scope']!=contract['scope'] or event['task_id']!=contract['task_id']:
            raise ValueError('Event order or binding mismatch')
        times=[event[k] for k in ('decision_cutoff','started_epoch','returned_epoch')]
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in times):
            raise ValueError('Invalid event clock')
        if not last_time<=times[0]<=times[1]<=times[2]:raise ValueError('Future/backwards event')
        if type(event['duration_ms']) not in (int,float) or event['duration_ms']<0 or not math.isfinite(event['duration_ms']):
            raise ValueError('Invalid cost')
        if event['before_snapshot']!=prior or event['before_snapshot'] not in snapshots or event['after_snapshot'] not in snapshots:
            raise ValueError('Broken snapshot chain')
        if event['result_sha256']!=digest(event['result']):raise ValueError('Result digest mismatch')
        if event['selector']!='fixed_fixture' or not event['decision_id']:
            raise ValueError('P0 never selects actions with a neural model')
        if event['observations_before_decision']!=observed_prefix(trace['events'][:i],times[0]):
            raise ValueError('Decision contains future/foreign/extra observation fields')
        prior=event['after_snapshot'];last_time=times[-1]
    if prior!=trace['final_snapshot'] or prior not in snapshots:
        raise ValueError('Missing final snapshot')
    if len({e['decision_id'] for e in trace['events']})!=len(trace['events']):
        raise ValueError('Duplicate decision identity')


def evaluate(trace,snapshots,contract):
    validate(trace,snapshots,contract)
    events=trace['events'];initial=snapshots[trace['initial_snapshot']];final=snapshots[trace['final_snapshot']]
    effect=False;order=True;clarification=False
    if contract['kind']=='read':
        for event in events:
            before=snapshots[event['before_snapshot']]
            source=contract['source'];entry=before['files'].get(source)
            result=event['result']
            if (event['tool']=='read_file' and event['arguments']=={'path':source}
                    and entry is not None and result.get('ok') is True
                    and result.get('source')==source and result.get('text')==entry['text']
                    and entry['sha256']==contract['value']):effect=True
    elif contract['kind']=='update':
        before={r['id']:r for r in initial['records']};after={r['id']:r for r in final['records']}
        target=contract['source']
        effect=(after.get(target)=={'id':target,'value':contract['value'],'status':'committed'}
                and set(after)==set(before)
                and all(after[k]==v for k,v in before.items() if k!=target))
    elif contract['kind']=='job':
        effect=final['job']=={'progress':3,'status':'completed'}
    elif contract['kind']=='clarify':
        missing=contract['missing_field']
        if missing in initial['request_schema'] and initial['request'].get(missing) is None:
            asks=[e for e in events if e['tool']=='ask_field']
            clarification=bool(asks) and all(e['arguments']=={'field':missing}
                and e['result'].get('ok') is True for e in asks)
            clarification=clarification and final['clarifications']==[missing]
    else:raise ValueError('Unknown contract')
    required=contract['required_order']
    if required:
        tools=[e['tool'] for e in events]
        order=all(tool in tools for tool in required)
        if order:order=[tools.index(t) for t in required]==sorted(tools.index(t) for t in required)
    forbidden=sum(e['proposed_action']=='forbidden_update' for e in events)
    streak=0;previous=None;repeat_calls=0;longest=0
    for event in events:
        signature=digest([event['tool'],event['arguments'],event['result']])
        unchanged=event['before_snapshot']==event['after_snapshot']
        streak=streak+1 if unchanged and signature==previous else (1 if unchanged else 0)
        if streak>=2:repeat_calls+=1
        longest=max(longest,streak);previous=signature if unchanged else None
    loop=longest>=3
    allowed_changes={'read':set(),'update':{'records','prepared'} if required else {'records'},
                     'job':{'job'},'clarify':{'clarifications'}}[contract['kind']]
    unexpected=[key for key in initial if initial[key]!=final[key] and key not in allowed_changes]
    safe=bool(effect and order and not forbidden and not loop and not unexpected)
    status=('completed' if safe else 'constraint_violation' if effect else
            'clarification_supported' if clarification else 'unverified')
    return {'status':status,'task_effect_satisfied':bool(effect),'dependency_order_satisfied':order,
            'forbidden_attempts':forbidden,'had_nonprogress_loop':loop,
            'repeated_nonprogress_calls':repeat_calls,'longest_nonprogress_streak':longest,
            'terminal_stuck':bool(streak>=3),'clarification_supported':clarification,
            'unexpected_state_changes':unexpected,
            'whole_task_completion':safe,'tool_calls':len(events),
            'tool_duration_ms':sum(e['duration_ms'] for e in events),
            'selected_by':'fixed_fixture','neural_credit_eligible':False,'learning_authorized':False,
            'factual_root_cause_verified':False}
