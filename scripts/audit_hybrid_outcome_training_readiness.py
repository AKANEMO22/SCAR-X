"""Read existing P0 artifacts and source interfaces; no imports of model code."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def literal(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    return ast.literal_eval(node.value)


def audit():
    run=ROOT/'runs/20261115T044756Z-outcome-contract-p0-v1-5842c6'
    inventory=json.loads((run/'artifact-inventory.json').read_text(encoding='utf-8'))
    for name,entry in inventory.items():
        path=(run/name).resolve()
        if not path.is_relative_to(run.resolve()):raise ValueError('Artifact escapes run')
        assert path.stat().st_size==entry['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    result=json.loads((run/'result.json').read_text(encoding='utf-8'));assert result['gate_pass']
    specs=json.loads((run/'locked-cases.json').read_text(encoding='utf-8'))
    selectors=Counter();tools=Counter();initial_states=set();private_tasks=set();rows=[]
    public_question_fields={'question','instruction','user_message','task_text'}
    empty_first=0;no_decision=0;supplied_user_task=0;authorized_labels=0
    for definition in specs:
        directory=run/'fixtures'/definition['name']
        trace=json.loads((directory/'trace.json').read_text(encoding='utf-8'))
        contract=json.loads((directory/'contract.json').read_text(encoding='utf-8'))
        outcome=json.loads((directory/'outcome.json').read_text(encoding='utf-8'))
        initial_states.add(trace['initial_snapshot'])
        private_tasks.add(json.dumps({k:v for k,v in contract.items() if k not in ('task_id','scope')},sort_keys=True))
        prompt_fields=public_question_fields&set(trace)
        supplied_user_task+=bool(prompt_fields)
        for event in trace['events']:
            selectors[event['selector']]+=1;tools[event['tool']]+=1
        if trace['events']:empty_first+=trace['events'][0]['observations_before_decision']==[]
        else:no_decision+=1
        authorized_labels+=bool(outcome['learning_authorized'])
        rows.append({'name':definition['name'],'public_user_task_fields':sorted(prompt_fields),
                     'recorded_decisions':len(trace['events']),
                     'learning_authorized':outcome['learning_authorized']})
    production=literal(ROOT/'src/scarx/hybrid/learner.py','ACTIONS')
    shadow=literal(ROOT/'src/scarx/evaluation/recovery_policy.py','ACTION_ORDER')
    fields=literal(ROOT/'src/scarx/evaluation/hybrid_incident_encoder.py','FIELDS')
    assert selectors=={'fixed_fixture':42} and len(specs)==14
    assert supplied_user_task==authorized_labels==0 and empty_first==13 and no_decision==1
    names=['src/scarx/hybrid/learner.py','src/scarx/hybrid/engine.py',
           'src/scarx/evaluation/recovery_policy.py','src/scarx/evaluation/hybrid_incident_encoder.py']
    return {'verdict':'P0_FIXTURES_ARE_NOT_A_NEURAL_POLICY_TRAINING_DATASET',
        'existing_P0_run':run.name,'all_existing_artifact_inventory_hashes_verified':True,
        'traces':len(specs),'actual_decision_origins':dict(selectors),
        'traces_with_a_public_user_task_field':supplied_user_task,
        'first_decisions_with_empty_prior_tool_history':empty_first,'traces_without_a_decision':no_decision,
        'distinct_initial_snapshot_hashes':len(initial_states),'distinct_private_checker_contracts':len(private_tasks),
        'learning_authorized_outcomes':authorized_labels,'raw_recorded_tool_actions':dict(tools),
        'production_action_order':list(production),'research_shadow_action_order':list(shadow),
        'raw_tool_names_are_not_neural_action_slot_labels':True,
        'legacy_shadow_observation_fields':sorted(fields),
        'P0_policy_view_fields':['tool','arguments','result','duration_ms'],
        'direct_encoder_compatibility':False,
        'what_is_missing':['Public task meaning from the actual request, not serialized private checker/gold',
            'Versioned causal observation adapter and explicit action macro binding',
            'Immutable decision-to-executed-action-to-independent-outcome attribution before any reward'],
        'not_a_prediction_accuracy_bound_or_new_model_result':True,
        'new_samples_tools_model_initializations_forward_or_updates':0,
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        'trace_details':rows,'research_or_production_PASS':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=audit()
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='trace_details'},ensure_ascii=False))
