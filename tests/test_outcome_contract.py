"""Pure malformed/tampered evidence regressions; no model or tool execution."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_test_pure_outcome_contract',ROOT/'src/scarx/evaluation/outcome_contract.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def example(path='requested.txt'):
    text='owned data';entry={'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest()}
    snapshot={'files':{'requested.txt':entry,'other.txt':entry},'records':[],
              'prepared':0,'job':{},'request':{},'request_schema':[],'clarifications':[]}
    key=C.digest(snapshot)
    contract={'format':C.FORMAT,'task_id':'task','scope':'owned','kind':'read','source':'requested.txt',
              'value':entry['sha256'],'required_order':[],'missing_field':None}
    result={'ok':True,'source':path,'text':text,'verified':True}
    event={'index':0,'task_id':'task','scope':'owned','decision_id':'d1','selector':'fixed_fixture',
        'proposed_action':'read_file','executed_action':'read_file','decision_cutoff':1.,'started_epoch':2.,
        'returned_epoch':3.,'duration_ms':0.1,'tool':'read_file','arguments':{'path':path},
        'result':result,'result_sha256':C.digest(result),'before_snapshot':key,'after_snapshot':key,
        'observations_before_decision':[]}
    trace={'format':C.FORMAT,'task_id':'task','scope':'owned','initial_snapshot':key,'final_snapshot':key,'events':[event]}
    return trace,{key:snapshot},contract


def test_asserted_verified_flag_cannot_make_the_other_file_a_task_proof():
    a,s,c=example('other.txt');out=C.evaluate(a,s,c)
    assert not out['task_effect_satisfied'] and not out['whole_task_completion'] and not out['learning_authorized']


@pytest.mark.parametrize('fault',['scope','task','clock','future_view','result_hash','snapshot_chain','snapshot_content','neural_selector'])
def test_tampered_or_future_evidence_is_rejected(fault):
    a,s,c=example();e=a['events'][0]
    if fault=='scope':e['scope']='foreign'
    elif fault=='task':e['task_id']='other task'
    elif fault=='clock':e['returned_epoch']=float('nan')
    elif fault=='future_view':e['observations_before_decision']=[{'contract':c}]
    elif fault=='result_hash':e['result']['text']='invented'
    elif fault=='snapshot_chain':e['before_snapshot']='missing'
    elif fault=='snapshot_content':next(iter(s.values()))['files']={}
    elif fault=='neural_selector':e['selector']='neural_policy'
    with pytest.raises(ValueError):C.evaluate(a,s,c)


def test_empty_events_do_not_default_to_a_perfect_task_score():
    a,s,c=example();a['events']=[]
    assert C.evaluate(a,s,c)['status']=='unverified'


def test_unrelated_side_effect_prevents_whole_task_completion():
    a,s,c=example();changed=copy.deepcopy(next(iter(s.values())))
    changed['prepared']=1;key=C.digest(changed);s[key]=changed
    a['events'][0]['after_snapshot']=a['final_snapshot']=key
    out=C.evaluate(a,s,c)
    assert out['task_effect_satisfied'] and out['unexpected_state_changes']==['prepared']
    assert not out['whole_task_completion']


def test_policy_observation_excludes_future_results_gold_and_path_identity():
    a,_,_=example();event=a['events'][0]
    event['result']['message']='C:/fixture/case-09/missing.txt'
    assert C.observed_prefix([event],2.5)==[]
    view=C.observed_prefix([event],4.)
    assert len(view)==1 and 'message' not in view[0]['result'] and 'verified' not in view[0]['result']
    assert set(view[0])=={'tool','arguments','result','duration_ms'}
