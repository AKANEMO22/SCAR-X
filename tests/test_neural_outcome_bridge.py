"""Pure interface/journal/envelope tests; no model initialization or forward."""
import copy
import io
import json
import time
import zipfile
import numpy as np
import pytest

from scarx.evaluation.neural_outcome_bridge import (
    ADAPTER,FORMAT,MACROS,BindingJournal,FrozenBridgePolicy,encode_task_observations,
)
from scarx.evaluation.recovery_policy import ACTION_ORDER
from scarx.evaluation.bridge_attempt_budget import AttemptBudget

TASK={'instruction':'Read requested.txt','request':{'tool':'read_file','arguments':{'path':'requested.txt'}}}


def observation():
    return {'tool':'read_file','reported_status':'error','error_type':'FileNotFoundError',
        'empty_result':False,'spillover':False,'streak':1,'probe':None,'returned_epoch':10.}


@pytest.mark.parametrize('bad',['future','private_task','private_observation','unsafe_path','bad_clock'])
def test_invalid_input_never_reaches_a_neural_forward(bad):
    task=copy.deepcopy(TASK);obs=[observation()];cutoff=11.
    if bad=='future':obs[0]['returned_epoch']=12.
    elif bad=='private_task':task['contract']={'expected':'gold'}
    elif bad=='private_observation':obs[0]['case_id']='secret fixture ID'
    elif bad=='unsafe_path':task['request']['arguments']['path']='../foreign.txt'
    elif bad=='bad_clock':cutoff=float('nan')
    with pytest.raises(ValueError):encode_task_observations(task,obs,cutoff)


def test_long_question_cannot_remove_the_observation_block():
    task=copy.deepcopy(TASK);task['instruction']='word '*180
    a=encode_task_observations(task,[observation()],11.)
    row=observation();row.update(reported_status='ok',error_type='')
    b=encode_task_observations(task,[row],11.)
    assert a.shape==b.shape==(16,64)
    np.testing.assert_array_equal(a[:8],b[:8]);assert not np.array_equal(a[8:],b[8:])


def test_empty_history_is_valid_and_private_bindings_are_not_implicit():
    assert encode_task_observations(TASK,[],11.).shape==(8,64)
    task=copy.deepcopy(TASK);task['request']['expected_contents']='gold'
    with pytest.raises(ValueError):encode_task_observations(task,[],11.)


def test_authoritative_receipt_checks_reject_rehashed_foreign_fields(tmp_path):
    journal=BindingJournal(tmp_path/'bindings.sqlite')
    try:
        journal.register_context('task','scope',TASK)
        # Serialized interface fixture, not a claimed native model output.
        decision={'id':'declared-serialization-fixture','task_id':'task','scope':'scope',
                  'public_task':TASK,'proposed_action':'retry'}
        journal.record_decision(decision)
        receipt={'decision_id':decision['id'],'task_id':'task','scope':'scope','proposed_action':'retry',
                 'adapter':ADAPTER,'macros':MACROS,'learning_authorized':False}
        journal.record_receipt(decision,receipt)
        for key,value in [('task_id','foreign'),('scope','foreign'),('decision_id','foreign'),('proposed_action','finish')]:
            changed={**receipt,key:value}
            with pytest.raises(ValueError):journal.validate_receipt(changed,decision['id'])
        other={**decision,'id':'another-valid-decision'};journal.record_decision(other)
        other_receipt={**receipt,'decision_id':other['id']};journal.record_receipt(other,other_receipt)
        with pytest.raises(ValueError):journal.validate_receipt(other_receipt,decision['id'])
        journal.revoke('task')
        assert journal.has_receipt(decision['id'])
        with pytest.raises(PermissionError):journal.context('task','scope',TASK)
    finally:journal.close()


@pytest.mark.parametrize('key,value',[('format','old'),('adapter','wrong'),('macros','wrong'),('actions',list(reversed(ACTION_ORDER)))])
def test_incompatible_envelopes_are_rejected_before_inner_model_load(tmp_path,key,value):
    journal=BindingJournal(tmp_path/'bindings.sqlite')
    try:
        payload=b'not a model; rejection must occur before deserialization'
        import hashlib
        meta={'format':FORMAT,'adapter':ADAPTER,'macros':MACROS,'actions':list(ACTION_ORDER),
            'seed':260113,'mode':'frozen','model_sha256':hashlib.sha256(payload).hexdigest(),
            'frozen_fingerprint':'fixture','journal_id':journal.identity,'pending':{}}
        meta[key]=value
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as z:
            z.writestr('metadata.json',json.dumps(meta));z.writestr('local-alif.npz',payload)
        with pytest.raises(ValueError):FrozenBridgePolicy.loads(stream.getvalue(),journal,None,None,None)
    finally:journal.close()


def test_precharge_ledger_and_owner_lock_cannot_be_reset_or_shared(tmp_path):
    path=tmp_path/'attempts.jsonl';start=time.time();budget=AttemptBudget(path,start)
    try:
        with pytest.raises(FileExistsError):AttemptBudget(path,start)
        for _ in range(24):budget.charge('forward',260113,'declared budget fixture, not a model call')
        with pytest.raises(RuntimeError):budget.charge('forward',260113,'would exceed')
        assert len(budget.records())==24
    finally:budget.close()
    second=AttemptBudget(path,start)
    try:
        with pytest.raises(RuntimeError):second.charge('forward',260137,'still consumed')
    finally:second.close()
