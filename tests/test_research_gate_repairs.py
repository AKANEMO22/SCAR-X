import copy
import numpy as np
import pytest
from scarx.evaluation.forensic.reference_evaluator import AuditViolation
from scarx.evaluation.forensic.research_gates import (
    retention_metrics,retention_gate,audit_p1_summaries,require_injective_codebook,binary_query_ambiguity_bound)


def matrix(old=.8):return [[.8,0],[old,.8]]


def test_empty_missing_summary_never_passes():
    result=audit_p1_summaries({})
    assert not result['overall_pass'] and len(result['tasks'])==4
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' for r in result['tasks'].values())


def test_high_accuracy_does_not_replace_credit_controls_or_ci():
    raw={'p1_results':{'xor':{'learned_mtsg':{'direct_acc_mean':1.,'probe_gain_lower95':-.1}}}}
    result=audit_p1_summaries(raw)['tasks']['xor']
    assert not result['pass'] and 'master_probe_gain_lower95_below_5pp' in result['failures']
    assert 'shuffled_signal' in result['missing']


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-.1,1.1,True,'0.9'])
def test_invalid_direct_metrics_cannot_silently_pass(bad):
    with pytest.raises(AuditViolation):audit_p1_summaries({'p1_results':{'xor':{'learned_mtsg':{'direct_acc_mean':bad}}}})


def test_retention_metrics_recomputed_from_best_to_final():
    r=retention_metrics([[.7,0,0],[.9,.8,0],[.7,.8,.8]])
    assert r['mean_forgetting']==pytest.approx(.2/3) and r['worst_forgetting']==pytest.approx(.2)


def test_average_cannot_hide_one_failed_seed():
    good={'seed':1,'matrix':matrix()};bad={'seed':2,'matrix':matrix(.7)}
    result=retention_gate([good,bad],[1,2])
    assert result['seeds_passing']==1 and not result['all_seed_numeric_gate_pass']


@pytest.mark.parametrize('records,seeds',[([],[]),([{'seed':1,'matrix':matrix()}],[1,2]),
    ([{'seed':1,'matrix':matrix()},{'seed':1,'matrix':matrix()}],[1,2])])
def test_missing_or_duplicate_seeds_fail_closed(records,seeds):
    with pytest.raises(AuditViolation):retention_gate(records,seeds)


def test_nan_observed_matrix_rejected_unobserved_future_ignored():
    m=matrix();m[0][1]=float('nan');retention_metrics(m)
    m[1][0]=float('nan')
    with pytest.raises(AuditViolation):retention_metrics(m)


def test_injective_codes_have_no_label_ambiguity():
    codes={i:(i//2,i%2) for i in range(4)}
    assert require_injective_codebook(codes)
    assert binary_query_ambiguity_bound(codes)['noiseless_uniform_key_binary_value_ceiling']==1


def test_colliding_query_codes_block_benchmark_before_training():
    codes={0:(1,0),1:(1,0)}
    with pytest.raises(AuditViolation,match='Ambiguous'):require_injective_codebook(codes)
    assert binary_query_ambiguity_bound(codes)['noiseless_uniform_key_binary_value_ceiling']==.75


def test_r8_8key_code_groups_have_finite_oracle_ceiling():
    codes={0:(1.5,0),1:(0,1.5),2:(1,1),3:(1.5,1.5),4:(0,0),5:(1.5,0),6:(0,0),7:(1.5,0)}
    assert binary_query_ambiguity_bound(codes)['noiseless_uniform_key_binary_value_ceiling']==.84375
