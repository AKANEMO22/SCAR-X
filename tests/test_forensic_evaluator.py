import numpy as np
import pytest
from scarx.evaluation.forensic.reference_evaluator import (AuditViolation,pair_metrics,oracle_events,
    validate_pair_inputs,seed_bootstrap,integrate_gates)


def fixture_pairs():
    values=[(0,0),(0,1),(1,0),(1,1)]
    x=np.zeros((8,12,8),np.float32);a=[];b=[];q=[];y=[]
    for p,(va,vb) in enumerate(values):
        for query in (0,1):
            i=2*p+query;x[i,2,[0,2+va]]=1.5;x[i,6,[1,2+vb]]=1.5;x[i,10,4+query]=1.5
            a.append(va);b.append(vb);q.append(query);y.append((va,vb)[query])
    return x,np.array(y),np.array(a),np.array(b),np.array(q),np.repeat(np.arange(4),2),np.array([False,True,True,False])


def test_independent_oracle_and_first_key_shortcut():
    x,y,a,b,q,ids,flips=fixture_pairs()
    assert np.array_equal(oracle_events(x)['labels'],y)
    result=pair_metrics(a,y,a,b,q,ids,flips)
    assert result['direct']==.75 and result['both_A']==0 and result['both_B']==1
    assert pair_metrics(y,y,a,b,q,ids,flips)['both_A']==1
    assert validate_pair_inputs(x,y,a,b,q,ids,flips)['non_query_mismatch_pairs']==0


def test_joint_score_is_neither_query_flip_nor_mean_direct():
    _,y,a,b,q,ids,flips=fixture_pairs()
    pred=1-y
    result=pair_metrics(pred,y,a,b,q,ids,flips)
    assert result['both_A']==0 and result['both_B']==0
    assert np.all(pred[::2][flips]!=pred[1::2][flips])


def test_confidence_interval_resamples_independent_seed_vector():
    result=seed_bootstrap({7:.1,11:.2,13:.3})
    assert result['independent_seed_count']==3
    assert result['mean']==pytest.approx(.2)
    assert result==seed_bootstrap({13:.3,7:.1,11:.2})


def test_missing_gate_never_becomes_pass_even_when_remaining_gates_true():
    names=['G'+str(i) for i in range(1,9)]
    assert integrate_gates(names,{n:True for n in names})['verdict']=='PASS'
    for invalid in (None,float('nan'),1,'true'):
        gates={n:True for n in names};gates['G8']=invalid
        assert integrate_gates(names,gates)['verdict']=='INCONCLUSIVE'
    assert integrate_gates(names,{n:True for n in names[:-1]})['verdict']=='INCONCLUSIVE'
    assert integrate_gates(names,{'G1':False})['verdict']=='FAIL'
