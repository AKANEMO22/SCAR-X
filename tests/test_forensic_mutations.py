import numpy as np
import pytest
from test_forensic_evaluator import fixture_pairs
from scarx.evaluation.forensic.reference_evaluator import (AuditViolation,pair_metrics,
 validate_pair_inputs,scaled_features,require_scaled,require_active_shift,checked_threshold,
 ensure_joinable,unique_top_level_definitions)


def test_wrong_query_tick_mutation_is_detected_on_actual_inputs():
    x,y,a,b,q,ids,flips=fixture_pairs()
    shifted=np.zeros_like(x);shifted[:,1:]=x[:,:-1]
    assert validate_pair_inputs(shifted,y,a,b,q,ids,flips,expected_tick=11)['oracle_all_rows']
    with pytest.raises(AuditViolation,match='query tick'):validate_pair_inputs(shifted,y,a,b,q,ids,flips,expected_tick=10)


def test_omitted_scaler_and_wrong_scaler_mutations_are_detected():
    raw=np.array([[1.,3.],[2.,5.]])
    scaled=scaled_features(raw,raw.mean(0),np.ones(2),'calib-A',5)
    assert require_scaled(scaled,'calib-A').shape==raw.shape
    with pytest.raises(AuditViolation):require_scaled(raw,'calib-A')
    with pytest.raises(AuditViolation):require_scaled(scaled,'calib-B')


def test_random_labels_cannot_be_substituted_for_oracle_targets():
    x,y,a,b,q,ids,flips=fixture_pairs();bad=y.copy();bad[0]^=1
    with pytest.raises(AuditViolation,match='labels'):validate_pair_inputs(x,bad,a,b,q,ids,flips)


def test_sibling_swap_and_pair_identity_mutations_are_detected():
    x,y,a,b,q,ids,flips=fixture_pairs();bad=ids.copy();bad[[1,3]]=bad[[3,1]]
    with pytest.raises(AuditViolation,match='pair IDs'):pair_metrics(y,y,a,b,q,bad,flips)
    broken=x.copy();broken[1,0,7]=1.5
    with pytest.raises(AuditViolation,match='noise'):validate_pair_inputs(broken,y,a,b,q,ids,flips)


def test_zero_pad_feature_noop_mutation_is_not_called_effective_stress():
    clean=np.zeros((3,48),np.float32)
    with pytest.raises(AuditViolation,match='NO_OP_SHIFT'):require_active_shift(clean,clean.copy())
    changed=clean.copy();changed[0,0]=1
    require_active_shift(clean,changed)


def test_direct_substituted_for_both_A_and_literal_true_are_detected():
    _,y,a,b,q,ids,flips=fixture_pairs();m=pair_metrics(a,y,a,b,q,ids,flips)
    assert checked_threshold(m['direct'],.7)
    assert not checked_threshold(m['both_A'],.7)
    with pytest.raises(AuditViolation,match='disagrees'):checked_threshold(m['both_A'],.7,declared=True)


def test_order_changed_between_matched_control_is_not_joinable():
    record={'protocol':'p','seed':11,'train':'data','scaler':'sc','head_init':'h','order':'first'}
    ensure_joinable(record,record,tuple(record))
    with pytest.raises(AuditViolation,match='NON_EQUIVALENT_CONTROL'):ensure_joinable(record,{**record,'order':'second'},tuple(record))


def test_duplicate_function_shadow_cannot_pass_audit():
    with pytest.raises(AuditViolation,match='shadowed'):unique_top_level_definitions('def score(): pass\ndef score(): return True')
