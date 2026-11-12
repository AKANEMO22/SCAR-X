import copy
import numpy as np
import pytest
from scarx.evaluation.r8_confirmation import SEEDS,PRIMARY_TASKS,expected_cells,cluster_statistics,summarize_confirmation


def rows_fixture():
    rows=[]
    for task,seed,arm,budget in expected_cells():
        good=arm=='stopped_correct';own=arm=='own_reset_correct'
        score=.95 if good else .85 if own else .65
        if task=='order':score=.99
        rows.append({'task':task,'seed':seed,'arm':arm,'budget':budget,
            'metrics':{k:score for k in ('id_direct','id_probe','ood_direct','ood_probe')},
            'hidden_unchanged':True,'initial_features_match':True,
            'predictions':{'id':[0,1]} if arm in ('stopped_zero','stopped_readout_only') else {'id':[1,0]}})
    return rows


def test_exact_approved_bank_and_job_count_without_opening_bank():
    assert len(SEEDS)==12 and len(set(SEEDS))==12 and len(expected_cells())==408
    assert sum(key[3]=='equal_updates' for key in expected_cells())==204


def test_bootstrap_and_sign_flip_are_exact_and_reproducible():
    # Synthetic effect vector only, no new-seed model initialization or inference.
    a=cluster_statistics(np.full(12,.1));b=cluster_statistics(np.full(12,.1))
    assert a==b and a['lower95']==pytest.approx(.1) and a['exact_two_sided_p']==2/4096


def test_missing_or_duplicate_cells_never_pass():
    assert not summarize_confirmation([])['master_p1_pass']
    rows=rows_fixture();assert not summarize_confirmation(rows[:-1])['complete']
    with pytest.raises(ValueError):summarize_confirmation(rows+[rows[0]])


def test_complete_numeric_fixture_passes_but_not_production():
    result=summarize_confirmation(rows_fixture())
    assert result['master_p1_pass'] and result['qualifying_primary_tasks']==3
    assert len(result['primary_holm_p'])==7 and not result['production_ready']


def test_no_ceiling_exception_for_no_hidden_gain():
    rows=rows_fixture()
    for row in rows:
        row['metrics']={k:1. for k in row['metrics']}
    result=summarize_confirmation(rows)
    assert not result['master_p1_pass'] and result['qualifying_primary_tasks']==0


def test_protected_regression_blocks_otherwise_passing_tasks():
    rows=rows_fixture()
    for row in rows:
        if row['task']=='order' and row['arm']=='stopped_correct':row['metrics']['id_direct']=.9
    assert not summarize_confirmation(rows)['master_p1_pass']


def test_ood_regression_cannot_be_hidden_by_id_accuracy():
    rows=rows_fixture()
    for row in rows:
        if row['task'] in PRIMARY_TASKS[:2] and row['arm']=='stopped_correct':row['metrics']['ood_probe']=.7
    result=summarize_confirmation(rows);assert result['qualifying_primary_tasks']==1 and not result['master_p1_pass']


def test_missing_control_invariance_and_nan_block_pass():
    rows=rows_fixture();rows[0]['metrics']['id_direct']=float('nan')
    with pytest.raises(ValueError):summarize_confirmation(rows)
    rows=rows_fixture()
    for row in rows:
        if row['arm']=='stopped_zero':row['hidden_unchanged']=False
    assert not summarize_confirmation(rows)['master_p1_pass']


def test_cpu_ceiling_does_not_require_identical_head_fit_at_different_update_counts():
    rows=rows_fixture()
    for row in rows:
        if row['budget']=='equal_cpu' and row['arm']=='stopped_zero':row['predictions']={'different_head_exposure':[1,1]}
    assert summarize_confirmation(rows)['controls_valid']
