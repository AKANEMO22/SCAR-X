import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_visual_word_readout import macro,job_parts
import pytest
import json
from scarx.runtime.experiment_status import visual_word_statistics


def test_empty_diagnostic_subset_is_missing_and_macro_does_not_reweight_by_casecount():
    assert macro([]) is None
    rows=[{'kind':'single_color','language':'vi','exact':True}]*9+[{'kind':'object_count','language':'vi','exact':False}]
    assert macro(rows)==.5


def test_worker_argument_contract_preserves_old_and_recovery_payloads():
    base=tuple(range(9));assert job_parts(base)==(base,None)
    source={'checkpoint':'public-complete-checkpoint'};assert job_parts(base+(source,))==(base,source)
    with pytest.raises(ValueError):job_parts(base+(source,source))


def test_public_visualword_metrics_keep_missing_and_signed_contrasts_without_private_fields(tmp_path):
    run=tmp_path/'runs'/'visual';run.mkdir(parents=True)
    data={'decision':'STOP_VISUAL_WORD_READOUT_V1','hidden_representation_learning':False,
        'summary':{'full':{'macro_exact':0.,'novel_composition_macro':None,'by_kind':{'single_color':.2},'private':'omit'}},
        'paired_trained_control_gains':{'no_image':{'mean':-.02,'lower95':-.03,'upper95':0.,'n_seeds':3,'private':'omit'}}}
    (run/'result.json').write_text(json.dumps(data),encoding='utf-8')
    r=visual_word_statistics(tmp_path,'visual');assert r['rows'][0]['macro_exact']==0 and r['rows'][0]['novel_composition_macro'] is None
    assert r['contrasts']['paired_trained_control_gains']['no_image']['mean']==-.02 and 'private' not in json.dumps(r)
    audit=tmp_path/'runs'/'audit';audit.mkdir();(audit/'result.json').write_text(json.dumps({'source_run':'runs/visual'}),encoding='utf-8')
    assert visual_word_statistics(tmp_path,'audit')['source_run']=='visual'
    (audit/'result.json').write_text(json.dumps({'source_run':'runs/../../outside'}),encoding='utf-8');assert not visual_word_statistics(tmp_path,'audit')


def test_visual_control_metrics_require_isolation_and_never_label_hidden_frozen(tmp_path):
    run=tmp_path/'runs'/'control';run.mkdir(parents=True)
    data={'decision':'ISOLATED_VISUAL_WORD_CONTROL_ONLY','control_only':True,'weights_imported_into_scarx':False,
        'interpretation':'CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET',
        'summary':{'bptt_control':{'macro_exact':.2,'novel_composition_macro':None,'by_kind':{'single_shape':0.}}},
        'paired_trained_control_gains':{'frozen_full':{'mean':-.01,'lower95':-.03,'upper95':.02,'n_seeds':3,'private':'omit'}}}
    (run/'result.json').write_text(json.dumps(data),encoding='utf-8');r=visual_word_statistics(tmp_path,'control')
    assert r['control_only'] and r['rows'][0]['condition']=='bptt_control' and r['rows'][0]['single_shape']==0.
    assert r['contrasts']['paired_trained_control_gains']['frozen_full']['mean']==-.01 and 'private' not in json.dumps(r)
    data['weights_imported_into_scarx']=True;(run/'result.json').write_text(json.dumps(data),encoding='utf-8')
    assert not visual_word_statistics(tmp_path,'control')
