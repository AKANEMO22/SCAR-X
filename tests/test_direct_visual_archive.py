import json,hashlib
from scarx.app.observatory import direct_visual_answer_archive
from scarx.runtime.experiment_status import visual_word_statistics
from scarx.evaluation.vision_fixtures import render,scenes


def test_directvisual_archive_validates_ownedpixels_publicpairs_condition_and_complete_rows(tmp_path):
    run=tmp_path/'runs'/'visual37';run.mkdir(parents=True);candidate=tmp_path/'data/candidates'/'dataset';candidate.mkdir(parents=True)
    pixels=render(scenes()[0]);(candidate/'development-001.png').write_bytes(pixels)
    scene={'id':'development-001','file':'development-001.png','sha256':hashlib.sha256(pixels).hexdigest(),'split':'development'}
    cases=[{'id':f'case{i}','scene_id':scene['id'],'prompt':f'Publicquestion{i}?','target':'red','kind':'single_color','language':'en','split':'development'} for i in range(864)]
    dataset=candidate/'dataset.json';dataset.write_text(json.dumps({'automatic_learning_allowed':False,'scenes':[scene],'cases':cases}),encoding='utf-8')
    config={'protocol':'experiments/direct-visual-bridge-v1.md','experiences':1536,'proposal30_approved':True,'dataset':dataset.relative_to(tmp_path).as_posix(),'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest()}
    (run/'manifest.json').write_text(json.dumps({'config':config}),encoding='utf-8')
    data={'record':{'seed':37313,'method':'correct','experiences':1536,'hidden_updates':1536},'outputs':{'development':[{**r,'generated':'Wrong.','exact':False} for r in cases]}}
    path=run/'visual37-37313-correct.json';path.write_text(json.dumps(data),encoding='utf-8');a=direct_visual_answer_archive(tmp_path,'visual37')
    assert a['case_count']==864 and a['method']=='correct' and len(a['images'])==1 and a['rows'][0]['generated']=='Wrong.'
    assert not direct_visual_answer_archive(tmp_path,'../../private')
    data['outputs']['development'][0]['prompt']='unreviewed private';path.write_text(json.dumps(data),encoding='utf-8');assert not direct_visual_answer_archive(tmp_path,'visual37')
    data['outputs']['development'][0]['prompt']=cases[0]['prompt'];path.write_text(json.dumps(data),encoding='utf-8');(candidate/'development-001.png').write_bytes(pixels+b'changed')
    assert not direct_visual_answer_archive(tmp_path,'visual37')


def test_directvisual_statistics_label_approved_credit_and_preserve_missing_signedmetrics(tmp_path):
    run=tmp_path/'runs'/'source37';run.mkdir(parents=True);data={'decision':'STOP_DIRECT_VISUAL_BRIDGE_V1','proposal30_approved':True,'deployed':False,
        'summary':{'correct':{'macro_exact':0.,'macro_train':1.,'novel_composition_macro':None,'probe_gain':-.1,'by_kind':{'single_shape':0.}}},
        'paired_gains':{'frozen':{'mean':-.05,'lower95':-.1,'upper95':.02,'n_seeds':3,'private':'omit'}}}
    (run/'result.json').write_text(json.dumps(data),encoding='utf-8');r=visual_word_statistics(tmp_path,'source37')
    assert r['direct_eligibility'] and not r['control_only'] and r['rows'][0]['condition']=='correct' and r['rows'][0]['probe_gain']==-.1
    assert r['rows'][0]['novel_composition_macro'] is None and r['contrasts']['paired_gains']['frozen']['mean']==-.05 and 'private' not in json.dumps(r)
    data['deployed']=True;(run/'result.json').write_text(json.dumps(data),encoding='utf-8');assert not visual_word_statistics(tmp_path,'source37')
