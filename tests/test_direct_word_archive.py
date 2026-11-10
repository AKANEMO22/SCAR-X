import json,hashlib
from scarx.app.observatory import direct_word_answer_archive
from scarx.evaluation.direct_word_data import TARGETS
from scarx.runtime.experiment_status import direct_word_statistics


def test_direct_word_archive_only_displays_reviewed_exact_publicpairs_and_latest_not_best(tmp_path):
    run=tmp_path/'runs'/'words';run.mkdir(parents=True);directory=tmp_path/'data/candidates'/'review';directory.mkdir(parents=True)
    cases=[{'id':f'{family}_{i}','prompt':f'publicprompt {family} {i}','target':target,'family':family,'language':family[:2],'split':'development'} for family,target in TARGETS.items() for i in range(8)]
    path=directory/'reviewed.json';path.write_text(json.dumps({'format':'scarx-direct-word-style-dataset-v1','all_rows_read':True,'automatic_learning_allowed':False,'items':cases}),encoding='utf-8')
    (run/'manifest.json').write_text(json.dumps({'config':{'protocol':'experiments/direct-word-bridge-v1.md','responses':1024,'dataset':path.relative_to(tmp_path).as_posix(),'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}),encoding='utf-8')
    outputs=[{**r,'generated':'Wrong.','exact':False} for r in cases];data={'record':{'seed':36313,'method':'correct','experiences':1024},'outputs':{'development':outputs}}
    measured=run/'word-36313-correct.json';measured.write_text(json.dumps(data),encoding='utf-8')
    result=direct_word_answer_archive(tmp_path,'words');assert result['case_count']==64 and result['rows'][0]['generated']=='Wrong.' and result['method']=='correct'
    assert not direct_word_answer_archive(tmp_path,'../../private')
    outputs[0]['prompt']='private unreviewed';measured.write_text(json.dumps(data),encoding='utf-8');assert not direct_word_answer_archive(tmp_path,'words')
    outputs[0]['prompt']=cases[0]['prompt'];measured.write_text(json.dumps(data),encoding='utf-8')
    compare=tmp_path/'runs'/'qwen';compare.mkdir();(compare/'manifest.json').write_text(json.dumps({'config':{}}),encoding='utf-8');(compare/'answer-source.json').write_text(json.dumps({'source_run':'runs/words'}),encoding='utf-8')
    assert direct_word_answer_archive(tmp_path,'qwen')['run_id']=='words'


def test_word_statistics_retain_signed_gain_and_source_stop_while_qwen_job_still_running(tmp_path):
    source=tmp_path/'runs'/'word';source.mkdir(parents=True);data={'decision':'STOP_DIRECT_WORD_BRIDGE_V1','deployed':False,
        'summary':{'correct':{'macro_development':0.,'macro_train':.5,'probe_gain':-.1,'restore_initial_hidden_dev':None,'private':'omit'}}}
    (source/'result.json').write_text(json.dumps(data),encoding='utf-8')
    compare=tmp_path/'runs'/'reference';compare.mkdir();(compare/'answer-source.json').write_text(json.dumps({'source_run':'runs/word'}),encoding='utf-8')
    r=direct_word_statistics(tmp_path,'reference');assert r['decision']=='STOP_DIRECT_WORD_BRIDGE_V1' and r['summary']['correct']['probe_gain']==-.1 and r['summary']['correct']['macro_development']==0.
    assert 'private' not in json.dumps(r) and r['summary']['correct']['restore_initial_hidden_dev'] is None
    (compare/'answer-source.json').write_text(json.dumps({'source_run':'runs/../../private'}),encoding='utf-8');assert not direct_word_statistics(tmp_path,'reference')
