import json,hashlib
from scarx.app.observatory import control_answer_archive


def test_archive_is_readonly_public_pair_filtered_and_not_selected_for_correctness(tmp_path):
    data=tmp_path/'data/dialogue';data.mkdir(parents=True)
    (data/'generation1.json').write_text(json.dumps({'families':[{'train':['public question'],'answer':'Hello.'}]}),encoding='utf-8')
    run=tmp_path/'runs'/'actual-control-record';run.mkdir(parents=True)
    value={'record':{'method':'isolated_surrogate_bptt','responses':1024,'seed':28001,'input_encoding':'legacy','checkpoint':'private-weight-path'},
        'outputs':{'development':[{'prompt':'public question','target':'Hello.','generated':'Wrong recorded answer.','normalized_style_exact':False,'private_guard':'never return'},
            {'prompt':'private question','target':'private target','generated':'private output'}]}}
    path=run/'control-28001-legacy.json';path.write_text(json.dumps(value),encoding='utf-8');before=hashlib.sha256(path.read_bytes()).hexdigest()
    result=control_answer_archive(tmp_path,run.name)
    assert result['case_count']==1 and result['rows'][0]['style_match'] is False and result['rows'][0]['generated']=='Wrong recorded answer.'
    assert 'private' not in json.dumps(result) and hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert control_answer_archive(tmp_path,'../../outside')=={}
    reference=tmp_path/'runs'/'qwen-reference';reference.mkdir()
    pointer=reference/'answer-source.json';pointer.write_text(json.dumps({'scarx_control_run':'runs/'+run.name}),encoding='utf-8')
    assert control_answer_archive(tmp_path,reference.name)['case_count']==1
    pointer.write_text(json.dumps({'scarx_control_run':'runs/../../outside'}),encoding='utf-8');assert not control_answer_archive(tmp_path,reference.name)
    path.write_text('x'*1_000_001,encoding='utf-8');assert control_answer_archive(tmp_path,run.name)=={}
    value['record']={'seed':31013,'condition':'paired','experiences':1024,'observations':1024}
    association=run/'31013-paired.json';association.write_text(json.dumps(value),encoding='utf-8')
    r=control_answer_archive(tmp_path,run.name);assert r['case_count']==1 and r['learning_mechanism']=='observed_association' and r['condition']=='paired'
    assert 'BPTT' not in r['scope'] and r['rows'][0]['generated']=='Wrong recorded answer.'
