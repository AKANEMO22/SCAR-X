import json,time
from pathlib import Path
import pytest
from scarx.runtime.experiment_status import start_run,progress,complete_run,dashboard,report
from scarx.runtime.experiment_status import primary_statistics
from scarx.runtime.experiment_status import retention_statistics
from scarx.runtime.experiment_status import _summary
from scarx.runtime.experiment_status import word_statistics
from scarx.runtime.experiment_status import credit_signal_statistics
from scarx.runtime.experiment_status import association_statistics
from scarx.app.generation import Generation
from scarx.app.capacity_lab import CapacityLab,cases


def test_observatory_checks_process_identity_bounds_events_and_retains_failure(tmp_path):
    run=tmp_path/'runs'/'audit-one';run.mkdir(parents=True)
    start_run(run,{'hidden':48,'api_key':'must_not_be_exposed'})
    for i in range(52):progress(run,'trial',i,52,{'accuracy':.2},neural={'computed':48,'allocated':48,'spikes':[0]*48})
    value=dashboard(tmp_path)['jobs'][0]
    assert value['process_alive'] and len(value['events'])==40 and 'api_key' not in value['config']
    path=tmp_path/'local/experiment-status/audit-one.json';record=json.loads(path.read_text(encoding='utf-8'));record['process_created']-=100
    path.write_text(json.dumps(record),encoding='utf-8')
    value=dashboard(tmp_path)['jobs'][0];assert not value['process_alive'] and value['status']=='stopped_without_result'
    complete_run(run,{'decision':'NO_GO','deployed':False})
    assert dashboard(tmp_path)['jobs'][0]['result']['decision']=='NO_GO'
    (run/'report.md').write_text('A failed experiment remains a result.',encoding='utf-8')
    assert b'failed' in report(tmp_path,'audit-one')
    with pytest.raises(ValueError):report(tmp_path,'../secret')


def test_confirmation_metrics_are_read_from_bounded_results_without_fabricating_missing_values(tmp_path):
    run=tmp_path/'runs'/'confirmation';run.mkdir(parents=True)
    source={'source_confirmation_statistics':{'order':{'probe_gain':{'mean':.18,'lower95':.14,'upper95':.23,'n_seeds':8},'direct':{'mean':.92},'holm_gain_p':.0234,'pass':True},
        'missing':{'probe_gain':{'mean':.2}}}}
    (run/'result.json').write_text(json.dumps(source),encoding='utf-8')
    values=primary_statistics(tmp_path,'confirmation');assert list(values)==['order'] and values['order']['seeds']==8 and values['order']['pass']
    assert values['order']['lower95']==.14 and not primary_statistics(tmp_path,'../outside')
    (run/'result.json').write_text('x'*262145,encoding='utf-8');assert not primary_statistics(tmp_path,'confirmation')


def test_retention_summary_follows_only_one_validated_source_and_keeps_offline_missing_values(tmp_path):
    base=tmp_path/'runs';source=base/'p2';audit=base/'audit';source.mkdir(parents=True);audit.mkdir()
    (source/'result.json').write_text(json.dumps({'summary':{'stream':{'acquisition':.57,'final':.35,'mean_forgetting':.29,'worst_forgetting':.72,'joint_refit_probe':.53},
        'offline_joint_full':{'final':.31,'acquisition':None,'mean_forgetting':None,'joint_refit_probe':.49}},'absolute_gates':{'stream':False}}),encoding='utf-8')
    (audit/'result.json').write_text(json.dumps({'source_run':'runs/p2'}),encoding='utf-8')
    values=retention_statistics(tmp_path,'audit');assert values['source_run']=='p2' and values['rows']['stream']['mean_forgetting']==.29
    assert values['rows']['offline_joint_full']['offline'] and values['rows']['offline_joint_full']['acquisition'] is None
    (audit/'result.json').write_text(json.dumps({'source_run':'runs/../../outside'}),encoding='utf-8');assert not retention_statistics(tmp_path,'audit')


def test_product_summary_excludes_private_paths_and_targets_from_status_journal():
    value={'conditions':[{'mode':'readout','curriculum_order_seed':1701,'screen_pass':False,
        'candidate_private_path':'private-weight-path','target':'private target',
        'states':{'canonicalreset':{'known':.9,'antigravity':.17,'lost_personal_guards':1,'prompt':'private guard'}}}]}
    summary=_summary(value);trial=summary['product_trials'][0]
    assert trial['mode']=='readout' and trial['known_reset']==.9 and trial['lost_guards_total']==1
    encoded=json.dumps(summary);assert 'private-' not in encoded and 'private target' not in encoded and 'private guard' not in encoded


def test_word_metrics_keep_actual_zero_missing_and_reject_private_unbounded_results(tmp_path):
    run=tmp_path/'runs'/'word-check';run.mkdir(parents=True)
    data={'summary':{'known-digest128-v1':{'full':{'development':{'normalized_style_exact':0.,'teacher_token_accuracy':.5},
        'checkpoint':'private/path','target':'private words','prior':{'normalized_style_exact':float('nan')}}}}}
    (run/'result.json').write_text(json.dumps(data),encoding='utf-8')
    rows=word_statistics(tmp_path,'word-check');assert len(rows)==1 and rows[0]['development']==0 and rows[0]['prior'] is None and rows[0]['train16'] is None
    assert 'private' not in json.dumps(rows) and not word_statistics(tmp_path,'../outside')
    audit=tmp_path/'runs'/'word-audit';audit.mkdir();(audit/'result.json').write_text(json.dumps({'source_run':'runs/word-check'}),encoding='utf-8')
    assert word_statistics(tmp_path,'word-audit')[0]['source_run']=='word-check'
    (audit/'result.json').write_text(json.dumps({'source_run':'runs/word-check','control_only':True}),encoding='utf-8')
    assert not word_statistics(tmp_path,'word-audit') # never label old local metrics as BPTT
    data['control_only']=True;(run/'result.json').write_text(json.dumps(data),encoding='utf-8')
    rows=word_statistics(tmp_path,'word-audit');assert rows[0]['control_only'] and 'BPTT' in rows[0]['method']
    data['summary']['known-digest128-v1']['local_reference']=data['summary']['known-digest128-v1']['full']
    (run/'result.json').write_text(json.dumps(data),encoding='utf-8')
    assert any('local-score' in r['method'] for r in word_statistics(tmp_path,'word-audit'))
    (audit/'result.json').write_text(json.dumps({'source_run':'runs/../../outside'}),encoding='utf-8');assert not word_statistics(tmp_path,'word-audit')
    (run/'result.json').write_text('x'*262145,encoding='utf-8');assert not word_statistics(tmp_path,'word-check')


def test_credit_signal_reader_keeps_negative_missing_and_only_descriptive_audit_fields(tmp_path):
    run=tmp_path/'runs'/'signal-check';run.mkdir(parents=True)
    data={'decision':'READONLY_CREDIT_SIGNAL_DIAGNOSTIC_ONLY','new_training':False,
        'summary':{'local_final':{'existing_score':{'replicate_cosine':-.2,'cross_signal_to_pair_noise':-.001,
            'pair_noise_energy':0.,'sample_mean_norm':float('inf'),'private_path':'never expose'}}},
        'paired_statistics':{'local_final':{'crossfit_vs_existing_noise_ratio':{'mean':.1,'lower95':.05,'upper95':.2,'n_seeds':3,'private_text':'secret'}}}}
    path=run/'result.json';path.write_text(json.dumps(data),encoding='utf-8');r=credit_signal_statistics(tmp_path,'signal-check')
    assert r['rows'][0]['replicate_cosine']==-.2 and r['rows'][0]['cross_signal_to_pair_noise']==-.001 and r['rows'][0]['pair_noise_energy']==0
    assert r['rows'][0]['sample_mean_norm'] is None and 'private' not in json.dumps(r)
    data['new_training']=True;path.write_text(json.dumps(data),encoding='utf-8');assert not credit_signal_statistics(tmp_path,'signal-check')
    assert not credit_signal_statistics(tmp_path,'../private')


def test_association_reader_keeps_actual_failure_zero_and_excludes_private_fields(tmp_path):
    run=tmp_path/'runs'/'association';run.mkdir(parents=True)
    data={'decision':'STOP_OBSERVED_ASSOCIATION_V1','summary':{'paired':{'development':{'normalized_style_exact':0.},'prior':{'normalized_style_exact':.02},'private':'not returned'}},'probe_by_condition':{'paired':.3}}
    (run/'result.json').write_text(json.dumps(data),encoding='utf-8')
    result=association_statistics(tmp_path,run.name);assert result['decision']=='STOP_OBSERVED_ASSOCIATION_V1' and result['rows'][0]['development']==0 and result['rows'][0]['prompt_probe']==.3
    assert 'private' not in json.dumps(result) and not association_statistics(tmp_path,'../private')
    audit=tmp_path/'runs'/'audit-association';audit.mkdir();(audit/'result.json').write_text(json.dumps({'source_run':'runs/'+run.name}),encoding='utf-8')
    assert association_statistics(tmp_path,audit.name)['source_run']==run.name


def test_capacity_lab_real_growth_dormancy_and_no_serving_state_change(tmp_path):
    app=Generation(tmp_path,background=False);lab=CapacityLab(app)
    try:
        before=app._pack();assert lab.start('release')['started']
        assert not lab.start('release')['started']
        deadline=time.monotonic()+15
        while lab.status()['status']=='running' and time.monotonic()<deadline:time.sleep(.03)
        value=lab.status();assert value['status']=='completed',value
        steps=value['steps'];assert len(steps)==5
        assert [row['active'] for row in steps]==[48,64,96,128,48]
        assert [row['allocated'] for row in steps]==[48,64,96,128,128]
        assert steps[-1]['dormant']==80 and app._pack()==before
        assert all(row['same_core_identity'] and not row['weights_updated'] and row['neural_ticks']>0 for row in steps)
        assert all(len(f['spikes'])==row['allocated'] and f['computed']==row['active'] for row in steps for f in row['frames'])
        assert len(cases()[3][2].split())>100
        restored=CapacityLab(app)
        assert restored.status()['restored_history'] and restored.status()['steps']==steps
    finally:lab.close();app.close()


def test_live_capacity_observation_grows_same_serving_core_without_synthetic_learning(tmp_path):
    app=Generation(tmp_path,background=False);lab=CapacityLab(app)
    try:
        serving=id(app.model.core);memories=list(app.memories);dialogue=app.dialogue.dumps();updates=app.model.core.updates
        old_input=app.model.core.w_in.copy();old_output=app.model.core.w_out.copy();old_recurrent=app.model.core.w_rec.copy()
        assert lab.start('live')['started']
        deadline=time.monotonic()+15
        while lab.status()['status']=='running' and time.monotonic()<deadline:time.sleep(.03)
        value=lab.status();assert value['status']=='completed',value
        assert app.model.core.config.hidden==128 and id(app.model.core)==serving
        assert app.model.core.updates==updates and app.memories==memories and app.dialogue.dumps()==dialogue
        import numpy as np
        np.testing.assert_array_equal(app.model.core.w_in[:48],old_input)
        np.testing.assert_array_equal(app.model.core.w_out[:,:48],old_output)
        np.testing.assert_array_equal(app.model.core.w_rec[:48],old_recurrent)
        assert value['checkpoint_before'] and len(value['steps'])==5
        assert [r['active'] for r in value['steps']]==[48,64,96,128,48]
    finally:lab.close();app.close()


def test_local_observer_http_requires_existing_mutation_auth_and_rejects_report_traversal(tmp_path):
    import threading,urllib.request,urllib.error
    from scarx.app.generation_server import make_server
    app=Generation(tmp_path,background=False);server=make_server(app,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}'
    def get(path):return json.load(urllib.request.urlopen(base+path,timeout=5))
    def post(path,headers=None):
        request=urllib.request.Request(base+path,b'{}',headers={'Content-Type':'application/json',**(headers or {})})
        return json.load(urllib.request.urlopen(request,timeout=5))
    try:
        data=get('/api/research');assert data['capacity_demo']['status']=='idle' and data['live_capacity']['model']=='scarx-generation1'
        with pytest.raises(urllib.error.HTTPError) as error:post('/api/capacity-demo')
        assert error.value.code==403
        token=get('/api/status')['token']
        with pytest.raises(urllib.error.HTTPError) as error:post('/api/capacity-demo',{'X-SCARX-Token':token,'Origin':'https://example.com'})
        assert error.value.code==403 and get('/api/research')['capacity_demo']['status']=='idle'
        with pytest.raises(urllib.error.HTTPError) as error:get('/api/research/report?run_id=..%2Fsecret')
        assert error.value.code==400
        assert post('/api/capacity-demo',{'X-SCARX-Token':token})['started']
        assert post('/api/capacity-demo/stop',{'X-SCARX-Token':token})['stop_requested']
    finally:server.shutdown();server.server_close();thread.join();app.close()
