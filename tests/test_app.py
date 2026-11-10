import json
import sqlite3
import subprocess
import sys
import threading
import urllib.request
import urllib.error
import numpy as np
import pytest
from scarx.app.state import Workbench
from scarx.app.server import make_server
from scarx.app.tasks import encode,normalize_sample,expected_label
from scarx import Config
from scarx.neural.plasticity import PlasticCore
from scarx.learning.controller import LearningController
from scarx.evaluation.tasks21 import representation


@pytest.fixture
def app(tmp_path):
    instance=Workbench(tmp_path)
    try: yield instance
    finally: instance.close()


def test_prediction_and_evaluation_are_neural_and_do_not_learn(app):
    before=app._pack()
    r=app.predict({'task':'A','signal':3,'noise_seed':999}); result=app.evaluate()
    assert r['prediction']==int(np.argmax(r['probabilities']))
    assert r['retrieval_used'] is False and len(r['spikes'])==12
    assert app._pack()==before
    assert result['scores']['A']['seen_exact']==0


def test_teach_unseen_variants_and_restart(app,tmp_path):
    before=app._evaluate()['A']['accuracy']
    result=app.teach({'task':'A','count':64,'epochs':4})
    first_accuracy=result['after']['A']['accuracy']
    result=app.teach({'task':'A','count':64,'epochs':4})
    assert result['after']['A']['accuracy']-before>=.20
    assert first_accuracy>before
    assert result['after']['A']['seen_exact']==0
    assert result['weight_changes']['w_out']>0 and result['weight_changes']['w_in']>0
    snapshot=app._pack(); scores=app._evaluate(); app.close(); probe=Workbench(tmp_path)
    try: assert probe._pack()==snapshot and probe._evaluate()==scores
    finally: probe.close()


def test_single_writer_per_directory(app,tmp_path):
    with pytest.raises(RuntimeError,match='SCAR-X'): Workbench(tmp_path)
    app.close()
    reopened=Workbench(tmp_path)
    reopened.close()


@pytest.mark.parametrize('data',[
 {'task':'A','signal':3,'label':0,'confirmation':True},
 {'task':'A','signal':3,'label':1},
 {'task':'Z','signal':0,'label':0,'confirmation':True},
 {'task':'A','signal':0,'amplitude':float('nan'),'label':0,'confirmation':True},
 {'task':'A','count':128,'epochs':999},
])
def test_invalid_feedback_never_mutates(app,data):
    before=app._pack(); checkpoint=app.active
    with pytest.raises(ValueError): app.teach(data)
    assert app._pack()==before and app.active==checkpoint
    assert app.db.execute('SELECT count(*) FROM episodes').fetchone()[0]==0


def test_commit_failure_rolls_back_neural_and_sqlite(app,monkeypatch):
    before=app._pack(); active=app.active; original=app._commit
    def fail(action,details):
        original(action,details)
        raise sqlite3.OperationalError('simulated commit interruption')
    monkeypatch.setattr(app,'_commit',fail)
    with pytest.raises(sqlite3.OperationalError): app.teach({'task':'A','count':4,'epochs':1})
    assert app._pack()==before and app.active==active
    assert app.db.execute('SELECT count(*) FROM episodes').fetchone()[0]==0
    assert app.db.execute('SELECT id FROM mvp_active').fetchone()[0]==active
    assert len(app.checkpoints())==1


def test_restore_includes_rng_importance_replay_and_resumes(app):
    first=app.active; initial=app._pack()
    app.teach({'task':'A','count':16,'epochs':2}); taught=app.active; payload=app._pack()
    app.restore(first)
    assert app._pack()==initial and app.replay.size==0
    app.restore(taught)
    assert app._pack()==payload and app.replay.size==16
    assert len(app.checkpoints())==4


def test_checkpoint_corruption_does_not_reset_active(app):
    active=app.active
    app.db.execute('UPDATE mvp_checkpoints SET sha256=? WHERE id=?',('bad',active)); app.db.commit()
    with pytest.raises(ValueError,match='Checksum'): app._payload(active)
    assert app.active==active


def test_import_checkpoint_preserves_prior_generation(app):
    initial=app._pack(); first=app.active
    app.teach({'task':'A','count':16,'epochs':2}); learned=app._pack()
    app.import_checkpoint(initial)
    assert app._pack()==initial and len(app.checkpoints())==3
    app.import_checkpoint(learned)
    assert app._pack()==learned and app.learner.model.updates==32
    assert app._payload(first)==initial


def test_invalid_import_does_not_replace_state(app):
    before=app._pack(); active=app.active
    with pytest.raises(Exception): app.import_checkpoint(b'not a valid checkpoint')
    assert app._pack()==before and app.active==active


def test_old_controller_restores_plasticity_v1(tmp_path):
    m=PlasticCore(Config(inputs=8,outputs=2)); controller=LearningController(m,tmp_path)
    path=tmp_path/'plastic.npz'; m.save(path)
    x,y=representation('order',23,1); m.forward(x[0],track=True); m.learn(int(y[0]))
    restored=controller.rollback(path)
    assert isinstance(restored,PlasticCore) and restored.updates==0


def test_label_is_not_in_encoding():
    sample=normalize_sample({'task':'A','signal':2}); a=encode(sample)
    sample['label']=1-expected_label(sample)
    np.testing.assert_array_equal(a,encode(sample))


def test_http_origin_limits_and_static_files(app):
    server=make_server(app,0); thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def send(path,data=None,headers=None):
        req=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),headers=headers or {})
        return urllib.request.urlopen(req,timeout=10)
    try:
        assert b'Neural Workbench' in send('/').read()
        status=json.load(send('/api/status'))
        with pytest.raises(urllib.error.HTTPError) as e: send('/api/teach',{'task':'A'})
        assert e.value.code==403
        headers={'Content-Type':'application/json','X-SCARX-Token':status['token'],'Origin':base}
        assert json.load(send('/api/predict',{'task':'A','signal':0},headers))['retrieval_used'] is False
        with pytest.raises(urllib.error.HTTPError) as e: send('/api/checkpoint',{},dict(headers,Origin='https://example.com'))
        assert e.value.code==403
        with pytest.raises(urllib.error.HTTPError) as e: send('/../../pyproject.toml')
        assert e.value.code==404
        rows=json.load(send('/api/checkpoints'))
        assert send('/api/export/'+rows[0]['id']).read()==app._payload(rows[0]['id'])
    finally: server.shutdown(); server.server_close(); thread.join()


def test_interrupted_process_retains_committed_generation(tmp_path):
    root=str(tmp_path/'interrupted')
    code="""
import os,sys
from scarx.app.state import Workbench
a=Workbench(sys.argv[1]); a.teach({'task':'A','count':16,'epochs':2})
print(a.active,flush=True)
original=a._commit
def crash(action,details):
    original(action,details)
    os._exit(77)
a._commit=crash
a.teach({'task':'B','count':16,'epochs':2})
"""
    result=subprocess.run([sys.executable,'-c',code,root],capture_output=True,text=True,timeout=30)
    assert result.returncode==77,result.stderr
    recovered=Workbench(root)
    try:
        assert recovered.active==result.stdout.strip()
        assert recovered.learner.model.updates==32 and recovered.replay.size==16
        assert recovered.db.execute('SELECT count(*) FROM episodes').fetchone()[0]==16
        assert len(recovered.checkpoints())==2
    finally: recovered.close()
