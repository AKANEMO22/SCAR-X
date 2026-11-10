import base64
import hashlib
import io
import json
import sqlite3
import numpy as np
import pytest
from scarx import Config,SpikingCore
from scarx.neural.language import WordLanguageCore
from scarx.app.state import Workbench
from scarx.app.openai_backend import chat,models
from scarx.app.pilots import LANGUAGE,VISION


@pytest.fixture
def lab(tmp_path):
    root=tmp_path/'source';directory=root/'language';directory.mkdir(parents=True)
    model=WordLanguageCore(['<eos>','<bos>','chào','bạn','.'],config_overrides={'readout_lr':.015,'weight_clip':20.})
    for _ in range(10):model.train_pair('xin chào','chào bạn.')
    model.save(directory)
    vision=SpikingCore(Config(inputs=64,outputs=10,hidden=48,plastic_rows=12));vision.save(root/'vision.npz')
    (root/'artifacts').mkdir()
    (root/'artifacts/pilot-release.json').write_text(json.dumps({'language':{'checkpoint':'language','quality_gate_passed':False},'vision':{'checkpoint':'vision.npz','scope':'digits-only'}}),encoding='utf-8')
    app=Workbench(tmp_path/'state');app.pilots.root=root
    try:yield app,app.pilots
    finally:app.close()


def image_uri():
    from PIL import Image
    a=np.zeros((8,8),dtype=np.uint8);a[:,3:5]=255
    stream=io.BytesIO();Image.fromarray(a).save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def test_pilot_model_catalog_truthful(lab):
    app,pilot=lab
    catalog={m['id']:m for m in models(app)['data']}
    assert LANGUAGE in catalog and VISION in catalog
    assert catalog[LANGUAGE]['capabilities']['vision'] is False
    assert 'FAILED' in catalog[LANGUAGE]['description']
    assert 'NOT screenshot' in catalog[VISION]['description']


def test_real_language_teaching_expansion_and_restore(lab):
    app,pilot=lab
    original=pilot._load('language').dumps();first=pilot.checkpoints()[0]['id'];base=app._pack()
    result=pilot.teach_language({'prompt':'xin chào','answer':'chào bạn mới.','confirmation':True,'repeats':8})
    assert result['weights_changed'] and result['new_vocabulary_tokens']==1
    assert pilot._load('language').dumps()!=original and app._pack()==base
    assert app.store.get(1)['verified']==0
    assert pilot.language('xin chào')['quality_gate_passed'] is False
    pilot.restore(first)
    assert pilot._load('language').dumps()==original


def test_inference_preserves_full_pilot_state(lab):
    _,pilot=lab
    model=pilot._load('language');before=model.dumps();pilot.language('xin chào');assert model.dumps()==before
    vision=pilot._load('vision');before=vision.dumps();result=pilot.vision({'image':image_uri()})
    assert result['digit']==int(np.argmax(result['probabilities']))
    assert vision.dumps()==before and result['scope'].startswith('digits-only')


def test_language_commit_fault_and_restart(lab,monkeypatch):
    app,pilot=lab
    original=pilot._load('language').dumps();prior_count=len(pilot.checkpoints());commit=pilot._commit
    def fail(*args):
        commit(*args);raise sqlite3.OperationalError('simulated fault')
    monkeypatch.setattr(pilot,'_commit',fail)
    with pytest.raises(sqlite3.OperationalError):pilot.teach_language({'prompt':'xin chào','answer':'chào bạn.','confirmation':True,'repeats':2})
    assert pilot._load('language').dumps()==original and len(pilot.checkpoints())==prior_count
    directory=app.directory;root=pilot.root;app.close();restored=Workbench(directory);restored.pilots.root=root
    try:assert restored.pilots._load('language').dumps()==original
    finally:restored.close()


def test_hermes_pilot_text_and_digit_are_actual_neural_outputs(lab):
    app,pilot=lab
    expected=pilot.language('xin chào')['text']
    result,_=chat(app,{'model':LANGUAGE,'messages':[{'role':'user','content':'xin chào'}]})
    assert result['choices'][0]['message']['content']==expected
    assert result['scarx']['quality_gate_passed'] is False
    response,_=chat(app,{'model':VISION,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':image_uri()}}]}]})
    assert json.loads(response['choices'][0]['message']['content'])['scope']=='digits-only; not screenshot/OCR/visualQA'


def test_feedback_requires_explicit_annotation_and_bad_image_rejected(lab):
    _,pilot=lab
    with pytest.raises(ValueError):pilot.teach_language({'prompt':'hi','answer':'hello'})
    with pytest.raises(ValueError):pilot.vision({'image':'https://remote.example/image.png'})
