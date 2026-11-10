import hashlib
import io
import json
from pathlib import Path
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import zipfile
import numpy as np
import pytest
from scarx.neural.unified import UnifiedCore
from scarx.app.generation import Generation, MODEL, ROOT
from scarx.app.generation_server import make_server


@pytest.fixture
def app(tmp_path):
    session=Generation(tmp_path,background=False)
    yield session
    session.close()


def test_one_core_for_both_modalities():
    model=UnifiedCore(['<eos>','<bos>','hello','.'])
    core=model.core; recurrent=core.w_rec
    original=core.fingerprint()
    model.train_pair('hi','hello.')
    after_text=core.fingerprint(); text_rec=recurrent.copy()
    model.image(np.arange(64,dtype=np.float32)%17,3)
    assert model.core is core and core.w_rec is recurrent
    assert original!=after_text!=core.fingerprint()
    assert np.linalg.norm(core.w_rec-text_rec)>0
    # Text and digit slices sit in the same readout; one recurrence, one state.
    assert core.w_out.shape==(14,48) and core.config.hidden==48


def test_inference_and_idle_do_not_reset_neurons_or_update_weights(monkeypatch):
    model=UnifiedCore(['<eos>','<bos>','hello','.'])
    def forbidden():raise AssertionError('Runtime reset_state called')
    monkeypatch.setattr(model.core,'reset_state',forbidden)
    before=model.core.fingerprint(); voltage=model.core.voltage.copy()
    model.generate('hi'); model.image(np.ones(64,np.float32)*8)
    assert model.ticks>0 and not np.array_equal(voltage,model.core.voltage)
    assert model.core.fingerprint()==before


def test_neural_exact_continuation_and_corruption():
    model=UnifiedCore(['<eos>','<bos>','hello','.'])
    model.train_pair('hi','hello.'); model.tick()
    clone=UnifiedCore.loads(model.dumps())
    assert clone.generate('hi')==model.generate('hi')
    assert clone.image(np.ones(64,np.float32)*8)==model.image(np.ones(64,np.float32)*8)
    for name,value in model.core.arrays().items():np.testing.assert_array_equal(value,clone.core.arrays()[name])
    output=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(model.dumps())) as source,zipfile.ZipFile(output,'w') as target:
        for name in source.namelist():
            data=source.read(name)
            if name=='metadata.json':
                meta=json.loads(data);meta['sha256']='0'*64;data=json.dumps(meta).encode()
            target.writestr(name,data)
    with pytest.raises(ValueError,match='checksum'):UnifiedCore.loads(output.getvalue())


def test_idle_trace_real_and_bounded_without_learning(app):
    before=app.model.core.fingerprint(); updates=app.model.core.updates
    app.state='idle'
    for _ in range(350):app.idle_once()
    assert app.model.core.fingerprint()==before and app.model.core.updates==updates
    assert app.model.ticks>350 and len(app.trace)<=256
    assert any(r['kind']=='probe_result' and r['verified'] is False for r in app.trace)
    for event in app.trace:
        if 'spikes' in event:
            assert len(event['spikes'])==len(event['voltage'])==48
            assert event['spiking']==sum(event['spikes']) and event['computed']==48
    assert app.status()['cores_loaded']==1


def test_whole_session_restart_resume_and_same_directory_lock(app,tmp_path):
    app.chat('hi');app.idle_once();app.rng.random(3)
    checkpoint=app.checkpoint()['checkpoint'];payload=app._pack()
    with pytest.raises(RuntimeError,match='đang'):Generation(tmp_path,background=False)
    app.close()
    restarted=Generation(tmp_path,background=False)
    try:
        assert restarted.active==checkpoint and restarted._pack()==payload
        expected=UnifiedCore.loads(app.model.dumps()).generate('hi')
        assert restarted.chat('hi')['text']==expected['text']
    finally:restarted.close()


def test_pending_probe_and_rng_resume(app,tmp_path):
    while app.probe is None:app.idle_once()
    app.idle_once(); payload=app._pack()
    replica=Generation(tmp_path/'replica',background=False)
    try:
        replica._unpack(payload)
        assert replica.probe==app.probe
        for _ in range(12):app.idle_once();replica.idle_once()
        assert app._pack()==replica._pack()
        assert app.rng.random()==replica.rng.random()
    finally:replica.close()


def test_teaching_failure_rolls_back_full_session_and_db(app,monkeypatch):
    before=app._pack();count=len(app.checkpoints())
    def fail():raise sqlite3.OperationalError('fault injection')
    monkeypatch.setattr(app,'_persist',fail)
    with pytest.raises(sqlite3.OperationalError):app.teach({'prompt':'hi','answer':'Hello.','repeats':2,'confirmation':True})
    assert app._pack()==before and len(app.checkpoints())==count


def test_teaching_explicit_and_restore(app):
    initial=app.checkpoint()['checkpoint'];before=app.model.core.fingerprint()
    with pytest.raises(ValueError):app.teach({'prompt':'hi','answer':'Hello.','verified':True})
    result=app.teach({'prompt':'hi','answer':'Hello.','repeats':2,'confirmation':True})
    assert result['verified'] is False and result['weight_delta']['w_out']>0
    assert app.model.core.fingerprint()!=before
    app.restore(initial)
    assert app.model.core.fingerprint()==before
    assert len(app.checkpoints())>=4


def test_foreground_pending_preempts_idle_scheduler(tmp_path):
    app=Generation(tmp_path)
    try:
        observed=threading.Event()
        original=app.model.on_tick
        def tick(*args):original(*args);observed.set()
        app.model.on_tick=tick
        assert observed.wait(2)
        with app.lock:
            app.pending=1;before=app.model.ticks
        # Event wait, shorter than test timeout; scheduler sees pending even if
        # it obtains session lock, and must not advance idle state.
        time.sleep(.16)
        with app.lock:
            assert app.model.ticks==before
            app.pending=0
        app.chat('hi')
        assert any(r['mode']=='foreground' for r in app.trace)
    finally:app.close()


@pytest.fixture
def endpoint(app):
    server=make_server(app,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield app,'http://127.0.0.1:'+str(server.server_port)
    server.shutdown();server.server_close();thread.join()


def request(url,data=None,token=None):
    headers={'Content-Type':'application/json'}
    if token:headers['X-SCARX-Token']=token
    return urllib.request.urlopen(urllib.request.Request(url,None if data is None else json.dumps(data).encode(),headers=headers),timeout=10)


def test_http_models_sdk_stream_and_scope(endpoint):
    app,base=endpoint
    values=json.load(request(base+'/v1/models'))['data'];assert [x['id'] for x in values]==[MODEL]
    openai=pytest.importorskip('openai')
    with openai.OpenAI(base_url=base+'/v1',api_key='no-key-required') as client:
        messages=[{'role':'user','content':'hi'}]
        result=client.chat.completions.create(model=MODEL,messages=messages)
        chunks=list(client.chat.completions.create(model=MODEL,messages=messages,stream=True))
        assert result.choices[0].message.content
        assert all('reasoning_content' not in c.choices[0].delta.model_dump(exclude_none=True) for c in chunks)
        assert chunks[-1].choices[0].finish_reason=='stop'
        assert result.scarx['shared_core'] is True
    cap=json.load(request(base+'/v1/capabilities'))
    assert cap['vision_scope']=='handwritten_digits_only' and cap['learned_reasoning'] is False


def test_legacy_and_invalid_input_rejected_without_weight_updates(endpoint):
    app,base=endpoint;before=app.model.core.fingerprint()
    for body in [
        {'model':'scarx-language-lab-v0.2','messages':[{'role':'user','content':'hi'}]},
        {'model':MODEL,'messages':[None]},
        {'model':MODEL,'messages':[{'role':'user','content':[]}]},
        {'model':MODEL,'tool_choice':'required','messages':[{'role':'user','content':'hi'}]},
    ]:
        with pytest.raises(urllib.error.HTTPError):request(base+'/v1/chat/completions',body)
    assert before==app.model.core.fingerprint()
    with pytest.raises(urllib.error.HTTPError) as error:request(base+'/api/chat',{'prompt':'hi'})
    assert error.value.code==403


def test_image_uses_same_core_through_http(endpoint):
    pytest.importorskip('PIL')
    app,base=endpoint;identity=id(app.model.core)
    sample=json.loads((ROOT/'artifacts/pilots/vision-examples.json').read_text(encoding='utf-8'))[0]
    body={'model':MODEL,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':sample['image']}}]}]}
    result=json.load(request(base+'/v1/chat/completions',body));classification=json.loads(result['choices'][0]['message']['content'])
    assert 0<=classification['digit']<=9 and len(classification['probabilities'])==10
    assert id(app.model.core)==identity and app.status()['cores_loaded']==1


def test_http_pause_resume_and_checkpoint(endpoint):
    app,base=endpoint;token=json.load(request(base+'/api/status'))['token']
    result=json.load(request(base+'/api/idle',{'enabled':False,'hz':5},token))
    assert result['enabled'] is False and result['unverified_self_training'] is False
    checkpoint=json.load(request(base+'/api/checkpoint',{},token))['checkpoint']
    assert checkpoint==app.active
    export=request(base+'/api/export').read();assert zipfile.is_zipfile(io.BytesIO(export))
