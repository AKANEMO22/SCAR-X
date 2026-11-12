"""Model-state engineering fixtures on pinned peer source; not quality training."""
from pathlib import Path
import hashlib,importlib.util,io,json,subprocess,sys,zipfile
import numpy as np
import pytest
from scarx.evaluation.forensic import r8_snapshot as S
from scarx.neural.core import Config
from scarx.neural.unified import UnifiedCore

ROOT=Path(__file__).resolve().parents[1]
PEER=ROOT.parent/'review-gemini-r8-fe472b3'


@pytest.fixture(scope='module')
def classes():
    assert subprocess.check_output(['git','-C',str(PEER),'rev-parse','HEAD'],text=True).strip()==S.SOURCE
    # Load two named research modules only. No app/main import, candidate change or campaign.
    for filename in ('r8_mtsg_eprop','r8_bf_cascade'):
        name='scarx.neural.'+filename
        if name not in sys.modules:
            spec=importlib.util.spec_from_file_location(name,PEER/'src/scarx/neural'/(filename+'.py'))
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return (sys.modules['scarx.neural.r8_mtsg_eprop'].MultiTimescaleSensoryGatedEProp,
            sys.modules['scarx.neural.r8_bf_cascade'].BennaFusiMetaplasticCascade)


def fixture_model(classes,index=0):
    cfg=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
    if index==0:model=classes[0](cfg,'shuffled_signal',sensory_threshold=.123)
    else:model=classes[1](cfg,use_cascade=True,use_context_gating=True,kappa1=.003,kappa2=.0007,sensory_threshold=.123)
    # A single explicit technical fixture, not a dataset or skill score.
    model.calibrated=True
    if index==0:model.signal_rng.permutation(48)
    model.core.spikes[:]=1;model.step(np.full(8,.03,np.float32),track=True)
    return model


@pytest.mark.parametrize('index',[0,1])
def test_full_roundtrip_exact_including_parameters_rng_and_controller(classes,index,tmp_path):
    model=fixture_model(classes,index);controller={'phase':'pending','position':1,'replay':[{'fixture':'p0'}],'rng':np.random.default_rng(919).bit_generator.state}
    raw=S.dumps(model,controller);loaded,state=S.loads(fixture_model(classes,index),raw)
    assert S.dumps(loaded,state)==raw and state==controller and loaded.sensory_threshold==.123
    if index==0:assert loaded.signal_rng.bit_generator.state==model.signal_rng.bit_generator.state
    path=tmp_path/'research.zip';S.save(model,path,controller)
    loaded,state=S.load(fixture_model(classes,index),path);assert S.dumps(loaded,state)==raw
    with pytest.raises(ValueError):UnifiedCore.loads(raw)


@pytest.mark.parametrize('index',[0,1])
def test_next_local_fixture_update_and_rollback_are_exact(classes,index):
    model=fixture_model(classes,index);raw=S.dumps(model)
    restored,_=S.loads(fixture_model(classes,index),raw)
    seq=np.full((3,8),.03,np.float32);seq[0,0]=1.5
    assert model.learn_episode(seq,1)==restored.learn_episode(seq,1)
    assert S.dumps(model)==S.dumps(restored)
    rolled,_=S.loads(model,raw);assert S.dumps(rolled)==raw
    if index==0:assert model.signal_rng.integers(10000)==restored.signal_rng.integers(10000)


def rewrite(raw,name,changed):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:members={n:z.read(n) for n in z.namelist()}
    meta=json.loads(members.pop('metadata.json'));meta.pop('metadata_sha256')
    members[name]=changed;meta['sha256'][name]=S.digest(changed)
    meta['metadata_sha256']=S.digest(S.encoded(meta));members['metadata.json']=S.encoded(meta)
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for key,value in members.items():z.writestr(key,value)
    return out.getvalue()


@pytest.mark.parametrize('kind',['float64','nan','wrongshape'])
def test_corrupt_state_rejected_without_changing_template(classes,kind):
    model=fixture_model(classes);raw=S.dumps(model);before=S.dumps(model)
    array=model.e_f.copy()
    if kind=='float64':array=array.astype(np.float64)
    elif kind=='nan':array.fill(float('nan'))
    else:array=array[:1]
    stream=io.BytesIO();np.save(stream,array,allow_pickle=False)
    corrupt=rewrite(raw,'model__e_f.npy',stream.getvalue())
    with pytest.raises(ValueError):S.loads(model,corrupt)
    assert S.dumps(model)==before


def test_huge_npy_header_rejected_before_any_array_load(classes,monkeypatch):
    model=fixture_model(classes);header=io.BytesIO()
    np.lib.format.write_array_header_1_0(header,{'descr':'<f4','fortran_order':False,'shape':(10**9,10**9)})
    corrupt=rewrite(S.dumps(model),'model__e_f.npy',header.getvalue())
    def forbidden(*args,**kwargs):raise AssertionError('np.load reached before header validation')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError,match='before allocation'):S.loads(model,corrupt)


def test_duplicate_members_and_truncated_payload_rejected(classes):
    model=fixture_model(classes);raw=S.dumps(model);out=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as z,zipfile.ZipFile(out,'w') as new:
        for n in z.namelist():new.writestr(n,z.read(n))
        new.writestr('metadata.json',z.read('metadata.json'))
    for corrupt in (out.getvalue(),raw[:-20]):
        with pytest.raises(ValueError):S.loads(model,corrupt)


def test_foreign_state_not_silently_dropped(classes):
    model=fixture_model(classes);model.optimizer=np.zeros(1)
    with pytest.raises(ValueError,match='Unknown learner state'):S.dumps(model)


def test_checksum_tamper_fails_before_commit(classes):
    model=fixture_model(classes);raw=S.dumps(model);out=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as old,zipfile.ZipFile(out,'w') as new:
        for name in old.namelist():
            blob=old.read(name)
            if name=='metadata.json':
                meta=json.loads(blob);meta['settings']['sensory_threshold']=.5;blob=S.encoded(meta)
            new.writestr(name,blob)
    before=S.dumps(model)
    with pytest.raises(ValueError,match='checksum'):S.loads(model,out.getvalue())
    assert S.dumps(model)==before


@pytest.mark.parametrize('controller',[{'replay':(1,2)},{'cursor':{1:'x'}},{'counter':np.int64(1)}])
def test_controller_types_cannot_be_silently_coerced(classes,controller):
    with pytest.raises(ValueError,match='lossless JSON'):S.dumps(fixture_model(classes),controller)


def test_failed_atomic_replace_preserves_committed_checkpoint(classes,tmp_path,monkeypatch):
    model=fixture_model(classes);path=tmp_path/'committed.zip';S.save(model,path);old=path.read_bytes()
    model.step(np.full(8,.05,np.float32),True)
    def reject_replace(*args):raise OSError('Injected replacement failure')
    monkeypatch.setattr(S.os,'replace',reject_replace)
    with pytest.raises(OSError,match='Injected'):S.save(model,path)
    assert path.read_bytes()==old and list(tmp_path.iterdir())==[path]
