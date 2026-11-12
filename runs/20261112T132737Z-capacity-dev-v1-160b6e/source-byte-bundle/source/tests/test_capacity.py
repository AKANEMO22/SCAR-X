import copy,io,json,subprocess,sys,zipfile
import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.evaluation.capacity import CapacityOwn,CapacityACR
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.acr import AdaptiveCreditRouting
from scarx.evaluation.forensic import r8_snapshot as codec

def cfg(h=48):return Config(inputs=8,outputs=2,hidden=h,fan_in=8,plastic_rows=12,seed=6291)
def xs():return np.random.default_rng(6292).uniform(-.15,1.1,(8,19,8)).astype(np.float32)

@pytest.mark.parametrize('cls,baseline,condition,arm',[
    (CapacityOwn,MatchedLocalCredit,'learned_mtsg','multi_gated'),
    (CapacityOwn,MatchedLocalCredit,'readout_only','multi_gated_readout_only'),
    (CapacityACR,AdaptiveCreditRouting,'learned_mtsg','groups_up'),
    (CapacityACR,AdaptiveCreditRouting,'shuffled_signal','groups_up'),
])
def test_h48_exact_complete_learner_parity(cls,baseline,condition,arm):
    a=cls(cfg(),condition)
    b=baseline(cfg(),arm,condition) if baseline is AdaptiveCreditRouting else baseline(cfg(),arm)
    a.calibrate(xs()[:3]);b.calibrate(xs()[:3])
    for i,x in enumerate(xs()):
        assert a.learn_episode(x,i%2)==b.learn_episode(x,i%2)
        for name,array in a.core.arrays().items():np.testing.assert_array_equal(array,b.core.arrays()[name])
        for name,array in vars(a).items():
            if isinstance(array,np.ndarray):np.testing.assert_array_equal(array,getattr(b,name))
        assert a.signal_rng.bit_generator.state==b.signal_rng.bit_generator.state
        assert a.total_ticks==b.total_ticks and a.hidden_updates==b.hidden_updates
    assert a.memory_bytes()==b.memory_bytes()

@pytest.mark.parametrize('h',[48,64,128])
@pytest.mark.parametrize('cls',[CapacityOwn,CapacityACR])
@pytest.mark.parametrize('condition',['learned_mtsg','readout_only','shuffled_signal'])
def test_all_sizes_full_checkpoint_next_update_and_rollback(h,cls,condition,tmp_path):
    m=cls(cfg(h),condition);m.calibrate(xs()[:2]);m.learn_episode(xs()[0],1)
    payload=m.dumps();n=cls.loads(payload)
    assert n.dumps()==payload
    assert m.learn_episode(xs()[1],0)==n.learn_episode(xs()[1],0)
    assert m.dumps()==n.dumps()
    codec.save(n,tmp_path/'checkpoint.zip',{'replay_indices':[0,1],'status':'technical-fixture'})
    loaded,controller=codec.load(cls(cfg(h),condition),tmp_path/'checkpoint.zip')
    assert loaded.dumps()==n.dumps() and controller['replay_indices']==[0,1]
    # Saved rollback snapshot replaces the complete learner, not weights alone.
    changed=copy.deepcopy(n);changed.learn_episode(xs()[2],1)
    rolled=cls.loads(n.dumps());assert rolled.dumps()==n.dumps()

@pytest.mark.parametrize('h',[64,128])
def test_same_h_initialization_and_neural_forward_invariant(h):
    models=[CapacityOwn(cfg(h)),CapacityOwn(cfg(h),'readout_only'),CapacityACR(cfg(h)),CapacityACR(cfg(h),'shuffled_signal')]
    for model in models[1:]:
        for key,value in models[0].core.arrays().items():np.testing.assert_array_equal(value,model.core.arrays()[key])
    for row in xs()[0]:
        for model in models:model.step(row)
        for model in models[1:]:
            for key in ('voltage','adaptation','spikes','features'):np.testing.assert_array_equal(getattr(models[0].core,key),getattr(model.core,key))
    assert all(np.isin(m.core.spikes,[0,1]).all() for m in models)
    assert models[0].e_v.shape==(h,16) and models[2].e_f_stop.shape==(h,16)
    assert models[0].memory_bytes()==596*h+864
    assert models[2].memory_bytes()==944*h+864

@pytest.mark.parametrize('h',[64,128])
def test_cold_process_resume_actual_python(h):
    m=CapacityACR(cfg(h),'shuffled_signal');m.calibrated=True
    for row in xs()[0][:7]:m.step(row)
    blob=m.dumps();m.learn_episode(xs()[1],1);expected=m.dumps()
    script="""import sys,base64,numpy as np
from scarx.evaluation.capacity import CapacityACR
m=CapacityACR.loads(base64.b64decode(sys.stdin.buffer.readline()))
x=np.random.default_rng(6292).uniform(-.15,1.1,(8,19,8)).astype(np.float32)
m.learn_episode(x[1],1)
sys.stdout.buffer.write(m.dumps())
"""
    import base64,os
    env=dict(os.environ,PYTHONPATH=str(__import__('pathlib').Path(__file__).resolve().parents[1]/'src'))
    result=subprocess.run([sys.executable,'-c',script],input=base64.b64encode(blob)+b'\n',capture_output=True,check=True,env=env)
    assert result.stdout==expected

def mutate(payload,name,value):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:files={n:z.read(n) for n in z.namelist()}
    meta=codec.metadata(files['metadata.json']);meta.pop('metadata_sha256')
    buf=io.BytesIO();np.save(buf,value,allow_pickle=False);files[name]=buf.getvalue();meta['sha256'][name]=codec.digest(files[name])
    meta['metadata_sha256']=codec.digest(codec.encoded(meta));files['metadata.json']=codec.encoded(meta)
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for k,v in files.items():z.writestr(k,v)
    return out.getvalue()

def test_loader_rejects_wrong_shape_dtype_indices_policy_and_unknown_state():
    m=CapacityACR(cfg(128));blob=m.dumps()
    for name,value in [('model__e_f.npy',np.zeros((48,16),np.float32)),
                       ('model__e_f.npy',np.zeros((128,16),np.float64)),
                       ('core__indices.npy',np.full((128,8),128,np.int32)),
                       ('model__gamma_e_vec.npy',np.full((128,1),.5,np.float32))]:
        with pytest.raises(ValueError):CapacityACR.loads(mutate(blob,name,value))
    with pytest.raises(ValueError):CapacityOwn.loads(blob)
    m.new_optimizer_state=np.zeros(3,np.float32)
    with pytest.raises(ValueError):m.dumps()

def test_nonapproved_sizes_rejected_before_array_allocation():
    for h in (1,256,1000000):
        with pytest.raises(ValueError):CapacityOwn(cfg(h))
