"""Technical fixtures only: no dataset training, DEV or confirmation bank."""
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.evaluation.literature_eprop import LiteratureEPropALIF as EProp
from scarx.evaluation.literature_braintrace import BrainTraceDRTRL as DRTRL
from scarx.evaluation.forensic.r8_snapshot import encoded,digest,metadata,loads


def config():return Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=971)
def fixture():return np.random.default_rng(972).uniform(-.2,1.2,(2,15,8)).astype(np.float32)
def model(kind,condition='learned_mtsg'):
    arm={'learned_mtsg':'multi_gated','shuffled_signal':'multi_gated_shuffle','readout_only':'multi_gated_readout_only','frozen_hidden':'multi_gated_frozen'}
    return DRTRL(config(),condition) if kind=='drtrl' else EProp(config(),kind,arm[condition])


@pytest.mark.parametrize('kind',['eprop_matched','eprop_aligned','drtrl'])
@pytest.mark.parametrize('condition',['learned_mtsg','shuffled_signal','readout_only','frozen_hidden'])
def test_full_state_mid_episode_roundtrip_and_next_update(kind,condition):
    a=model(kind,condition);a.calibrated=True;xs=fixture()
    a.learn_episode(xs[0],1);a.reset_episode()
    for row in xs[1,:7]:a.step(row)
    blob=a.dumps();b=type(a).loads(blob)
    assert b.dumps()==blob
    for row in xs[1,7:]:
        a.step(row);b.step(row)
    assert a.dumps()==b.dumps()
    assert a.learn_episode(xs[0],0)==b.learn_episode(xs[0],0)
    assert a.dumps()==b.dumps()
    if condition in ('readout_only','frozen_hidden'):
        np.testing.assert_array_equal(a.core.w_in,a.w_in_0)
        np.testing.assert_array_equal(a.core.w_rec,a.w_rec_0)
    # Rollback means replacing the complete object with the saved state.
    rolled=type(a).loads(blob)
    assert rolled.dumps()==blob and rolled.dumps()!=a.dumps()


@pytest.mark.parametrize('kind',['eprop_matched','eprop_aligned','drtrl'])
def test_cold_process_resume_exact_byte_sha(kind):
    a=model(kind,'shuffled_signal');a.calibrated=True;xs=fixture()
    a.learn_episode(xs[0],1);a.reset_episode()
    for row in xs[1,:7]:a.step(row)
    blob=a.dumps()
    for row in xs[1,7:]:a.step(row)
    a.learn_episode(xs[0],0)
    expected=hashlib.sha256(a.dumps()).hexdigest()
    program='''import base64,hashlib,json,sys,numpy as np
from scarx.evaluation.literature_eprop import LiteratureEPropALIF
from scarx.evaluation.literature_braintrace import BrainTraceDRTRL
p=json.load(sys.stdin)
cls=BrainTraceDRTRL if p['kind']=='drtrl' else LiteratureEPropALIF
m=cls.loads(base64.b64decode(p['blob']))
for row in np.asarray(p['suffix'],np.float32):m.step(row)
m.learn_episode(np.asarray(p['next'],np.float32),0)
print(hashlib.sha256(m.dumps()).hexdigest())
'''
    env=os.environ.copy();root=Path(__file__).resolve().parents[1]
    env['PYTHONPATH']=str(root/'src')
    result=subprocess.run([sys.executable,'-X','utf8','-c',program],input=json.dumps({
        'kind':kind,'blob':base64.b64encode(blob).decode('ascii'),'suffix':xs[1,7:].tolist(),'next':xs[0].tolist()}),
        capture_output=True,text=True,encoding='utf-8',env=env,cwd=root,timeout=20,check=True)
    assert result.stdout.strip()==expected


def rewrite(blob,change):
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        fields={name:archive.read(name) for name in archive.namelist()}
    meta=metadata(fields['metadata.json']);change(fields,meta)
    meta.pop('metadata_sha256')
    meta['sha256']={name:digest(raw) for name,raw in fields.items() if name!='metadata.json'}
    meta['metadata_sha256']=digest(encoded(meta));fields['metadata.json']=encoded(meta)
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive:
        for name,raw in fields.items():archive.writestr(name,raw)
    return stream.getvalue()


@pytest.mark.parametrize('kind',['eprop_matched','eprop_aligned','drtrl'])
def test_policy_mechanism_and_float64_tampering_rejected_without_mutation(kind):
    a=model(kind);blob=a.dumps()
    def gamma(fields,meta):
        raw=io.BytesIO();np.save(raw,np.full((48,1),.5,np.float32),allow_pickle=False)
        fields['model__gamma_e_vec.npy']=raw.getvalue()
    def policy(fields,meta):meta['settings']['gate_mode']='activity' if meta['settings']['gate_mode']=='none' else 'none'
    def mechanism(fields,meta):meta['mechanism_sha256']='0'*64
    def dtype(fields,meta):
        raw=io.BytesIO();np.save(raw,np.zeros((48,16),np.float64),allow_pickle=False)
        fields['model__e_v.npy']=raw.getvalue()
    for mutation in (gamma,policy,mechanism,dtype):
        with pytest.raises(ValueError):loads(a,rewrite(blob,mutation))
        assert a.dumps()==blob
    with pytest.raises(ValueError):type(a).loads(blob[:-5])
    assert a.dumps()==blob


def test_reference_kinds_and_eprop_variants_cannot_cross_load():
    a=model('eprop_matched');b=model('eprop_aligned');c=model('drtrl')
    for target,source in ((a,b),(b,a),(c,a),(a,c)):
        with pytest.raises(ValueError):loads(target,source.dumps())


def test_isolated_full_surrogate_oracle_against_dense_state_jacobian():
    """Independent four-state dense recurrence, no learning or reverse pass.

    Small topology is a numerical oracle fixture, never an H48 quality score.
    It checks recurrent mixing paths rather than only the zero-recurrence case.
    """
    from scarx.neural.core import SpikingCore
    from scarx.evaluation.literature_braintrace import full_surrogate_sensitivity
    c=Config(inputs=2,outputs=2,hidden=3,fan_in=2,plastic_rows=3,seed=973)
    core=SpikingCore(c);xs=np.random.default_rng(974).uniform(-.2,1.3,(9,2)).astype(np.float32)
    result=full_surrogate_sensitivity(core,xs)
    m=copy.deepcopy(core);m.reset_state();h=c.hidden;k=c.inputs+c.fan_in;p=h*k
    recurrence=np.zeros((h,h),np.float64)
    for j in range(h):recurrence[j,m.indices[j]]=m.w_rec[j]
    sensitivity=np.zeros((4*h,p),np.float64);ident=np.eye(h)
    for row in xs:
        old=m.spikes.copy();theta=c.threshold+c.adaptation_strength*m.adaptation
        u=(c.leak*m.voltage+np.sum(m.w_rec*old[m.indices],axis=1))+m.w_in@row
        raw=(u-theta)/.4;sigmoid=1/(1+np.exp(-np.clip(raw,-20,20)))
        psi=(sigmoid*(1-sigmoid)/.4*(np.abs(raw)<20)).astype(np.float32).astype(np.float64)
        pre=np.concatenate((np.broadcast_to(row,(h,c.inputs)),old[m.indices]),axis=1)
        injection=np.zeros((h,p))
        for j in range(h):injection[j,j*k:(j+1)*k]=pre[j]
        current_jac=np.column_stack((c.leak*ident,np.zeros((h,h)),recurrence,np.zeros((h,h))))
        margin_jac=current_jac.copy();margin_jac[:,h:2*h]-=c.adaptation_strength*ident
        spike_jac=psi[:,None]*margin_jac
        m.step(row,track=False)
        voltage_jac=current_jac-theta[:,None]*spike_jac
        voltage_jac[:,h:2*h]-=m.spikes[:,None]*c.adaptation_strength*ident
        adaptation_jac=spike_jac.copy();adaptation_jac[:,h:2*h]+=c.adaptation_decay*ident
        feature_jac=(1-c.feature_decay)*spike_jac;feature_jac[:,3*h:]+=c.feature_decay*ident
        jac=np.vstack((voltage_jac,adaptation_jac,spike_jac,feature_jac))
        spike_injection=psi[:,None]*injection
        direct=np.vstack((injection-theta[:,None]*spike_injection,
                             spike_injection,spike_injection,(1-c.feature_decay)*spike_injection))
        sensitivity=jac@sensitivity+direct
    np.testing.assert_allclose(result,sensitivity[3*h:],rtol=2e-7,atol=2e-8)
