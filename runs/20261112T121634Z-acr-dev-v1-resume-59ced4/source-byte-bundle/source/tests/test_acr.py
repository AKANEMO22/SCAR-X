import copy,io,json,zipfile
import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit
from scarx.evaluation.acr import AdaptiveCreditRouting as ACR,NormMismatch,ray_to_radius,matched_update

def cfg():return Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
def sequences():return np.random.default_rng(108).uniform(-.1,1.5,(12,21,8)).astype(np.float32)

def compare(a,b,endpoint):
    for name in a.core.arrays():np.testing.assert_array_equal(a.core.arrays()[name],b.core.arrays()[name])
    for name in ('syn_trace','syn_energy','f_mean','f_var'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    stream=('e_v','e_a','e_f_own') if endpoint==0 else ('e_v_stop','e_a_stop','e_f_stop')
    for an,bn in zip(stream,('e_v','e_a','e_f')):np.testing.assert_array_equal(getattr(a,an),getattr(b,bn))
    np.testing.assert_array_equal(a.e_f,b.e_f)
    assert a.signal_rng.bit_generator.state==b.signal_rng.bit_generator.state
    for name in ('episodes_learned','hidden_updates','total_ticks'):assert getattr(a,name)==getattr(b,name)

@pytest.mark.parametrize('endpoint,cls',[(0,MatchedLocalCredit),(100,StoppedResetCredit)])
@pytest.mark.parametrize('condition,arm',[('learned_mtsg','multi_gated'),('shuffled_signal','multi_gated_shuffle'),('readout_only','multi_gated_readout_only')])
def test_exact_endpoint_full_optimizer_resume(endpoint,cls,condition,arm):
    a=ACR(cfg(),'static_'+str(endpoint),condition);b=cls(cfg(),arm)
    xs=sequences();a.calibrate(xs[:3]);b.calibrate(xs[:3])
    for i,x in enumerate(xs[:6]):
        assert a.learn_episode(x,i%2)==b.learn_episode(x,i%2)
        compare(a,b,endpoint)
    a=ACR.loads(a.dumps());b=cls.loads(b.dumps())
    assert a.learn_episode(xs[7],1)==b.learn_episode(xs[7],1);compare(a,b,endpoint)

def test_scalar_independent_recurrence_and_one_forward(monkeypatch):
    m=ACR(cfg(),'static_50');count=[0];original=m.core.step
    def step(*a,**kw):count[0]+=1;return original(*a,**kw)
    monkeypatch.setattr(m.core,'step',step)
    traces=[np.zeros((48,16),np.float64) for _ in range(6)]
    for row in sequences()[0]:
        v=m.core.voltage.copy();ad=m.core.adaptation.copy();z=m.core.spikes.copy()
        u=(cfg().leak*v+np.sum(m.core.w_rec*z[m.core.indices],axis=1))+m.core.w_in@row
        theta=cfg().threshold+cfg().adaptation_strength*ad
        # Independent scalar logistic surrogate with pinned width/cutoff.
        import math
        def scalar(d):
            raw=float(d)/.4;p=1/(1+math.exp(-max(-20,min(20,raw))))
            return p*(1-p)/.4 if abs(raw)<20 else 0
        psi=np.array([scalar(d) for d in u-theta])
        pre=np.concatenate((np.broadcast_to(row,(48,8)),z[m.core.indices]),1)
        m.step(row)
        for offset,stop in ((0,False),(3,True)):
            ev,ea,ef=traces[offset:offset+3];ue=.85*ev+pre;ze=psi[:,None]*(ue-.08*ea)
            traces[offset]=ue if stop else ue-theta[:,None]*ze-m.core.spikes[:,None]*.08*ea
            traces[offset+1]=.95*ea+ze
            ze[:,8:]*=1 if np.max(np.abs(row))>.6 else .15
            traces[offset+2]=m.gamma_e_vec*ef+(1-m.gamma_e_vec)*ze
        for name,oracle in zip(('e_v','e_a','e_f_own','e_v_stop','e_a_stop','e_f_stop'),traces):
            np.testing.assert_allclose(getattr(m,name),oracle,rtol=1e-4,atol=5e-6)
    assert count[0]==21

def test_dynamic_mix_after_filter_and_past_alpha_irrelevance():
    m=ACR(cfg(),'state');n=copy.deepcopy(m)
    for i,row in enumerate(sequences()[0]):
        m.step(row);n.step(row)
        np.testing.assert_allclose(m.e_f,(1-m.alpha[:,None])*m.e_f_own+m.alpha[:,None]*m.e_f_stop,rtol=0,atol=0)
        # Earlier mixtures are not fed back into either stream or q.
        if i<20:n.alpha[:]=1-n.alpha;n.mix()
    np.testing.assert_array_equal(m.e_f,n.e_f)
    np.testing.assert_array_equal(m.router_q,n.router_q)

@pytest.mark.parametrize('profile',['state','state_shuffle','random_alpha','state_signal_shuffle','linear_up','intrinsic'])
def test_full_snapshot_next_update(profile):
    m=ACR(cfg(),profile);m.calibrated=True;m.learn_episode(sequences()[0],1)
    payload=m.dumps();n=ACR.loads(payload)
    assert n.dumps()==payload
    assert m.learn_episode(sequences()[1],0)==n.learn_episode(sequences()[1],0)
    assert m.dumps()==n.dumps()
    with pytest.raises(ValueError):ACR.loads(payload[:-5])

def test_frozen_alias_and_readout_inference_invariance():
    a=ACR(cfg(),'static_50','frozen_hidden');b=ACR(cfg(),'readout_only')
    a.calibrated=b.calibrated=True
    for i,x in enumerate(sequences()[:4]):assert a.learn_episode(x,i%2)==b.learn_episode(x,i%2)
    np.testing.assert_array_equal(a.predict_many(sequences()),b.predict_many(sequences()))
    np.testing.assert_array_equal(a.core.w_in,a.w_in_0)
    m=ACR(cfg(),'state');n=ACR(cfg(),'static_0')
    np.testing.assert_array_equal(m.features_many(sequences()),n.features_many(sequences()))

def test_alpha_controls_distribution_and_distinct_state_shuffle():
    models=[ACR(cfg(),p) for p in ('state','state_shuffle','random_alpha')]
    for row in sequences()[0]:
        for m in models:m.step(row)
        np.testing.assert_array_equal(np.sort(models[0].alpha),np.sort(models[2].alpha))
        assert np.all((models[0].alpha>=0)&(models[0].alpha<=1))
    assert not np.array_equal(models[1].alpha,models[2].alpha)

def test_physical_norm_and_fail_closed(monkeypatch):
    w=np.ones((48,16),np.float32);c=w+.01
    radius=float(np.linalg.norm(c.astype(float)-w))/2
    q=ray_to_radius(w,c,radius)
    assert abs(np.linalg.norm(q.astype(float)-w)-radius)<=radius*1e-5
    with pytest.raises(NormMismatch):ray_to_radius(w,np.nextafter(w,np.float32(2)),1e-10)
    a=ACR(cfg());b=ACR(cfg(),'static_0');a.calibrated=b.calibrated=True
    ap=a.dumps();bp=b.dumps()
    def bad(*args,**kw):raise NormMismatch('fixture')
    monkeypatch.setattr('scarx.evaluation.acr.ray_to_radius',bad)
    with pytest.raises(NormMismatch):matched_update(a,b,sequences()[0],1)
    assert a.dumps()==ap and b.dumps()==bp

def test_logged_signal_equals_preupdate_optimizer_direction():
    m=ACR(cfg(),'static_50');m.calibrated=True
    old=m.syn_trace.copy();m.learn_episode(sequences()[0],1)
    np.testing.assert_array_equal(m.syn_trace,.9*old+.1*(m.last_signal[:,None]*m.e_f))
