import hashlib,io,json,zipfile
import numpy as np
import pytest
from scarx.neural.unified import UnifiedCore
from scarx.evaluation import readout_usefulness as module
from scarx.evaluation.readout_usefulness import UsefulnessPrototype,P0_FIXTURES,counterfactual_benefit
from scarx.evaluation.usefulness_validation import oracle_loss,checkpoint_check,budget_check


def test_deletion_formula_extreme_probability_and_mask_matches_independent_forward():
    w=np.array([[.2,-.7,.5],[1.,-.3,.8],[-.5,.9,1.2]]);f=np.array([.5,1.,.2]);b=np.array([1000.,-1000.,0.]);mask=np.ones(3,bool)
    result=counterfactual_benefit(w,f,b,mask,0);base=oracle_loss(w,f,b,mask,0)
    for k in range(3):
        for j in range(3):
            removed=w.copy();removed[k,j]=0
            assert result[k,j]==pytest.approx(oracle_loss(removed,f,b,mask,0)-base,abs=1e-9)
    mask[1]=False
    assert not counterfactual_benefit(w,f,b,mask,0)[1].any()
    with pytest.raises(ValueError):counterfactual_benefit(w,f,b,mask,1)


def test_probability_rounded_to_one_still_has_nonzero_deletion_benefit():
    w=np.array([[20.,0.],[0.,0.],[0.,0.]]);f=np.ones(2);b=np.zeros(3);mask=np.ones(3,bool)
    logits=(w@f).astype(np.float32);p=np.exp(logits-max(logits));p/=p.sum();assert p[0]==1
    value=counterfactual_benefit(w,f,b,mask,0)[0,0]
    assert value==pytest.approx(np.log(3.)-oracle_loss(w,f,b,mask,0),abs=1e-12) and value>1.


@pytest.mark.parametrize('seed',[42201,42203,42207])
def test_beta_zero_matches_full_G1_neural_payload_after_each_segment(seed):
    p=UsefulnessPrototype(seed,0.);base=UnifiedCore(p.vocabulary,seed)
    for _ in range(4):
        for identifier,fixture in P0_FIXTURES.items():
            assert p.teach_fixture(identifier)==base.train_pair(fixture['prompt'],fixture['answer'])
            assert p.base_dumps()==base.dumps()


@pytest.mark.parametrize('correct,initial_importance',[(True,0.),(True,.25),(False,.25)])
def test_importance_gating_and_only_readout_delta_differs_at_same_endpoint(correct,initial_importance):
    p=UsefulnessPrototype(42201,4.);p.begin_fixture('p0-alpha');c=p.core;y=p.ids['alpha']
    c.features.fill(.5);c.w_out.fill(0);c.w_out[y]=.5;c.bias.fill(0);c.bias[y]=2 if correct else -10
    if not correct:c.bias[p.ids['beta']]=20
    p.omega.fill(initial_importance);before_omega=p.omega.copy();base=UnifiedCore.loads(p.base_dumps())
    base.word_step(1,base.context(P0_FIXTURES['p0-alpha']['prompt']),0,True)
    probability=base.probabilities('language');assert (int(np.argmax(probability))==y)==correct
    mask=np.zeros(len(probability),bool)
    for token in ('alpha','beta','gamma'):mask[p.ids[token]]=True
    expected=.995*before_omega
    if correct:
        loss=oracle_loss(base.core.w_out,base.core.features,base.core.bias,mask,y)
        for k in np.flatnonzero(mask):
            for j in range(48):
                deleted=base.core.w_out.astype(np.float64);deleted[k,j]=0
                benefit=oracle_loss(deleted,base.core.features,base.core.bias,mask,y)-loss
                expected[k,j]=max(expected[k,j],np.clip(benefit,0,1))
    base.learn(y,'language');p.advance()
    np.testing.assert_allclose(p.omega,expected,atol=1e-6,rtol=0)
    for name,array in base.core.arrays().items():
        if name!='w_out':np.testing.assert_array_equal(getattr(c,name),array)
    assert c.reward_mean==base.core.reward_mean and c.updates==base.core.updates
    if correct and initial_importance==0:assert p.omega.any()
    if initial_importance>0:assert np.any(c.w_out!=base.core.w_out)


def test_pending_segment_rng_replay_rollback_roundtrip(tmp_path):
    result=checkpoint_check(tmp_path/'p0.zip')
    assert result['pending_token_boundary_new_instance_exact'] and result['rejected_transaction_full_rollback']


def test_replay_stays_bounded_and_persists_after_more_than_capacity_segments():
    model=UsefulnessPrototype()
    for _ in range(33):model.teach_fixture('p0-alpha')
    assert len(model.replay)==32 and model.core.updates==99
    assert UsefulnessPrototype.loads(model.dumps()).dumps()==model.dumps()


def test_matched_budget_detects_unrepresentable_float32_delta():
    result=budget_check()
    assert result['valid_interior_groups']==32 and result['maximum_relative_applied_norm_error']<=1e-5
    assert result['quantized_edge']['rejected'] and result['universally_feasible_matching'] is False


def test_atomic_save_failure_rolls_back_full_transaction_and_keeps_previous_file(tmp_path,monkeypatch):
    model=UsefulnessPrototype();path=tmp_path/'previous.zip';model.save(path);before=model.dumps();old=path.read_bytes()
    def fail(*_a,**_kw):raise OSError('Injected atomic replace failure')
    monkeypatch.setattr(module,'replace_with_retry',fail)
    with pytest.raises(OSError):model.teach_fixture('p0-alpha',commit=lambda m:m.save(path))
    assert model.dumps()==before and path.read_bytes()==old and not list(tmp_path.glob('p0-checkpoint-*.tmp'))


@pytest.mark.parametrize('corruption',['importance','controller','provenance'])
def test_corrupt_checkpoint_rejected_before_mutating_live_research_model(corruption):
    model=UsefulnessPrototype();model.begin_fixture('p0-alpha');model.advance();before=model.dumps()
    with zipfile.ZipFile(io.BytesIO(before)) as z:blobs={n:z.read(n) for n in z.namelist()}
    meta=json.loads(blobs['metadata.json'])
    if corruption=='importance':
        stream=io.BytesIO();np.savez_compressed(stream,omega=np.full_like(model.omega,np.nan));blobs['state.npz']=stream.getvalue()
        meta['sha256']['state.npz']=hashlib.sha256(blobs['state.npz']).hexdigest()
    elif corruption=='controller':meta['controller']['endpoints']+=1
    else:meta['controller']['pending']['source_sha256']='0'*64
    blobs['metadata.json']=json.dumps(meta).encode();stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for name,raw in blobs.items():z.writestr(name,raw)
    with pytest.raises(ValueError):model.rollback(stream.getvalue())
    assert model.dumps()==before
