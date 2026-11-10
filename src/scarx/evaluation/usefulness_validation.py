"""Independent P0 numerical/state checks. No quality corpus or pilot training."""
import ast,io,json,time,zipfile
from pathlib import Path
import numpy as np
from ..neural.unified import UnifiedCore
from .readout_usefulness import UsefulnessPrototype,P0_FIXTURES,counterfactual_benefit,matched_candidates,AppliedBudgetError


def oracle_loss(w,f,b,mask,target):
    logits=np.asarray(w,np.float64)@np.asarray(f,np.float64)+np.asarray(b,np.float64)
    values=logits[mask];maximum=float(values.max())
    return maximum+float(np.log(np.exp(values-maximum).sum()))-float(logits[target])


def math_check():
    rng=np.random.default_rng(42001);maximum=fd_maximum=0.;synapses=finite_differences=0
    matrices=[]
    for outputs,hidden in ((3,4),(9,48),(266,48)):
        w=rng.uniform(-1,1,(outputs,hidden));f=rng.uniform(0,1,hidden);b=rng.normal(0,.3,outputs);mask=np.ones(outputs,bool)
        if outputs==9:mask[1::3]=False
        matrices.append((w,f,b,mask,0))
    matrices.append((rng.uniform(-20,20,(3,4)),np.ones(4),np.array([1000.,-1000.,0.]),np.ones(3,bool),0))
    for w,f,b,mask,target in matrices:
        actual=counterfactual_benefit(w,f,b,mask,target);base=oracle_loss(w,f,b,mask,target)
        for k in np.flatnonzero(mask):
            for j in range(w.shape[1]):
                removed=w.copy();removed[k,j]=0
                expected=oracle_loss(removed,f,b,mask,target)-base
                maximum=max(maximum,abs(float(actual[k,j])-expected));synapses+=1
        if w.shape[0]!=3 or np.max(abs(b))<10:
            logits=w@f+b;values=np.exp(logits[mask]-max(logits[mask]));p=np.zeros(len(b));p[mask]=values/values.sum()
            for flat in range(min(16,w.size)):
                k,j=np.unravel_index(flat,w.shape)
                plus=w.copy();minus=w.copy();plus[k,j]+=1e-6;minus[k,j]-=1e-6
                numeric=(oracle_loss(plus,f,b,mask,target)-oracle_loss(minus,f,b,mask,target))/2e-6
                expected=(p[k]-(k==target))*f[j] if mask[k] else 0.
                fd_maximum=max(fd_maximum,abs(numeric-expected));finite_differences+=1
        assert np.count_nonzero(actual[~mask])==0
    assert maximum<=1e-9 and fd_maximum<=1e-7
    assert np.max(abs(counterfactual_benefit(np.ones((3,4)),np.zeros(4),np.zeros(3),np.ones(3,bool),0)))<1e-12
    assert not counterfactual_benefit(np.ones((3,4)),np.ones(4),np.zeros(3),np.array([True,False,False]),0).any()
    return {'deletion_synapses':synapses,'max_absolute_deletion_error':maximum,'finite_difference_elements':finite_differences,'max_absolute_output_derivative_error':fd_maximum,'extreme_logits_checked':True,'masked_and_zero_feature_checked':True,'hidden_derivatives_used':False}


def trajectories():
    rows=[]
    for seed in (42201,42203,42207):
        beta0=UsefulnessPrototype(seed,0.);base=UnifiedCore(beta0.vocabulary,seed=seed)
        for _ in range(4):
            for identifier,fixture in P0_FIXTURES.items():
                actual=beta0.teach_fixture(identifier);expected=base.train_pair(fixture['prompt'],fixture['answer'])
                assert actual==expected and beta0.base_dumps()==base.dumps()
        model=UsefulnessPrototype(seed,4.);before={n:getattr(model.core,n).copy() for n in ('w_in','w_rec','w_out')};losses=[];start=time.perf_counter()
        for _ in range(4):
            for identifier in P0_FIXTURES:losses.append(model.teach_fixture(identifier))
        seconds=time.perf_counter()-start
        rows.append({'seed':seed,'beta0_base_payload_exact_after_each_segment':True,'beta0_segments':12,'beta0_endpoints':beta0.core.updates,
                     'beta4_segments':12,'beta4_endpoints':model.core.updates,'beta4_seconds':seconds,'beta4_losses':losses,
                     'beta4_omega_nonzero':int(np.count_nonzero(model.omega)),'beta4_omega_bytes':model.omega.nbytes,
                     'weight_delta':{n:float(np.linalg.norm(getattr(model.core,n)-before[n])) for n in before},
                     'changed_synapses':{n:int(np.count_nonzero(getattr(model.core,n)!=before[n])) for n in before}})
    return rows


def checkpoint_check(path):
    model=UsefulnessPrototype(42201,4.);model.teach_fixture('p0-alpha');model.omega[:]=.25;model.controller['replay_cursor']=1
    model.draw_fixture();model.permuted_g();model.begin_fixture('p0-beta');model.advance();pending=model.dumps()
    model.save(path);cold=UsefulnessPrototype.load(path);assert cold.dumps()==pending
    for _ in range(3):assert model.draw_fixture()==cold.draw_fixture();np.testing.assert_array_equal(model.permuted_g(),cold.permuted_g())
    while model.controller['pending']:
        assert model.advance()==cold.advance();assert model.dumps()==cold.dumps()
    for identifier in ('p0-gamma','p0-alpha'):
        assert model.teach_fixture(identifier)==cold.teach_fixture(identifier);assert model.dumps()==cold.dumps()
    before=model.dumps();core=model.core
    def reject(m):
        m.draw_fixture();m.permuted_g();m.omega.fill(.75)
        raise RuntimeError('Injected P0 rejected transaction')
    try:model.teach_fixture('p0-beta',commit=reject)
    except RuntimeError as exc:assert 'Injected P0' in str(exc)
    else:raise AssertionError('Fault injection did not execute')
    assert model.dumps()==before and model.core is core
    for call in (lambda:model.train_pair('unverified','alpha.',verified=True),lambda:model.begin_fixture('not-a-source'),lambda:model.image(np.ones((8,8)),train_label=1)):
        try:call()
        except ValueError:pass
        else:raise AssertionError('Unverified fixture/image teaching must fail')
        assert model.dumps()==before
    try:UnifiedCore.loads(before)
    except ValueError:pass
    else:raise AssertionError('Research wrapper must not load as product checkpoint')
    return {'pending_token_boundary_cold_exact':True,'post_resume_trajectories_exact':True,'rng_fixture_and_shuffle_exact':True,'replay_and_controller_exact':True,'rejected_transaction_full_rollback':True,'core_identity_preserved_on_rollback':True,'unverified_sources_rejected_without_mutation':True,'product_codec_rejects_wrapper':True,'snapshot_bytes':len(pending)}


def budget_check():
    rng=np.random.default_rng(42003);maximum=0.;cases=0
    for _ in range(32):
        weights=[rng.normal(0,.03,(9,48)).astype(np.float32) for _ in range(3)]
        candidates=[np.clip(w+np.clip(rng.normal(0,.02,w.shape).astype(np.float32),-.025,.025),-20,20) for w in weights]
        before=[w.copy() for w in weights];result,metrics=matched_candidates(weights,candidates)
        maximum=max(maximum,max(abs(n-metrics['radius'])/metrics['radius'] for n in metrics['actual_norms']));cases+=1
        for original,snapshot in zip(weights,before):np.testing.assert_array_equal(original,snapshot)
    w=np.ones((1,1),np.float32);candidate=np.nextafter(w,np.full_like(w,2.))
    try:matched_candidates([w,w],[candidate,candidate])
    except AppliedBudgetError as exc:edge={'radius':exc.radius,'actual_norms':exc.norms,'rejected':True,'reason':'Positive half-ULP radius not representable in float32 weight state'}
    else:raise AssertionError('HalfULP budget must not be reported matched')
    _,zero=matched_candidates([w,w],[w,candidate]);assert zero['zero_radius'] and zero['actual_norms']==[0.,0.]
    return {'valid_interior_groups':cases,'maximum_relative_applied_norm_error':maximum,'quantized_edge':edge,'zero_radius_reported':True,'universally_feasible_matching':False,'raw_norm_used_as_applied':False}


def signal_audit():
    paths=[Path(__file__),Path(__file__).with_name('readout_usefulness.py')];imports=[];reverse_calls=[]
    for path in paths:
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(n.name for n in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='backward':reverse_calls.append(node.lineno)
    assert not any(i.split('.')[0] in ('torch','jax','tensorflow','transformers') for i in imports) and not reverse_calls
    return {'imports_checked':sorted(set(imports)),'autograd_or_transformer_imports':False,'backward_calls':0,'hidden_signal':'Delegates exactly UnifiedCore.learn scalar rule; only Wout candidate overwritten','scope':'Research modules only; full suite also checks production/native parity'}


def utility_timing():
    rng=np.random.default_rng(42001);w=rng.normal(0,.1,(266,48));f=rng.uniform(0,1,48);b=np.zeros(266);mask=np.ones(266,bool)
    start=time.perf_counter()
    for _ in range(100):counterfactual_benefit(w,f,b,mask,0)
    return {'outputs':266,'hidden':48,'calls':100,'seconds':time.perf_counter()-start,'importance_float32_bytes':266*48*4,'quality_corpus_read':False}
