"""Engineering checks against actual pinned source; no new scored experiment."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--kind',choices=('r6d','r6b'),required=True)
    args=parser.parse_args();source=Path(args.source).resolve();sys.path.insert(0,str(source/'src'))
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('independent_ref',root/'src/scarx/evaluation/forensic/reference_evaluator.py')
    ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
    import numpy as np
    results={}
    if args.kind=='r6d':
        from scarx.evaluation import r5_independent as G,r5_probe as P,r6d_audit as A
        from scarx.neural.core import SpikingCore
        x,y,a,b,q,pid,flips=G.independent_paired_set_v1(530001,32)
        results['clean']=ref.validate_pair_inputs(x,y,a,b,q,pid,flips,expected_tick=10)
        for name,transformed,qt in [('temporal',G.shifted(x,1,seed=130001),11),('distractor',G.with_distractors(x,130058),10)]:
            try:ref.validate_pair_inputs(transformed,y,a,b,q,pid,flips,expected_tick=qt);violation=None
            except ref.AuditViolation as exc:violation=str(exc)
            relaxed=ref.validate_pair_inputs(transformed,y,a,b,q,pid,flips,expected_tick=qt,common_noise=False)
            results[name]={'violation':violation,'row_oracle_and_pair_facts':relaxed}
        actual=P.capture(130001,x)
        zero=P.capture(130001,G.shifted_zero_pad(x),query_tick=11)
        noisy=P.capture(130001,G.shifted(x,seed=130001),query_tick=11)
        results['shift_feature_check']={'zero_pad_max_post_difference':float(np.max(np.abs(P.state_vector(actual,'post')-P.state_vector(zero,'post')))),
            'noisy_pad_max_post_difference':float(np.max(np.abs(P.state_vector(actual,'post')-P.state_vector(noisy,'post'))))}
        c,idx,wi,wr=P.weights(130001);spike_error=feature_error=voltage_error=0.
        for index,seq in enumerate(x[:8]):
            core=SpikingCore(c)
            assert np.array_equal(core.indices,idx) and np.array_equal(core.w_in,wi) and np.array_equal(core.w_rec,wr)
            for tick,vector in enumerate(seq):
                core.step(vector,track=False)
                spike_error=max(spike_error,float(np.max(np.abs(core.spikes-actual['spike_train'][index,tick]))))
                if tick==10:
                    feature_error=max(feature_error,float(np.max(np.abs(core.features-actual['post']['features'][index]))))
                    voltage_error=max(voltage_error,float(np.max(np.abs(core.voltage-actual['post']['voltage'][index]))))
        results['forward_parity']={'examples':8,'seed':130001,'max_spike_error':spike_error,'max_post_feature_error':feature_error,'max_post_voltage_error':voltage_error,
            'exact_zero_all_state_components_measured':max(spike_error,feature_error,voltage_error)==0}
        # Matching formulas under constructed stats is not matching the actual
        # calibration pipeline: show both on one published small data fixture.
        train,*_=G.independent_query_set_v1(330001,1024);raw=A.raw_features(130001,train)
        head=P.Head(raw.shape[1],130001)
        for f in raw:head.observe(f)
        sc=A.fixed_calibration(130001,'r5b')
        original=np.stack([head.basis(f) for f in raw]);fixed=sc(raw)
        results['normalization_source_difference']={'max_abs_difference':float(np.max(np.abs(original-fixed))),
            'original_train_stats_ddof':1,'r6d_pinned_stats_ddof':0,'fixed_calib_sha256':sc.digest(),
            'original_stats_sha256':hashlib.sha256(head.mean.tobytes()+head.m2.tobytes()).hexdigest()}
    else:
        from scarx.evaluation.r6_candidate import NoClip
        from scarx.evaluation import r5b_probe as P
        h=NoClip(2,5,lr=.5,zscore=False)
        f=np.array([50.,0],np.float32);before=h.w.copy();bias=h.b.copy();p=h.probs(f);e=-p.copy();e[1]+=1
        expected=np.clip(before+.5*e[:,None]*f[None,:],-3,3);expected_b=np.clip(bias+.5*e,-3,3)
        h.learn(f,1)
        assert np.array_equal(expected,h.w) and np.array_equal(expected_b,h.b)
        results['real_candidate']={'module':NoClip.__module__,'file':str(Path(sys.modules[NoClip.__module__].__file__)),
            'one_step_unclipped_formula_exact':True,'weights_finite':bool(np.isfinite(h.w).all())}
        spec=importlib.util.spec_from_file_location('shadow_toy_test',source/'tests/test_r6_toys.py');toy=importlib.util.module_from_spec(spec);spec.loader.exec_module(toy)
        results['test_shadow']={'imported_candidate_is_test_binding':toy.NoClip is NoClip,'test_binding_module':toy.NoClip.__module__}
        results['duplicate_functions']=[]
        try:ref.unique_top_level_definitions((source/'scripts/grok_r6_causal.py').read_text())
        except ref.AuditViolation as exc:results['duplicate_functions'].append(str(exc))
    print(json.dumps(results,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
