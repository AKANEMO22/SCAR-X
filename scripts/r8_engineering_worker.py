"""Bounded peer-code engineering counterexamples; no learner update calls."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,zipfile
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np

SHA='fe472b3b1257cdb264805bf79fd53e5647e3568d'


def main():
    source=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SHA:
        raise ValueError('Pinned owned source required')
    if out.exists():raise ValueError('Fresh output required')
    sys.path.insert(0,str(source/'src'))
    from scarx.neural.core import Config
    from scarx.neural.r8_mtsg_eprop import MultiTimescaleSensoryGatedEProp as M
    from scarx.neural.r8_bf_cascade import BennaFusiMetaplasticCascade as B
    from scarx.evaluation.r8_gate_evaluator import R8GateEvaluator as G
    import scarx.evaluation.r8_benchmark_integrity as generators

    cfg=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
    result={'source_sha':SHA,'new_quality_training':False,'fixture_learn_calls':0,
        'empty_evaluator_overall_pass':G({}).evaluate_full()['overall_pass']}
    bogus={'learned_mtsg':{'direct_acc_mean':.99,'probe_acc_mean':.8,'probe_gain_lower95':-.5},
           'frozen_hidden':{'probe_acc_mean':.5},'r35_baseline':{'probe_acc_mean':.5},
           'shuffled_signal':{'probe_acc_mean':.8},'causal_revert':{'drop_mean':0.}}
    result['high_accuracy_bypasses_ci_causal_and_shuffle']=G({'p1_results':{'xor':bogus}}).evaluate_p1_task('xor')
    loose={'bf_cascade':{'acquisition_mean':.9,'final_mean':.7,'mean_forgetting':.2},
           'unconsolidated':{'mean_forgetting':.3}}
    result['p2_20pp_forgetting_is_accepted']=G({'p2_results':loose}).evaluate_p2()['pass']

    m=M(cfg,'learned_mtsg',sensory_threshold=.123);m.core.spikes[:]=1
    m.signal_rng.permutation(48);payload=m.dumps();loaded=M.loads(payload)
    result['checkpoint']={'threshold_before':m.sensory_threshold,'threshold_after':loaded.sensory_threshold,
        'rng_state_preserved':m.signal_rng.bit_generator.state==loaded.signal_rng.bit_generator.state}
    stimulus=np.full(8,.03,np.float32);m.step(stimulus,True);loaded.step(stimulus,True)
    result['checkpoint']['next_trace_max_error']=float(np.max(np.abs(m.e_f-loaded.e_f)))

    with zipfile.ZipFile(io.BytesIO(payload)) as z:members={n:z.read(n) for n in z.namelist()}
    with np.load(io.BytesIO(members['state.npz']),allow_pickle=False) as z:states={n:z[n].copy() for n in z.files}
    states['e_f']=states['e_f'].astype(np.float64);states['e_f'].fill(float('nan'))
    stream=io.BytesIO();np.savez(stream,**states);members['state.npz']=stream.getvalue();stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for name,raw in members.items():z.writestr(name,raw)
    try:
        malformed=M.loads(stream.getvalue())
        result['checkpoint']['float64_nan_state_accepted']=not np.isfinite(malformed.e_f).all()
    except Exception as exc:result['checkpoint']['float64_nan_state_accepted']=False

    learned=M(cfg,'learned_mtsg');shuffled=M(cfg,'shuffled_signal')
    learned.core.spikes[:]=1;shuffled.core.spikes[:]=1
    learned.step(stimulus,True);shuffled.step(stimulus,True)
    result['shuffle_changes_trace_rule_before_any_error_signal']={
        'forward_voltage_equal':bool(np.array_equal(learned.core.voltage,shuffled.core.voltage)),
        'eligibility_max_difference':float(np.max(np.abs(learned.e_f-shuffled.e_f))),
        'hidden_weight_updates':learned.hidden_updates+shuffled.hidden_updates}

    b=B(cfg,use_context_gating=False)
    arrays=[v for obj in (b,b.core) for v in obj.__dict__.values() if isinstance(v,np.ndarray)]
    unique={id(v):v for v in arrays}
    result['bf_state']={'has_dumps':hasattr(b,'dumps'),'has_loads':hasattr(b,'loads'),
        'reported_array_bytes':b.memory_bytes(),'actual_unique_allocated_array_bytes':sum(v.nbytes for v in unique.values()),
        'unreported_w_ctx_bytes':b.w_ctx.nbytes}

    # Oracle fixture only: remove background noise without changing key/value/query logic.
    original_rng=generators.np.random.default_rng
    class NoNoise:
        def __init__(self,rng):self.rng=rng
        def normal(self,loc,scale,size):return np.zeros(size)
        def __getattr__(self,name):return getattr(self.rng,name)
    collisions={}
    try:
        generators.np.random.default_rng=lambda seed:NoNoise(original_rng(seed))
        for keys in (2,4,8):
            x,y=generators.generate_multikey_associative(42,256,n_keys=keys)
            seen={};conflicts=[]
            for i,(row,label) in enumerate(zip(x,y)):
                digest=hashlib.sha256(row.tobytes()).hexdigest()
                if digest in seen and seen[digest][1]!=int(label):
                    conflicts.append({'row_a':seen[digest][0],'row_b':i,'label_a':seen[digest][1],'label_b':int(label)})
                else:seen[digest]=(i,int(label))
            collisions[str(keys)]={'zero_noise_rows':len(y),'conflicting_identical_input_rows':len(conflicts),'first_conflict':conflicts[:1]}
    finally:generators.np.random.default_rng=original_rng
    result['actual_generator_ambiguity_fixture']=collisions
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'empty_pass':result['empty_evaluator_overall_pass'],'query_conflicts':collisions['8']['conflicting_identical_input_rows'],'training':False}))


if __name__=='__main__':main()
