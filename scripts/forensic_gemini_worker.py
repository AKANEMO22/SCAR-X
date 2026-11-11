"""Gemini CKCER engineering and historical prediction audit; no fresh seeds."""
from pathlib import Path
import hashlib,importlib.util,inspect,json,os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];source=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
out.mkdir(exist_ok=False,parents=True);(out/'arrays').mkdir()
sys.path.insert(0,str(source/'src'))
import numpy as np
from scarx.evaluation import r6_associative as R
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
spec=importlib.util.spec_from_file_location('independent_ref',ROOT/'src/scarx/evaluation/forensic/reference_evaluator.py')
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
cap=apply_memory_limit()


def save(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    # Actual class engineering check: zero inactive weight update != zero shared
    # bias interference. No task accuracy or trained representation is claimed.
    m=np.stack([np.arange(96),np.arange(96)+1,np.arange(96)-1]).astype(np.float32)
    h=R.CKCERHead(dim=96,mode='A',seed=5);h.observe(m)
    cue0=np.array([[1.,0.]],np.float32);cue1=np.array([[0.,1.]],np.float32)
    feature=h.build_features(m[:1],cue=cue0)[0];before=h.scores(m[:1],cue=cue1).copy()
    weights_before=h.w.copy();h.step_update(feature,1);after=h.scores(m[:1],cue=cue1)
    checks={'inactive_branch_weight_delta_max':float(np.max(np.abs(h.w[:,96:192]-weights_before[:,96:192]))),
        'inactive_query_scores_change_due_to_shared_bias_max':float(np.max(np.abs(after-before))),
        'recurrent_hidden_training_present':False,'oja_gate_location':'CKCERHead.w_q, outside frozen recurrent weights'}
    save('engineering.json',checks)
    observations=[];current_seed=None;original=R.evaluate_counterfactual_detailed
    def score(sa,sb,ya,yb,*args,**kwargs):
        source_metric=original(sa,sb,ya,yb,*args,**kwargs)
        pred=np.empty(len(ya)*2,np.int64);pred[::2]=np.argmax(sa,1);pred[1::2]=np.argmax(sb,1)
        y=np.empty_like(pred);y[::2]=ya;y[1::2]=yb
        a=np.repeat(ya,2);b=np.repeat(yb,2);q=np.tile([0,1],len(ya));pid=np.repeat(np.arange(len(ya)),2)
        metric=ref.pair_metrics(pred,y,a,b,q,pid,ya!=yb)
        for name in ('both_A','both_B','both_all'):
            if metric[name]!=source_metric[name]:raise ValueError('Gemini metric differs from independent pair scorer')
        caller=inspect.currentframe().f_back
        identifier=f'{len(observations):04d}'
        target=out/'arrays'/(identifier+'.npz')
        np.savez_compressed(target,prediction=pred,label=y,vA=a,vB=b,q=q,pair_id=pid,scores_a=sa,scores_b=sb)
        observations.append({'id':identifier,'seed':current_seed,'caller':caller.f_code.co_name,'source_line':caller.f_lineno,
            'condition':caller.f_locals.get('cond_name'),'independent_metrics':metric,
            'raw_event_oracle_scope':'Not independently checked for all auxiliary/noise samples; facts here are derived from paired query labels, for metric arithmetic verification only.',
            'arrays':str(target.relative_to(out)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
        return source_metric
    R.evaluate_counterfactual_detailed=score
    rows=[]
    with ResourceMonitor(interval=.05) as monitor:
        for current_seed in (121011,121017,121023,121029,121031,121037,121041,121043):
            rows.append(R.evaluate_seed_full(current_seed));monitor.check()
            print(json.dumps({'historical_seed_done':current_seed}),flush=True)
    saved=json.loads((source/'artifacts/gemini-r6/confirmation.json').read_text())
    save('result.json',{'run_class':'HISTORICAL_REPRODUCTION_NOT_FRESH','source_sha':'8f24a5abde0598201e558eee559564ccde889145',
        'seed_results':rows,'published_seed_results_exact':rows==saved['seed_results'],'prediction_observations':observations,
        'engineering':checks,'resources':monitor.report(),'cap':cap,'new_candidate':False,'hidden_synapse_representation_learning_demonstrated':False})
    print(json.dumps({'published_seed_results_exact':rows==saved['seed_results'],'pair_prediction_groups':len(observations),'engineering':checks}),flush=True)


if __name__=='__main__':main()
