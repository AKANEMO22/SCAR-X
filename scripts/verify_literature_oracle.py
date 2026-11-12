"""Bounded isolated frozen-weight CONTROL, not a DEV/training runner.

Creates unique evidence, no updates or labels from a dataset. Uses a shared
fixed random output-sensitivity vector to compare credit directions only.
"""
from pathlib import Path
import hashlib,json,os,platform,sys,time,uuid
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.neural.core import Config
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit
from scarx.evaluation.acr import AdaptiveCreditRouting
from scarx.evaluation.literature_eprop import LiteratureEPropALIF
from scarx.evaluation.literature_braintrace import BrainTraceDRTRL,full_surrogate_sensitivity,direction_metrics
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def main():
    run=ROOT/'runs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-literature-oracle-v1-'+uuid.uuid4().hex[:6])
    run.mkdir(parents=True,exist_ok=False);cap=apply_memory_limit();rows=[]
    monitor=ResourceMonitor(interval=.05);started=time.perf_counter()
    methods=('own','stopped','eprop_matched','eprop_aligned','braintrace_drtrl','acr_static_50','acr_state')
    with monitor:
        for seed in (422,773,1127):
            config=Config(inputs=8,hidden=48,outputs=2,fan_in=8,plastic_rows=12,seed=seed)
            # These are numerical fixtures, not scientific DEV seeds or tasks.
            x=np.random.default_rng(seed+1).uniform(-.1,1.1,(21,8)).astype(np.float32)
            signal=np.random.default_rng(seed+2).normal(size=48)
            base=MatchedLocalCredit(config);fingerprint=base.core.fingerprint()
            cpu=time.process_time();exact=full_surrogate_sensitivity(base.core,x);oracle_cpu=time.process_time()-cpu
            reference=signal@exact
            payload={'sequence':x,'signal':signal,'full_feature_sensitivity':exact,'reference_direction':reference}
            for name in methods:
                if name=='own':m=MatchedLocalCredit(config)
                elif name=='stopped':m=StoppedResetCredit(config)
                elif name.startswith('eprop_'):m=LiteratureEPropALIF(config,name)
                elif name=='braintrace_drtrl':m=BrainTraceDRTRL(config)
                else:m=AdaptiveCreditRouting(config,'static_50' if name=='acr_static_50' else 'state')
                before=m.core.fingerprint();cpu=time.process_time()
                for row in x:m.step(row,track=True)
                elapsed=time.process_time()-cpu
                if m.core.fingerprint()!=before or before!=fingerprint or m.hidden_updates or m.episodes_learned:
                    raise ValueError('Frozen-weight oracle unexpectedly learned')
                direction=(signal[:,None]*m.e_f).ravel()
                payload[name+'_eligibility']=m.e_f.copy();payload[name+'_direction']=direction
                if name=='own':features=m.core.features.copy();state=(m.core.voltage.copy(),m.core.adaptation.copy(),m.core.spikes.copy())
                else:
                    for actual,want in zip((m.core.voltage,m.core.adaptation,m.core.spikes),state):np.testing.assert_array_equal(actual,want)
                    np.testing.assert_array_equal(m.core.features,features)
                rows.append({'seed':seed,'method':name,**direction_metrics(direction,reference),
                    'cpu_seconds':elapsed,'full_oracle_cpu_seconds':oracle_cpu,'persistent_array_bytes':m.memory_bytes(),
                    'actual_learning_updates':0,'forward_unchanged':True})
                monitor.check()
            np.savez_compressed(run/f'fixture-{seed}.npz',**payload)
    paths=('src/scarx/evaluation/literature_braintrace.py','src/scarx/evaluation/literature_eprop.py',
           'src/scarx/evaluation/acr.py','src/scarx/evaluation/r8_matched_credit.py',
           'src/scarx/evaluation/r8_stopped_reset.py','src/scarx/evaluation/readout_eligibility.py',
           'src/scarx/neural/core.py','scripts/verify_literature_oracle.py')
    result={'scope':'Isolated frozen-weight surrogate CONTROL, random fixed signal; no task accuracy claim',
        'rows':rows,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        'braintrace_upstream_sha':'fafa25753b743a2e2d5b47209c4ee947abc02cfb',
        'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()},
        'resources':monitor.report(),'cap':cap,'wall_seconds':time.perf_counter()-started,
        'full_oracle_state_sensitivity_bytes':4*48*(48*16)*8,
        'candidate_bptt':False,'new_training_updates':0,'fresh_confirmation':False,'production_change':False}
    (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':str(run),'rows':len(rows),'new_training_updates':0,'resources':result['resources']}),flush=True)


if __name__=='__main__':main()
