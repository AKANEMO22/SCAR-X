"""Finite preregistered literature comparison. No production writes/confirmation."""
from pathlib import Path
import json,os,platform,subprocess,sys,time,uuid
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import psutil
from scarx.neural.core import Config
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit,PEER,PEER_SHA
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit
from scarx.evaluation.literature_eprop import LiteratureEPropALIF
from scarx.evaluation.literature_braintrace import BrainTraceDRTRL
from scarx.evaluation.r8_matched_study import SEEDS,TASKS,dataset,direct_predict,refit_predict
from scarx.evaluation.acr_study import evaluate,stress_cases,sha
from scarx.evaluation.confirmation_resources import ConfirmationResources
from scarx.runtime.limits import apply_memory_limit
from run_r8_matched_dev import train,atomic_json as write

METHODS=('own','stop','eprop_matched','eprop_aligned','braintrace','readout_only')
PROTOCOL='experiments/literature-reference-dev-v1.md'
FILES=(PROTOCOL,'scripts/run_literature_reference_dev.py','scripts/run_r8_matched_dev.py',
 'src/scarx/evaluation/literature_eprop.py','src/scarx/evaluation/literature_braintrace.py',
 'src/scarx/evaluation/r8_matched_credit.py','src/scarx/evaluation/r8_stopped_reset.py',
 'src/scarx/evaluation/r8_matched_study.py','src/scarx/evaluation/acr_study.py',
 'src/scarx/evaluation/readout_eligibility.py','src/scarx/evaluation/forensic/r8_snapshot.py',
 'src/scarx/evaluation/confirmation_resources.py','src/scarx/runtime/limits.py',
 'src/scarx/runtime/resources.py','src/scarx/neural/core.py')

def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()

def factory(method,seed,data,shuffle=False):
    c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed)
    arm='multi_gated_shuffle' if shuffle else 'multi_gated'
    if method=='own':m=MatchedLocalCredit(c,arm)
    elif method=='stop':m=StoppedResetCredit(c,arm)
    elif method.startswith('eprop_'):m=LiteratureEPropALIF(c,method,arm)
    elif method=='braintrace':m=BrainTraceDRTRL(c,'shuffled_signal' if shuffle else 'learned_mtsg')
    elif method=='readout_only':m=MatchedLocalCredit(c,'multi_gated_readout_only')
    else:raise ValueError('Unregistered method')
    m.calibrate(data['train'][0][:32]);return m

def lock_sources():
    external={}
    refs=ROOT.parent.parent/'references'
    for folder,names in {
      'eligibility_propagation':['LICENSE','Figure_3_and_S7_e_prop_tutorials/models.py'],
      'braintrace':['LICENSE','braintrace/_algorithm/d_rtrl.py','braintrace/_algorithm/param_dim_vjp.py'],
    }.items():
        path=refs/folder
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        expected={'eligibility_propagation':'efd02e6879c01cda3fa9a7838e8e2fd08163c16e','braintrace':'fafa25753b743a2e2d5b47209c4ee947abc02cfb'}[folder]
        if actual!=expected or subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('External source moved or dirty')
        external[folder]={'sha':actual,
          'files':{name:sha((path/name).read_bytes()) for name in names}}
    return {'files':{f:sha((ROOT/f).read_bytes()) for f in FILES},'external':external,
            'peer':{f:sha((PEER/f).read_bytes()) for f in ('src/scarx/neural/r8_mtsg_eprop.py','src/scarx/evaluation/r8_benchmark_integrity.py')}}

def assert_lock(lock):
    if lock_sources()!=lock:raise ValueError('Source or external pin changed')
    if subprocess.check_output(['git','-C',str(PEER),'rev-parse','HEAD'],text=True).strip()!=PEER_SHA:raise ValueError('Peer moved')
    if subprocess.check_output(['git','-C',str(PEER),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Peer dirty')

def choose(rows):
    def mean(method,task,key):
        group=[r['metrics'][key] for r in rows if r['method']==method and r['task']==task and r['budget']=='equal_updates' and not r['shuffled']]
        if len(group)!=3:raise ValueError('Incomplete method group')
        return float(np.mean(group))
    def rank(method):
        diffs=[mean(method,t,k)-max(mean(e,t,k) for e in ('own','stop')) for t in TASKS[:3] for k in ('id_direct','id_probe')]
        direct=[mean(method,t,'id_direct')-max(mean(e,t,'id_direct') for e in ('own','stop')) for t in TASKS[:3]]
        return min(diffs),float(np.mean(direct))
    return max(sorted(('eprop_matched','eprop_aligned','braintrace')),key=rank)

def main():
    if git('status','--porcelain').strip():raise ValueError('Commit protocol/source before opening DEV')
    reserve=ROOT/'local/literature-reference-dev-v1-reservation';reserve.parent.mkdir(exist_ok=True);reserve.mkdir(exist_ok=False)
    run=ROOT/'runs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-literature-reference-dev-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    for name in ('jobs','datasets','stress'): (run/name).mkdir()
    (reserve/'run.txt').write_text(str(run),encoding='utf-8')
    cap=apply_memory_limit();start=time.perf_counter();deadline=start+3000;lock=lock_sources();assert_lock(lock)
    manifest={'source_commit':git('rev-parse','HEAD'),'main_before':git('rev-parse','main'),'protocol':PROTOCOL,
       'seeds':list(SEEDS),'methods':METHODS,'source_lock':lock,'fresh_confirmation':False,'pid':os.getpid(),
       'environment':{'python':sys.version,'numpy':np.__version__,'psutil':psutil.__version__,'platform':platform.platform(),'threads':1}}
    write(run/'manifest.json',manifest);write(ROOT/'local/literature-observer-active.json',{'run':str(run)})
    rows=[];status={'run':run.name,'status':'PREFLIGHT','completed_jobs':0,'total_jobs':132,
       'model':'Unchanged H48 ALIF; single-trace literature adapters','fresh_confirmation':False,'observer':'http://127.0.0.1:8772'}
    events={'learning':[],'answers':[]};error=None;nominee=None
    def publish():write(run/'progress.json',status);write(run/'events.json',events)
    publish();print(json.dumps({'run':str(run),'pid':os.getpid(),'observer':status['observer']}),flush=True)
    monitor=ConfirmationResources(interval=.05)
    def check(end=deadline):
        owner=ROOT/'local/literature-observer-owner.json'
        if monitor.observer_identity is None and owner.exists():
            rec=json.loads(owner.read_text(encoding='utf-8'));p=psutil.Process(rec['pid'])
            if Path(p.cwd()).resolve()!=ROOT or not any(arg.endswith('serve_literature_observer.py') for arg in p.cmdline()):raise ValueError('Observer ownership')
            monitor.observer_identity=(p.pid,p.create_time())
        monitor.check()
        if time.perf_counter()>min(end,deadline):raise TimeoutError('Registered time limit')
    with monitor:
      try:
        bank={};hashes={}
        for task in TASKS:
          for seed in SEEDS:
            d=dataset(task,seed);bank[(task,seed)]=d
            arrays={split+'_'+name:v for split in d for name,v in zip(('x','y'),d[split][:2])}
            path=run/'datasets'/f'{task}-{seed}.npz';np.savez(path,**arrays);hashes[path.name]=sha(path.read_bytes())
            sets=[{sha(x.tobytes()) for x in d[s][0]} for s in d]
            if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]):raise ValueError('Split overlap')
            check()
        write(run/'dataset-hashes.json',hashes)
        def cell(method,task,seed,budget='equal_updates',shuffle=False):
            assert_lock(lock);began=time.perf_counter();end=min(deadline,began+120);d=bank[(task,seed)]
            folder=run/'jobs'/f'{method}-{task}-{seed}-{budget}{"-shuffle" if shuffle else ""}'
            folder.mkdir();m=factory(method,seed,d,shuffle);initial=m.dumps();(folder/'initial.zip').write_bytes(initial)
            status.update(status='TRAINING',task=task,seed=seed,arm=method+('-shuffle' if shuffle else ''),budget=budget);publish()
            def progress(event):
                check(end);event={'task':task,'seed':seed,'arm':status['arm'],'budget':budget,**event}
                events['learning']=(events['learning']+[event])[-32:];status.update(latest_learning=event,resources=monitor.report());publish()
            training,history=train(m,*d['train'][:2],budget,end,progress)
            np.savez(folder/'learning.npz',history=history,columns=np.array(['step','loss','physical_hidden_norm','physical_readout_norm','train_cpu_seconds']))
            metrics,predictions,interventions,family=evaluate(m,d);blob=m.dumps();(folder/'model.zip').write_bytes(blob)
            if type(m).loads(blob).dumps()!=blob:raise ValueError('Checkpoint roundtrip')
            hidden_unchanged=bool(np.array_equal(m.core.w_in,m.w_in_0) and np.array_equal(m.core.w_rec,m.w_rec_0))
            if method=='readout_only' and (not hidden_unchanged or m.hidden_updates or np.any(history[:,2])):raise ValueError('Control hidden moved')
            row={'method':method,'task':task,'seed':seed,'budget':budget,'shuffled':shuffle,'metrics':metrics,'predictions':predictions,
               'interventions':interventions,'training':training,'mean_physical_update_norm':float(np.mean(history[:,2])),
               'checkpoint_sha256':sha(blob),'initial_checkpoint_sha256':sha(initial),'array_bytes':m.memory_bytes(),
               'signal_rng':m.signal_rng.bit_generator.state,'hidden_unchanged':hidden_unchanged,
               'wall_seconds':time.perf_counter()-began,'resources':monitor.report(),'folder':folder.name}
            write(folder/'result.json',row);rows.append(row)
            for j in range(4):events['answers'].append({'task':task,'seed':seed,'arm':method,'budget':budget,
              'input':d['id'][2][j],'prediction':predictions['id']['direct'][j],'target':predictions['id']['target'][j],'qwen_output':None})
            events['answers']=events['answers'][-20:];status.update(completed_jobs=len(rows),latest_metrics=metrics,resources=monitor.report());publish();check(end)
            print(json.dumps({'completed':len(rows),'method':method,'task':task,'seed':seed,'budget':budget,'shuffled':shuffle}),flush=True)
        for method in METHODS:
          for task in TASKS:
            for seed in SEEDS:cell(method,task,seed)
        nominee=choose(rows);write(run/'nominee-lock.json',{'method':nominee,'scope':'one global reference, no per-task choice'})
        for task in TASKS:
          for seed in SEEDS:cell(nominee,task,seed,shuffle=True)
        for method in ('own','stop',nominee,'readout_only'):
          for task in TASKS:
            for seed in SEEDS:cell(method,task,seed,'equal_cpu')
        for method in ('own','stop',nominee):
          for task in TASKS:
            for seed in SEEDS:
                row=next(r for r in rows if r['method']==method and r['task']==task and r['seed']==seed and r['budget']=='equal_updates' and not r['shuffled'])
                template=factory(method,seed,bank[(task,seed)]);m=type(template).loads((run/'jobs'/row['folder']/'model.zip').read_bytes())
                d=bank[(task,seed)];fit=m.features_many(d['refit'][0]);stress={}
                for name,(xs,ys) in stress_cases(task,d).items():
                    f=m.features_many(xs);dp=direct_predict(m,f);pp=refit_predict(fit,d['refit'][1],f)
                    stress[name]={'direct':dp.tolist(),'probe':pp.tolist(),'target':ys.tolist(),'direct_accuracy':float(np.mean(dp==ys)),'probe_accuracy':float(np.mean(pp==ys))}
                    if name=='query_flip':stress[name]['paired_direct_flip_fraction']=float(np.mean(dp==1-m.predict_many(d['id'][0][:128])))
                write(run/'stress'/f'{method}-{task}-{seed}.json',stress);check()
        assert_lock(lock);check();status['status']='COMPLETE_AWAITING_AUDIT'
      except Exception as exc:error=type(exc).__name__+': '+str(exc);status.update(status='STOPPED',error=error)
    result={'error':error,'completed_jobs':len(rows),'nominee':nominee,'source_commit':manifest['source_commit'],
       'resources':monitor.report(),'cap':cap,'wall_seconds':time.perf_counter()-start,'fresh_confirmation':False,
       'production_ready':False,'main_unchanged':git('rev-parse','main')==manifest['main_before']}
    write(run/'result.json',result);status['result']=result;publish();print(json.dumps(result),flush=True)
    return int(error is not None)

if __name__=='__main__':raise SystemExit(main())
