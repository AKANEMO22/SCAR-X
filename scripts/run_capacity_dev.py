"""Approved finite capacity experiment. True training, immutable old H48 evidence."""
from pathlib import Path
import json,os,platform,shutil,subprocess,sys,time,uuid
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import psutil
from scarx.neural.core import Config
from scarx.evaluation.capacity import CapacityOwn,CapacityACR
from scarx.evaluation.acr import AdaptiveCreditRouting
from scarx.evaluation.r8_matched_credit import PEER,PEER_SHA
from scarx.evaluation.r8_matched_study import SEEDS,TASKS,dataset
from scarx.evaluation.acr_study import evaluate,sha
from scarx.evaluation.confirmation_resources import ConfirmationResources
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import activity
from run_r8_matched_dev import train,atomic_json as write

PROTOCOL='experiments/capacity-h64-h128-approved-v1.md'
PARENT=ROOT.parent/'scarx-acr-20261112'/'runs/20261112T121634Z-acr-dev-v1-resume-59ced4'
ARMS=('own','groups_up','readout_only','groups_shuffle')
FILES=(PROTOCOL,'src/scarx/evaluation/capacity.py','src/scarx/evaluation/forensic/r8_snapshot.py',
 'src/scarx/evaluation/acr.py','src/scarx/evaluation/r8_matched_credit.py','src/scarx/evaluation/r8_matched_study.py',
 'src/scarx/evaluation/readout_eligibility.py','src/scarx/evaluation/acr_study.py','src/scarx/neural/core.py',
 'src/scarx/evaluation/confirmation_resources.py','src/scarx/runtime/limits.py','src/scarx/runtime/resources.py',
 'scripts/run_capacity_dev.py','scripts/run_r8_matched_dev.py','tests/test_capacity.py')
MAP={'own':('static_0','learned_mtsg'),'groups_up':('groups_up','learned_mtsg'),
     'readout_only':('readout_only','readout_only'),'groups_shuffle':('groups_up','shuffled_signal')}

def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
def model(h,arm,seed,data):
    c=Config(inputs=8,outputs=2,hidden=h,fan_in=8,plastic_rows=12,seed=seed)
    cls=CapacityACR if arm.startswith('groups') else CapacityOwn
    condition='shuffled_signal' if arm=='groups_shuffle' else 'readout_only' if arm=='readout_only' else 'learned_mtsg'
    m=cls(c,condition);m.calibrate(data['train'][0][:32]);return m
def source_lock():
    return {'files':{p:sha((ROOT/p).read_bytes()) for p in FILES},
        'peer':{p:sha((PEER/p).read_bytes()) for p in ('src/scarx/neural/r8_mtsg_eprop.py','src/scarx/evaluation/r8_benchmark_integrity.py')}}
def check_lock(lock):
    if source_lock()!=lock:raise ValueError('Registered source changed')
    if subprocess.check_output(['git','-C',str(PEER),'rev-parse','HEAD'],text=True).strip()!=PEER_SHA or subprocess.check_output(['git','-C',str(PEER),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Peer source moved or dirty')

def main():
    dirty=git('status','--porcelain','--untracked-files=no')
    if dirty:raise ValueError('Commit tracked source/protocol before quality study')
    reserve=ROOT/'local/capacity-dev-v1-reservation';reserve.parent.mkdir(exist_ok=True);reserve.mkdir(exist_ok=False)
    run=ROOT/'runs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-capacity-dev-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    for d in ('datasets','jobs','reused-h48'): (run/d).mkdir()
    (reserve/'run.txt').write_text(str(run),encoding='utf-8')
    start=time.perf_counter();deadline=start+3600;cap=apply_memory_limit();lock=source_lock();check_lock(lock)
    old= json.loads((PARENT/'audit-v1.json').read_text(encoding='utf-8'))
    if old['integrity']!='PASS' or old['models']!=375:raise ValueError('Verified completed H48 evidence required')
    manifest={'source_commit':git('rev-parse','HEAD'),'base_sha':'22bb9781f4ed8f60f7180bdb56b6570e64e1f503',
       'main_before':git('rev-parse','main'),'protocol':PROTOCOL,'source_lock':lock,'peer_sha':PEER_SHA,
       'height_grid':[48,64,128],'seeds':list(SEEDS),'arms':ARMS,'max_new_jobs':204,'expected_records':252,
       'parent_run':PARENT.name,'parent_manifest_sha256':sha((PARENT/'manifest.json').read_bytes()),
       'parent_audit_sha256':sha((PARENT/'audit-v1.json').read_bytes()),'worker_limit_seconds':3600,
       'user_approval':'ok tôi duyệt thử H64 và H128, bắt đầu chạy đi, giữ nguyên ALIF và học synapse cục bộ, cho quá trình học với kết quả lên localhost để tôi xem luôn',
       'fresh_confirmation':False,'p2':False,'pid':os.getpid(),
       'environment':{'python':sys.version,'numpy':np.__version__,'psutil':psutil.__version__,'platform':platform.platform(),'threads':1}}
    write(run/'manifest.json',manifest)
    rows=[];events={'learning':[],'answers':[],'completed':[],'activity':[]}
    status={'run':run.name,'status':'DATA_PREFLIGHT','completed_jobs':0,'total_jobs':204,'reused_cells':0,
       'height':None,'core':'Unchanged hard ALIF, local synaptic learning','model':'Capacity research, no Hermes/live model changes',
       'new_confirmation':False,'observer':'http://127.0.0.1:8773'}
    def publish():write(run/'progress.json',status);write(run/'events.json',events)
    def technical(event):
        events['activity'].append({'time':time.time(),'event':event,'scope':'Technical activity, not neural thought'});events['activity']=events['activity'][-12:];publish()
    publish();write(ROOT/'local/capacity-observer-active.json',{'run':str(run)})
    technical('Protocol/source locked. H64/H128 approved; no primary H48 training repeated.')
    activity(run,'Bắt đầu capacity DEV đã duyệt','H64/H128; unchangedALIF/localcredit; fixedseeds/controls/budgets; no main/Hermes.',status='started',evidence='manifest.json')
    print(json.dumps({'run':str(run),'pid':os.getpid(),'observer':status['observer']}),flush=True)
    monitor=ConfirmationResources(interval=.05);error=None;new_jobs=0;reuse={};bank={};hashes={}
    def check(end=None):
        owner=ROOT/'local/capacity-observer-owner.json'
        if monitor.observer_identity is None and owner.exists():
            rec=json.loads(owner.read_text(encoding='utf-8'))
            try:
                p=psutil.Process(rec['pid'])
                if Path(p.cwd()).resolve()==ROOT and any(x.endswith('serve_capacity_observer.py') for x in p.cmdline()):monitor.observer_identity=(p.pid,p.create_time())
            except psutil.NoSuchProcess:pass
        monitor.check()
        if time.perf_counter()>min(deadline,end or deadline):raise TimeoutError('Registered capacity time limit')
    def event_result(r,data):
        events['completed'].append({'height':r['height'],'arm':r['arm'],'task':r['task'],'seed':r['seed'],'budget':r['budget'],
          'metrics':r['metrics'],'episodes':r['training']['episodes'],'cpu_seconds':r['training']['train_cpu_seconds'],'reused':r['reused']})
        for j in range(3):events['answers'].append({'height':r['height'],'arm':r['arm'],'task':r['task'],'seed':r['seed'],
           'input':data['id'][2][j],'prediction':r['predictions']['id']['direct'][j],'target':r['predictions']['id']['target'][j]})
        events['answers']=events['answers'][-20:];publish()
    with monitor:
      try:
        for task in TASKS:
          for seed in SEEDS:
            d=dataset(task,seed);bank[(task,seed)]=d
            original=PARENT/'datasets'/f'{task}-{seed}.npz'
            with np.load(original,allow_pickle=False) as z:
                for split in d:
                    if not np.array_equal(z[split+'_x'],d[split][0]) or not np.array_equal(z[split+'_y'],d[split][1]):raise ValueError('Approved frozen data changed')
            path=run/'datasets'/original.name;shutil.copyfile(original,path);hashes[path.name]=sha(path.read_bytes());check()
        write(run/'dataset-hashes.json',hashes)
        # Quality reuse only. Old CPU numbers are preserved as historical,
        # not treated as contemporaneous single-trace efficiency measurements.
        for arm in ARMS:
          for task in TASKS:
            for seed in SEEDS:
                profile,condition=MAP[arm]
                candidates=[]
                for path in (PARENT/'jobs').glob(f'{task}-{seed}-*-equal_updates/result-0.json'):
                    r=json.loads(path.read_text(encoding='utf-8'))
                    if (r['profile'],r['condition'])==(profile,condition):candidates.append((path,r))
                if len(candidates)!=1:raise ValueError('Ambiguous H48 reuse cell')
                path,oldrow=candidates[0];folder=run/'reused-h48'/f'{arm}-{task}-{seed}';shutil.copytree(path.parent,folder)
                for file in path.parent.iterdir():reuse[str((folder/file.name).relative_to(run)).replace('\\','/')]={'source_path':str(file),'sha256':sha(file.read_bytes())}
                blob=(folder/'model-0.zip').read_bytes()
                if sha(blob)!=oldrow['checkpoint_sha256']:raise ValueError('H48 checkpoint changed')
                legacy=AdaptiveCreditRouting.loads(blob)
                template=model(48,arm,seed,bank[(task,seed)])
                initial=AdaptiveCreditRouting.loads((folder/'initial-0.zip').read_bytes())
                for name,v in template.core.arrays().items():np.testing.assert_array_equal(v,initial.core.arrays()[name])
                for name in ('f_mean','f_var','syn_trace','syn_energy'):np.testing.assert_array_equal(getattr(template,name),getattr(initial,name))
                row={'height':48,'arm':arm,'task':task,'seed':seed,'budget':'equal_updates','reused':True,
                   'metrics':oldrow['metrics'],'predictions':oldrow['predictions'],'interventions':oldrow['interventions'],
                   'training':oldrow['training'],'checkpoint_sha256':oldrow['checkpoint_sha256'],'folder':str(folder.relative_to(run)).replace('\\','/'),
                   'source_result_sha256':sha(path.read_bytes()),'array_bytes':template.memory_bytes(),
                   'reused_source_array_bytes':legacy.memory_bytes(),'resources':oldrow['resources'],
                   'cpu_scope':'Historical dual-stream H48 measurements; not used for single-trace CPU claim'}
                write(folder/'capacity-result.json',row);rows.append(row);status['reused_cells']+=1;event_result(row,bank[(task,seed)]);check()
        write(run/'reused-h48-files.json',reuse)
        technical('48 H48 primary cells reused with hashes/init parity. Starting new H64/H128 primary learning.')
        def cell(h,arm,task,seed,budget):
            nonlocal new_jobs
            check_lock(lock);begin=time.perf_counter();end=min(deadline,begin+120);d=bank[(task,seed)]
            folder=run/'jobs'/f'H{h}-{arm}-{task}-{seed}-{budget}';folder.mkdir()
            m=model(h,arm,seed,d);initial=m.dumps();(folder/'initial.zip').write_bytes(initial)
            status.update(status='TRAINING',height=h,neurons_allocated=h,synapses=h*16,task=task,seed=seed,arm=arm,budget=budget);publish()
            with (folder/'learning-progress.jsonl').open('x',encoding='utf-8') as log:
                def progress(e):
                    check(end);e={'height':h,'arm':arm,'task':task,'seed':seed,'budget':budget,
                       'spiking_neurons_at_last_tick':int(m.core.spikes.sum()),'hidden_updates':m.hidden_updates,**e}
                    log.write(json.dumps(e,allow_nan=False)+'\n');log.flush()
                    events['learning']=(events['learning']+[e])[-40:];status.update(latest_learning=e,resources=monitor.report());publish()
                training,history=train(m,*d['train'][:2],budget,end,progress)
            np.savez(folder/'learning.npz',history=history,columns=np.array(['step','loss','physical_hidden_norm','physical_readout_norm','train_cpu_seconds']))
            metrics,predictions,interventions,_=evaluate(m,d);blob=m.dumps();(folder/'model.zip').write_bytes(blob)
            if type(m).loads(blob).dumps()!=blob:raise ValueError('Capacity checkpoint roundtrip')
            if arm=='readout_only' and (m.hidden_updates or not np.array_equal(m.core.w_in,m.w_in_0) or not np.array_equal(m.core.w_rec,m.w_rec_0) or np.any(history[:,2])):raise ValueError('Readout-only hidden changed')
            r={'height':h,'arm':arm,'task':task,'seed':seed,'budget':budget,'reused':False,
               'metrics':metrics,'predictions':predictions,'interventions':interventions,'training':training,
               'checkpoint_sha256':sha(blob),'initial_sha256':sha(initial),'signal_rng_state':m.signal_rng.bit_generator.state,
               'array_bytes':m.memory_bytes(),'mean_physical_hidden_norm':float(history[:,2].mean()),
               'resources':monitor.report(),'folder':str(folder.relative_to(run)).replace('\\','/'),'wall_seconds':time.perf_counter()-begin}
            write(folder/'result.json',r);rows.append(r);new_jobs+=1;status.update(completed_jobs=new_jobs,latest_metrics=metrics,resources=monitor.report())
            event_result(r,d);check(end);print(json.dumps({'completed':new_jobs,'total':204,'height':h,'arm':arm,'task':task,'seed':seed,'budget':budget}),flush=True)
        for h in (64,128):
          for arm in ARMS:
            for task in TASKS:
              for seed in SEEDS:cell(h,arm,task,seed,'equal_updates')
        technical('Primary learning complete. Measuring contemporaneous equal-CPU baselines including H48; no candidate selection.')
        for h in (48,64,128):
          for arm in ('own','groups_up','readout_only'):
            for task in TASKS:
              for seed in SEEDS:cell(h,arm,task,seed,'equal_cpu')
        if new_jobs!=204 or len(rows)!=252:raise ValueError('Incomplete capacity grid')
        check_lock(lock);check();status['status']='COMPLETE_AWAITING_AUDIT'
      except Exception as exc:error=type(exc).__name__+': '+str(exc);status.update(status='STOPPED',error=error)
    result={'error':error,'new_jobs':new_jobs,'row_count':len(rows),'reused_h48_cells':status['reused_cells'],
       'resources':monitor.report(),'cap':cap,'wall_seconds':time.perf_counter()-start,'main_unchanged':git('rev-parse','main')==manifest['main_before'],
       'source_commit':manifest['source_commit'],'fresh_confirmation':False,'p2':False,'production_ready':False}
    write(run/'result.json',result);status.update(result=result,resources=monitor.report());publish()
    print(json.dumps(result),flush=True);return int(error is not None)

if __name__=='__main__':raise SystemExit(main())
