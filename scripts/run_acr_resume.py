"""Continue the interrupted registered DEV; never repeat completed training cells."""
from pathlib import Path
import copy,hashlib,json,os,platform,subprocess,sys,time,uuid,shutil
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import psutil
from scarx.evaluation.acr import AdaptiveCreditRouting as ACR,AB_PROFILES,C_PROFILES,hidden_matrix,matched_update,NormMismatch
from scarx.evaluation.acr_study import TASKS,SEEDS,ENVS,build_data,evaluate,nominate,broad_checks,stress_cases,sha
from scarx.evaluation.r8_matched_credit import PEER,PEER_SHA,MatchedLocalCredit
from scarx.evaluation.r8_matched_study import direct_predict,refit_predict
from scarx.evaluation.confirmation_resources import ConfirmationResources
from scarx.neural.core import Config
from scarx.runtime.limits import apply_memory_limit

PROTOCOL='experiments/acr-dev-v1.md'
PARENT_RUN=ROOT/'runs/20261112T073944Z-acr-dev-v1-f947fb'
RESUME_PROTOCOL='experiments/acr-resume-20261112.md'
FILES=('src/scarx/evaluation/acr.py','src/scarx/evaluation/acr_study.py',
       'src/scarx/evaluation/forensic/r8_snapshot.py','src/scarx/evaluation/r8_matched_credit.py',
       'src/scarx/evaluation/r8_stopped_reset.py','src/scarx/evaluation/r8_matched_study.py',
       'src/scarx/evaluation/readout_eligibility.py','src/scarx/neural/core.py',
       'src/scarx/evaluation/confirmation_resources.py','src/scarx/runtime/resources.py',
       'src/scarx/runtime/limits.py','scripts/run_acr_dev.py',
       'tests/test_acr.py','tests/test_acr_study.py',PROTOCOL,RESUME_PROTOCOL,'scripts/run_acr_resume.py')
PEER_FILES=('src/scarx/neural/r8_mtsg_eprop.py','src/scarx/evaluation/r8_benchmark_integrity.py')

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)

def source_check(lock):
    for f,h in lock['source_sha256'].items():
        if sha((ROOT/f).read_bytes())!=h:raise ValueError('Source changed: '+f)
    for f,h in lock['peer_sha256'].items():
        if sha((PEER/f).read_bytes())!=h:raise ValueError('Peer source changed: '+f)
    if subprocess.check_output(['git','-C',str(PEER),'rev-parse','HEAD'],text=True).strip()!=PEER_SHA:raise ValueError('Peer HEAD moved')
    if subprocess.check_output(['git','-C',str(PEER),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Peer dirty')

def make(seed,p,data,condition='learned_mtsg'):
    m=ACR(Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed),p,condition)
    m.calibrate(data['train'][0][:32]);return m

def main():
    began=time.perf_counter();cap=apply_memory_limit()
    if git('status','--porcelain').strip():raise ValueError('Commit code/protocol/tests before quality outcomes')
    reserve=ROOT/'local/acr-resume-v1-reservation';reserve.parent.mkdir(exist_ok=True);reserve.mkdir(exist_ok=False)
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());run=ROOT/'runs'/f'{stamp}-acr-dev-v1-resume-{uuid.uuid4().hex[:6]}'
    run.mkdir();(reserve/'run.txt').write_text(str(run),encoding='utf-8')
    for name in ('jobs','datasets','diagnostics'): (run/name).mkdir()
    parent_manifest=json.loads((PARENT_RUN/'manifest.json').read_text(encoding='utf-8'))
    source_check(parent_manifest)
    old_progress=json.loads((PARENT_RUN/'progress.json').read_text(encoding='utf-8'))
    try:
        owner=psutil.Process(parent_manifest['pid'])
        if any(arg.endswith('run_acr_dev.py') for arg in owner.cmdline()):raise ValueError('Original worker still alive')
    except psutil.NoSuchProcess:pass
    prior=sorted((PARENT_RUN/'jobs').glob('*/result-0.json'))
    if len(prior)!=55 or len(list((PARENT_RUN/'jobs').iterdir()))!=55:raise ValueError('Interrupted grid changed; inspect before continuation')
    copied=[];copied_files={}
    for path in prior:
        row=json.loads(path.read_text(encoding='utf-8'))
        if row['budget']!='equal_updates' or row['training']['episodes']!=1024:raise ValueError('Only completed primary cells reusable')
        if sha((path.parent/'model-0.zip').read_bytes())!=row['checkpoint_sha256']:raise ValueError('Prior checkpoint changed')
        if {p.name for p in path.parent.iterdir()}!={'initial-0.zip','learning-0.npz','model-0.zip','result-0.json'}:
            raise ValueError('Unexpected prior cell members')
        for prior_file in path.parent.iterdir():
            relative=str(prior_file.relative_to(PARENT_RUN)).replace('\\','/')
            copied_files[relative]=sha(prior_file.read_bytes())
        shutil.copytree(path.parent,run/'jobs'/path.parent.name);copied.append(row)
    for relative,h in copied_files.items():
        if sha((run/relative).read_bytes())!=h:raise ValueError('Copied evidence changed')
    consumed=old_progress['resources']['elapsed_seconds']
    # Snapshot is a lower bound, reserve an extra minute for unobserved shutdown.
    remaining=9000-consumed-60
    deadline=began+remaining
    lock={'source_commit':git('rev-parse','HEAD'),'base_sha':'93b5b96bebaf22dbc39a3f96b2cac7c917d5e377',
          'main_before':git('rev-parse','main'),'source_sha256':{f:sha((ROOT/f).read_bytes()) for f in FILES},
          'peer_sha':PEER_SHA,'peer_sha256':{f:sha((PEER/f).read_bytes()) for f in PEER_FILES},
          'environment':{'python':sys.version,'numpy':np.__version__,'psutil':psutil.__version__,'platform':platform.platform(),
                         'numerical_threads':1,'cpu_endpoints':'Dual-stream ACR endpoints; not original single-stream efficiency comparison'},
          'protocol':PROTOCOL,'resume_protocol':RESUME_PROTOCOL,'seeds':list(SEEDS),'confirmation':False,
          'limit_seconds':remaining,'pid':os.getpid(),'parent_run':PARENT_RUN.name,
          'parent_source_commit':parent_manifest['source_commit'],'parent_manifest_sha256':sha((PARENT_RUN/'manifest.json').read_bytes()),
          'reused_completed_cells':55,'reused_files_sha256':copied_files,
          'prior_elapsed_lower_bound':consumed,'shutdown_allowance_seconds':60}
    source_check(lock);write(run/'manifest.json',lock)
    rows=copied;events={'learning':[],'answers':[]};status={'run':run.name,'status':'RESUME_PREFLIGHT','completed_jobs':55,
       'total_jobs':345,'maximum_trained_models':375,'fresh_confirmation':False,'model':'One H48 ALIF forward, dual eligibility; research only','observer':'http://127.0.0.1:8771'}
    def publish():write(run/'progress.json',status);write(run/'events.json',events)
    publish();write(ROOT/'local/acr-observer-active.json',{'run':str(run)})
    print(json.dumps({'run':str(run),'pid':os.getpid(),'observer':status['observer']}),flush=True)
    all_data={};data_hashes={};error=None;nominee=None;norm_valid=True;check=None
    monitor=ConfirmationResources(interval=.05)
    def safety(job_deadline=None):
        owner=ROOT/'local/acr-observer-owner.json'
        if monitor.observer_identity is None and owner.exists():
            identity=json.loads(owner.read_text(encoding='utf-8'));process=psutil.Process(identity['pid'])
            if Path(process.cwd()).resolve()!=ROOT or not any(arg.endswith('serve_acr_observer.py') for arg in process.cmdline()):
                raise ValueError('ACR observer ownership mismatch')
            monitor.observer_identity=(process.pid,process.create_time())
        monitor.check()
        if time.perf_counter()>min(deadline,job_deadline or deadline):raise TimeoutError('Registered DEV time limit')
    with monitor:
      try:
        for seed in SEEDS:
            bank=build_data(seed)
            for task,d in bank.items():
                all_data[(task,seed)]=d
                fields={split+'_'+k:v for split in ('train','refit','id','ood') for k,v in zip(('x','y'),d[split][:2])}
                for split in ('train','refit','id','ood'):
                    if split+'_families' in d:fields[split+'_families']=d[split+'_families']
                path=run/'datasets'/f'{task}-{seed}.npz'
                with np.load(PARENT_RUN/'datasets'/path.name,allow_pickle=False) as original:
                    if set(original.files)!=set(fields) or any(not np.array_equal(original[key],value) for key,value in fields.items()):raise ValueError('Frozen parent data no longer matches source')
                shutil.copyfile(PARENT_RUN/'datasets'/path.name,path);data_hashes[path.name]=sha(path.read_bytes())
                sets=[{sha(seq.tobytes()) for seq in d[s][0]} for s in ('train','refit','id','ood')]
                if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]):raise ValueError('Split overlap')
            safety()
        write(run/'dataset-hashes.json',data_hashes)

        def job(task,seed,profile,budget='equal_updates',condition='learned_mtsg',norm_pair=False):
            nonlocal norm_valid
            source_check(lock);start=time.perf_counter();end=min(deadline,start+120);d=all_data[(task,seed)]
            label=profile+('-signal_shuffle' if condition=='shuffled_signal' else '')
            key=f'{task}-{seed}-{label}-{budget}';folder=run/'jobs'/key
            if folder.exists():
                existing=json.loads((folder/'result-0.json').read_text(encoding='utf-8'))
                if budget!='equal_updates' or existing['training']['episodes']!=1024:raise ValueError('Refuse replay/incomplete prior cell')
                return existing  # Reused bytes; no calibration/forward/learning repeated.
            folder.mkdir()
            m=make(seed,profile,d,condition);models=[m]
            if norm_pair:models.append(make(seed,'static_0',d))
            for i,model in enumerate(models):(folder/f'initial-{i}.zip').write_bytes(model.dumps())
            history=[[] for _ in models];cpu=[0. for _ in models];invalid=None;norm_records=[]
            status.update(status='TRAINING',task=task,seed=seed,arm=label,budget=budget);publish()
            limit=4096 if budget=='equal_cpu' else 1024
            for step in range(limit):
                safety(end);x=d['train'][0][step%1024];y=int(d['train'][1][step%1024])
                before=[hidden_matrix(model) for model in models]
                if norm_pair:
                    tick=time.process_time()
                    try:la,lb,radius,na,nb=matched_update(*models,x,y)
                    except NormMismatch as exc:
                        invalid=str(exc);norm_valid=False;break
                    elapsed=time.process_time()-tick;cpu[0]+=elapsed/2;cpu[1]+=elapsed/2
                    # Allocation half is bookkeeping only; do not call it measured per-model CPU.
                    losses=[la,lb];norm_records.append([radius,na,nb])
                else:
                    tick=time.process_time();losses=[m.learn_episode(x,y)];cpu[0]+=time.process_time()-tick
                for i,(model,loss) in enumerate(zip(models,losses)):
                    norm=float(np.linalg.norm(hidden_matrix(model).astype(float)-before[i].astype(float)))
                    if not np.isfinite(loss+norm):raise ValueError('Nonfinite learning')
                    history[i].append([step+1,loss,norm,cpu[i]])
                if step==0 or (step+1)%64==0:
                    event={'task':task,'seed':seed,'arm':label,'budget':budget,'step':step+1,'loss':losses[0],
                           'hidden_update_norm':history[0][-1][2],'train_cpu_seconds':cpu[0]}
                    events['learning']=(events['learning']+[event])[-32:]
                    status.update(latest_learning=event,resources=monitor.report());publish()
                if budget=='equal_cpu' and cpu[0]>=.5:break
            for i,model in enumerate(models):
                np.savez(folder/f'learning-{i}.npz',history=np.asarray(history[i],np.float64),
                         columns=np.array(['step','loss','physical_hidden_norm','train_cpu_seconds']))
                (folder/f'model-{i}.zip').write_bytes(model.dumps())
            if norm_pair:np.savez(folder/'norm-pairs.npz',radius_raw=np.asarray(norm_records,np.float64))
            if invalid is not None:
                row={'task':task,'seed':seed,'profile':profile,'condition':condition,'budget':budget,'norm_valid':False,
                     'error':invalid,'completed_updates':len(history[0]),'actual_models':2}
                write(folder/'invalid-control.json',row);status['completed_jobs']+=1;publish();return row
            outputs=[]
            for i,model in enumerate(models):
                metrics,predictions,interventions,family=evaluate(model,d);blob=model.dumps();(folder/f'model-{i}.zip').write_bytes(blob)
                if ACR.loads(blob).dumps()!=blob:raise ValueError('Checkpoint roundtrip failed')
                unchanged=bool(np.array_equal(model.core.w_in,model.w_in_0) and np.array_equal(model.core.w_rec,model.w_rec_0))
                if profile=='readout_only' and (not unchanged or model.hidden_updates):raise ValueError('Frozen hidden changed')
                row={'task':task,'seed':seed,'profile':profile if i==0 else 'static_0','condition':model.condition,
                     'budget':budget,'metrics':metrics,'predictions':predictions,'interventions':interventions,'family_metrics':family,
                     'training':{'episodes':len(history[i]),'hidden_updates':model.hidden_updates,'train_cpu_seconds':cpu[i],
                         'cpu_is_pair_half_allocation':norm_pair,'mean_update_norm':float(np.mean(np.asarray(history[i])[:,2])),
                         'train_wall_and_evaluation_seconds':time.perf_counter()-start},
                     'checkpoint_sha256':sha(blob),'initial_checkpoint_sha256':sha((folder/f'initial-{i}.zip').read_bytes()),
                     'signal_rng_state':model.signal_rng.bit_generator.state,'router_rng_state':model.router_rng.bit_generator.state,
                     'hidden_unchanged':unchanged,'array_bytes':model.memory_bytes(),'extra_array_bytes':model.memory_bytes()-MatchedLocalCredit(model.config).memory_bytes(),
                     'norm_valid':True if norm_pair else None,'resources':monitor.report(),'path':str(folder.relative_to(run)),'model_index':i}
                write(folder/f'result-{i}.json',row);rows.append(row);outputs.append(row)
            for j in range(6):
                events['answers'].append({'task':task,'seed':seed,'arm':label,'budget':budget,'input':d['id'][2][j],
                    'prediction':outputs[0]['predictions']['id']['direct'][j],'target':outputs[0]['predictions']['id']['target'][j],
                    'qwen_output':None,'scope':'Actual neural event output; no Qwen/LLM called'})
            events['answers']=events['answers'][-24:];status['latest_metrics']=outputs[0]['metrics'];status['completed_jobs']+=1
            status['resources']=monitor.report();publish();safety(end)
            print(json.dumps({'completed':status['completed_jobs'],'task':task,'seed':seed,'profile':label,'budget':budget}),flush=True)
            return outputs[0]

        for profile in AB_PROFILES+('readout_only',):
            for task in ENVS:
                for seed in SEEDS:job(task,seed,profile)
        ab_nominee=nominate(rows,AB_PROFILES+('readout_only',))
        uniform_broad=[p for p in AB_PROFILES[:5] if p not in ('static_0','static_100') and broad_checks(rows,p)['broad_numeric']]
        write(run/'stage-ab.json',{'nominee':ab_nominee,'uniform_broad':uniform_broad,'open_state':not bool(uniform_broad)})
        profiles=AB_PROFILES+('readout_only',)
        if not uniform_broad:
            for profile in C_PROFILES:
                for task in ENVS:
                    for seed in SEEDS:job(task,seed,profile)
            profiles+=C_PROFILES
        nominee=nominate(rows,profiles);check=broad_checks(rows,nominee)
        status['total_jobs']=165+(60 if not uniform_broad else 0)+(0 if nominee=='state' else 15)+15*len(dict.fromkeys(('static_50',nominee)))+75
        write(run/'nominee-lock.json',{'profile':nominee,'checks':check,'selection':'one global profile, never per-task'})
        shuffle_profile='state_signal_shuffle' if nominee=='state' else nominee
        if nominee!='state':
            for task in ENVS:
                for seed in SEEDS:job(task,seed,nominee,condition='shuffled_signal')
        for profile in dict.fromkeys(('static_50',nominee)):
            for task in ENVS:
                for seed in SEEDS:job(task,seed,profile,budget='norm_matched',norm_pair=True)
        for profile,condition in (('static_0','learned_mtsg'),('static_100','learned_mtsg'),(nominee,'learned_mtsg'),(nominee,'shuffled_signal'),('readout_only','learned_mtsg')):
            for task in ENVS:
                for seed in SEEDS:job(task,seed,profile,budget='equal_cpu',condition=condition)

        # Fixed final-checkpoint diagnostic clones, never fed back to candidates.
        for profile in dict.fromkeys(('static_0','static_100',nominee)):
          for task in TASKS:
            for seed in SEEDS:
                row=next(r for r in rows if r['profile']==profile and r['task']==task and r['seed']==seed and r['budget']=='equal_updates' and r['condition']=='learned_mtsg')
                blob=(run/row['path']/f"model-{row['model_index']}.zip").read_bytes();model=ACR.loads(blob);d=all_data[(task,seed)]
                fit=model.features_many(d['refit'][0]);stress={}
                for name,(xs,ys) in stress_cases(task,d).items():
                    f=model.features_many(xs);dp=direct_predict(model,f);pp=refit_predict(fit,d['refit'][1],f)
                    stress[name]={'direct':dp.tolist(),'probe':pp.tolist(),'target':ys.tolist(),
                        'direct_accuracy':float(np.mean(dp==ys)),'probe_accuracy':float(np.mean(pp==ys))}
                    if name=='query_flip':
                        original=model.predict_many(d['id'][0][:128])
                        stress[name]['paired_direct_flip_fraction']=float(np.mean(dp==1-original))
                write(run/'diagnostics'/f'{task}-{seed}-{profile}-stress.json',stress)
                for index in (0,63,511,1023):
                    m=ACR.loads(blob);x=d['train'][0][index];target=int(d['train'][1][index]);m.reset_episode();tape={}
                    for t,xrow in enumerate(x):
                        m.step(xrow)
                        for name,array in [('u',m.last_u),('theta',m.last_theta),('z',m.core.spikes),('adaptation',m.last_a),
                          ('own',m.e_f_own),('stop',m.e_f_stop),('mixed',m.e_f),('alpha',m.alpha),('q',m.router_q)]:tape.setdefault(name,[]).append(array.copy())
                    # Replay from identical pre-diagnostic RNG/state, not from
                    # the state whose routing RNG was already consumed by tape.
                    m=ACR.loads(blob);pre_prediction=m.predict(x);loss=m.learn_episode(x,target)
                    tape.update(signal=m.last_signal,hidden_delta=m.last_delta,loss=np.asarray(loss),target=np.asarray(target),
                        direct_prediction=np.asarray(pre_prediction),scope=np.asarray('Final checkpoint diagnostic clone, not original training trajectory'))
                    np.savez_compressed(run/'diagnostics'/f'{task}-{seed}-{profile}-{index}.npz',**{k:np.asarray(v) for k,v in tape.items()})
                    safety()
        source_check(lock);safety();status['status']='DEV_COMPLETE_AWAITING_INDEPENDENT_AUDIT'
      except Exception as exc:
        error=type(exc).__name__+': '+str(exc);status.update(status='DEV_STOPPED',error=error)
    result={'error':error,'nominee':nominee,'checks':check,'norm_controls_valid':norm_valid,'rows':len(rows),
        'completed_jobs':status['completed_jobs'],'resources':monitor.report(),'memory_cap':cap,'source_commit':lock['source_commit'],
        'main_unchanged':git('rev-parse','main')==lock['main_before'],'new_confirmation':False,'production_ready':False,
        'wall_seconds':time.perf_counter()-began,'reused_completed_cells':55,'prior_elapsed_lower_bound':consumed,'decision':'AWAITING_AUDIT' if error is None else 'INCOMPLETE_NO_ADVANCE'}
    write(run/'result.json',result);status.update(result=result,resources=monitor.report());publish();print(json.dumps(result),flush=True)
    return int(error is not None)

if __name__=='__main__':raise SystemExit(main())
