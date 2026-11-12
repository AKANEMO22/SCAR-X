"""Registered second local-trace investigation using frozen matched-study data."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.r8_matched_credit import PEER_SHA
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit,ARMS
from scarx.evaluation.r8_matched_study import TASKS,SEEDS,event_oracle,paired_statistics,holm
from scarx.evaluation.runs import new_run,write_json
from scarx.neural.core import Config
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import activity,append
from run_r8_matched_dev import atomic_json,train,evaluate


def summaries(rows,baseline):
    result={};pvalues={}
    for budget in ('equal_updates','equal_cpu'):
        tasks={}
        for task in TASKS:
            index={(r['seed'],r['arm']):r for r in rows if r['task']==task and r['budget']==budget}
            if len(index)!=len(SEEDS)*len(ARMS):continue
            means={arm:{key:float(np.mean([index[(s,arm)]['metrics'][key] for s in SEEDS])) for key in ('id_direct','id_probe','ood_direct','ood_probe')} for arm in ARMS}
            effects={
                'vs_own_reset':[index[(s,'multi_gated')]['metrics']['id_probe']-baseline[(task,s,'multi_gated',budget)]['metrics']['id_probe'] for s in SEEDS],
                'vs_shuffle':[index[(s,'multi_gated')]['metrics']['id_probe']-index[(s,'multi_gated_shuffle')]['metrics']['id_probe'] for s in SEEDS],
                'vs_readout':[index[(s,'multi_gated')]['metrics']['id_probe']-index[(s,'multi_gated_readout_only')]['metrics']['id_probe'] for s in SEEDS]}
            contrasts={name:paired_statistics(values) for name,values in effects.items()}
            if budget=='equal_updates' and task!='order':
                pvalues.update({task+'/'+name:c['exact_two_sided_p'] for name,c in contrasts.items()})
            tasks[task]={'arm_means':means,'contrasts':contrasts}
        result[budget]=tasks
    result['primary_holm_p']=holm(pvalues) if pvalues else {};result['master_p1_certified']=False
    result['reason']='Three reused DEV seeds and explicit reused own-reset baseline; no independent confirmation'
    return result


def main():
    receipt=json.loads((ROOT/'artifacts/r8-matched-credit-dev-v1.json').read_text(encoding='utf-8'))
    if receipt['error'] is not None or receipt['completed_jobs']!=96 or receipt['row_count']!=192:raise ValueError('Complete registered baseline required')
    previous=ROOT/receipt['run'];hashes=json.loads((previous/'dataset-hashes.json').read_text(encoding='utf-8'))
    for file,expected in hashes.items():
        if hashlib.sha256((previous/'datasets'/file).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen baseline data changed')
    reservation=ROOT/'local/r8-reset-credit-dev-v1-reservation';reservation.mkdir(exist_ok=False)
    cap=apply_memory_limit();main_before=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()
    run=new_run(ROOT,'r8-reset-credit-dev-v1',{'display_title':'R8 · source-informed stopped-reset local eligibility DEV',
        'protocol':'experiments/r8-reset-credit-dev-v1.md','peer_sha':PEER_SHA,'seeds':SEEDS,'arms':ARMS,
        'base_run':previous.name,'fresh_confirmation':False,'forward_changed':False,'main_or_live_change':False},
        data_description='Reused exact frozen synthetic tensors from registered matched study, no new samples/teacher/private data.',
        data_files=['experiments/r8-reset-credit-dev-v1.md','tests/test_r8_stopped_reset.py','tests/test_r8_matched_credit.py',
                    receipt['run']+'/dataset-hashes.json'])
    (reservation/'run.txt').write_text(str(run),encoding='utf-8');events={'learning':[],'answers':[]};rows=[];error=None
    status={'run':run.name,'status':'P0_ENGINEERING','completed_jobs':0,'total_jobs':48,'policy':'Stopped-reset own-neuron eligibility, same ALIF forward',
        'new_confirmation':False,'main_training':False}
    def publish():atomic_json(run/'progress.json',status);atomic_json(run/'events.json',events)
    publish();atomic_json(ROOT/'local/research-observer-active.json',{'run':str(run)})
    log=(run/'observer-server.log').open('w',encoding='utf-8')
    observer=subprocess.Popen([sys.executable,str(ROOT/'scripts/serve_matched_observer.py'),'--run',str(run),
        '--active-file',str(ROOT/'local/research-observer-active.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    activity(run,'Kiểm giả thuyết eligibility qua reset','Chỉ đổi local voltage derivative, giữ forward ALIF và mọi budget/control.',status='started',evidence='manifest.json')
    print(json.dumps({'run':run.name,'pid':os.getpid(),'observer_pid':observer.pid}),flush=True)
    baseline={}
    for file in (previous/'jobs').glob('*/*-result.json'):
        row=json.loads(file.read_text(encoding='utf-8'));baseline[(row['task'],row['seed'],row['arm'],row['budget'])]=row
    command=[sys.executable,'-X','utf8','-m','pytest','-q','-p','no:cacheprovider','tests/test_r8_stopped_reset.py','tests/test_r8_matched_credit.py',
        'tests/test_r8_snapshot_adapter.py','tests/test_research_gate_repairs.py','--junitxml='+str(run/'p0-tests.xml')]
    start=time.perf_counter();deadline=start+1800;(run/'jobs').mkdir()
    with ResourceMonitor(interval=.05) as monitor:
        try:
            tests=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=45)
            (run/'p0-tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8')
            if tests.returncode:raise RuntimeError('P0 trace/persistence/parity failure; no quality training')
            for task in TASKS:
                for seed in SEEDS:
                    data={}
                    with np.load(previous/'datasets'/f'{task}-{seed}.npz',allow_pickle=False) as source:
                        for split in ('train','refit','id','ood'):
                            x=source[split+'_x'];y=source[split+'_y'];labels,inputs=event_oracle(task,x)
                            if not np.array_equal(labels,y):raise ValueError('Frozen event oracle mismatch')
                            data[split]=(x,y,inputs)
                    for arm in ARMS:
                        job_deadline=min(deadline,time.perf_counter()+120);job=run/'jobs'/f'{task}-{seed}-{arm}';job.mkdir()
                        status.update(status='TRAINING_STOPPED_RESET',task=task,seed=seed,arm=arm)
                        for budget in ('equal_updates','equal_cpu'):
                            model=StoppedResetCredit(Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed),arm)
                            model.calibrate(data['train'][0][:32]);status['budget']=budget
                            def progress(measurement):
                                event={'task':task,'seed':seed,'arm':arm,'budget':budget,**measurement}
                                events['learning'].append(event);events['learning']=events['learning'][-40:]
                                status.update(latest_learning=event,resources=monitor.report());publish();monitor.check()
                            training,history=train(model,*data['train'][:2],budget,job_deadline,progress)
                            np.savez(job/(budget+'-learning.npz'),history=history)
                            metrics,predictions=evaluate(model,data);payload=model.dumps();(job/(budget+'-model.zip')).write_bytes(payload)
                            row={'task':task,'seed':seed,'arm':arm,'budget':budget,'training':training,'metrics':metrics,'predictions':predictions,
                                'checkpoint_sha256':hashlib.sha256(payload).hexdigest()}
                            write_json(job/(budget+'-result.json'),row);rows.append(row)
                            for i in range(8):events['answers'].append({'task':task,'seed':seed,'arm':arm,'budget':budget,'input':data['id'][2][i],
                                'prediction':predictions['id']['direct'][i],'target':predictions['id']['target'][i],'qwen_output':None,
                                'scope':'Actual event-task neural answer, not general conversation'})
                            events['answers']=events['answers'][-32:];status['latest_metrics']=metrics;publish();monitor.check()
                            append(run,'learning',{'task':task,'seed':seed,'condition':arm,'budget':budget,**training,'metrics':metrics,'actual':True})
                            if time.perf_counter()>job_deadline:raise TimeoutError('Registered job wall limit')
                        status['completed_jobs']+=1;publish();print(json.dumps({'completed':status['completed_jobs'],'task':task,'seed':seed,'arm':arm}),flush=True)
            status['status']='COMPLETE_DEV_ONLY'
        except Exception as exc:error=type(exc).__name__+': '+str(exc);status.update(status='STOPPED',error=error)
    result={'summary':summaries(rows,baseline),'row_count':len(rows),'completed_jobs':status['completed_jobs'],'error':error,
        'resources':monitor.report(),'cap':cap,'p0_test_command':command,'frozen_dataset_sha256':hashes,'base_run':previous.name,
        'main_sha_before':main_before,'main_sha_after':subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip(),
        'forward_changed':False,'new_confirmation':False,'decision':'DEV_COMPLETE_NOT_MASTER_P1_GO' if error is None else 'STOPPED_PRESERVE_PARTIAL'}
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/r8-reset-credit-dev-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    status.update(decision=result['decision'],resources=result['resources']);publish()
    (run/'report.md').write_text('# Registered reset-credit DEV\n\nNo architecture change or fresh confirmation.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    activity(run,'Hoàn tất kiểm reset-credit',result['decision'],status='completed' if error is None else 'failed',evidence='result.json')
    print(json.dumps({'run':run.name,'decision':result['decision'],'error':error}),flush=True);return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
