"""Registered matched R8 DEV study; bounded, auditable, no main/live mutation."""
from pathlib import Path
import copy,hashlib,itertools,json,os,subprocess,sys,time,urllib.request
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit,ARMS,PEER_SHA,PEER
from scarx.evaluation.r8_matched_study import TASKS,SEEDS,dataset,direct_predict,refit_predict,paired_statistics,holm
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.forensic.r8_snapshot import dumps,loads
from scarx.neural.core import Config
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import activity,append


def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)


def model_for(seed,arm,train_x):
    model=MatchedLocalCredit(Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed),arm)
    model.calibrate(train_x[:32]);return model


def train(model,x,y,budget,deadline,on_progress):
    limit=1024 if budget=='equal_updates' else 4096;start=time.perf_counter();cpu=0.;records=[]
    for step in range(limit):
        if time.perf_counter()>deadline:raise TimeoutError('Registered wall budget exceeded')
        before=[model.core.w_in.copy(),model.core.w_rec.copy(),model.core.w_out.copy(),model.core.bias.copy()]
        began=time.process_time();loss=model.learn_episode(x[step%len(x)],int(y[step%len(y)]));cpu+=time.process_time()-began
        hidden=float(np.linalg.norm(np.r_[(model.core.w_in-before[0]).ravel(),(model.core.w_rec-before[1]).ravel()].astype(float)))
        readout=float(np.linalg.norm(np.r_[(model.core.w_out-before[2]).ravel(),(model.core.bias-before[3]).ravel()].astype(float)))
        if not np.isfinite(loss) or not np.isfinite(hidden+readout):raise ValueError('Nonfinite loss/update')
        records.append([step+1,loss,hidden,readout,cpu])
        if step==0 or (step+1)%64==0:on_progress({'step':step+1,'loss':loss,'hidden_update_norm':hidden,'readout_update_norm':readout,'train_cpu_seconds':cpu})
        if budget=='equal_cpu' and cpu>=.5:break
    on_progress({'step':len(records),'loss':loss,'hidden_update_norm':hidden,'readout_update_norm':readout,'train_cpu_seconds':cpu})
    return {'episodes':len(records),'hidden_updates':model.hidden_updates,'readout_updates':model.episodes_learned,
        'train_cpu_seconds':cpu,'train_wall_seconds':time.perf_counter()-start,'equal_cpu_ceiling_seconds':.5 if budget=='equal_cpu' else None,
        'cpu_ceiling_overshoot_seconds':max(0,cpu-.5) if budget=='equal_cpu' else 0},np.asarray(records,np.float64)


def evaluate(model,data):
    fitx,fity,_=data['refit'];features=model.features_many(fitx);metrics={};predictions={}
    for split in ('id','ood'):
        x,y,_=data[split];f=model.features_many(x);direct=direct_predict(model,f);probe=refit_predict(features,fity,f)
        metrics[split+'_direct']=float(np.mean(direct==y));metrics[split+'_probe']=float(np.mean(probe==y))
        predictions[split]={'direct':direct.tolist(),'probe':probe.tolist(),'target':y.tolist()}
    blob=model.dumps()
    for name,rin,rrec in [('revert_in',True,False),('revert_rec',False,True),('revert_both',True,True)]:
        clone=type(model).loads(blob);clone.revert_hidden_weights(rin,rrec)
        pred=refit_predict(clone.features_many(fitx),fity,clone.features_many(data['id'][0]))
        metrics[name+'_probe']=float(np.mean(pred==data['id'][1]))
    return metrics,predictions


def summarize(rows):
    output={};primary_p={}
    pairs=[('multiscale','multi_ungated','fast_ungated'),('gate','multi_gated','multi_ungated'),
        ('hidden','multi_gated','multi_gated_frozen'),('zero','multi_gated','multi_gated_zero'),('routing','multi_gated','multi_gated_shuffle')]
    for budget in ('equal_updates','equal_cpu'):
        by_task={}
        for task in TASKS:
            indexed={(r['seed'],r['arm']):r for r in rows if r['task']==task and r['budget']==budget}
            if len(indexed)!=len(SEEDS)*len(ARMS):continue
            means={arm:{key:float(np.mean([indexed[(seed,arm)]['metrics'][key] for seed in SEEDS])) for key in ('id_direct','id_probe','ood_direct','ood_probe')} for arm in ARMS}
            contrasts={}
            for name,left,right in pairs:
                s=paired_statistics([indexed[(seed,left)]['metrics']['id_probe']-indexed[(seed,right)]['metrics']['id_probe'] for seed in SEEDS]);contrasts[name]=s
                if budget=='equal_updates' and task!='order':primary_p[task+'/'+name]=s['exact_two_sided_p']
            by_task[task]={'arm_means':means,'contrasts':contrasts}
        output[budget]=by_task
    output['primary_holm_p']=holm(primary_p) if primary_p else {}
    output['master_p1_certified']=False;output['reason']='3 reused DEV seeds only; no fresh confirmation, no ceiling exemptions'
    return output


def main():
    reservation=ROOT/'local/r8-matched-dev-v1-reservation';reservation.parent.mkdir(parents=True,exist_ok=True)
    reservation.mkdir(exist_ok=False) # do not spend a second study budget on an accidental rerun
    cap=apply_memory_limit();main_before=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()
    run=new_run(ROOT,'r8-matched-credit-dev-v1',{'display_title':'R8 matched controls · actual neural DEV learning',
        'protocol':'experiments/r8-matched-credit-dev-v1.md','peer_sha':PEER_SHA,'seeds':SEEDS,'arms':list(ARMS),
        'tasks':TASKS,'fresh_confirmation':False,'main_or_live_change':False},
        data_description='Pinned synthetic event tasks, source-independent event-label oracle; reused DEV only, no language/teacher/private data.',
        data_files=['experiments/r8-matched-credit-dev-v1.md','tests/test_r8_matched_credit.py','tests/test_r8_matched_study.py'])
    (reservation/'run.txt').write_text(str(run),encoding='utf-8');events={'learning':[],'answers':[]};rows=[];error=None
    status={'run':run.name,'status':'P0_ENGINEERING','completed_jobs':0,'total_jobs':96,'model':'one H48 ALIF research core per isolated arm; no serving model',
        'main_training':False,'new_confirmation':False}
    def publish():atomic_json(run/'progress.json',status);atomic_json(run/'events.json',events)
    publish();print(json.dumps({'run':run.name,'observer':'http://127.0.0.1:8767','pid':os.getpid()}),flush=True)
    log=(run/'observer-server.log').open('w',encoding='utf-8')
    observer=subprocess.Popen([sys.executable,str(ROOT/'scripts/serve_matched_observer.py'),'--run',str(run)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    activity(run,'Bắt đầu DEV causal có đối chứng','Giữ ALIF/learning equation; sửa matching, ghi actual losses/updates/predictions.',status='started',evidence='manifest.json')
    started=time.perf_counter();deadline=started+1800;test_command=[sys.executable,'-X','utf8','-m','pytest','-q','-p','no:cacheprovider',
        'tests/test_r8_matched_credit.py','tests/test_r8_matched_study.py','tests/test_r8_snapshot_adapter.py','tests/test_research_gate_repairs.py',
        '--junitxml='+str(run/'p0-tests.xml')]
    with ResourceMonitor(interval=.05) as monitor:
        try:
            tests=subprocess.run(test_command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=45)
            (run/'p0-tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8')
            if tests.returncode:raise RuntimeError('P0 engineering failed; no quality training')
            all_data={};(run/'datasets').mkdir();(run/'jobs').mkdir();dataset_hashes={}
            for task in TASKS:
                for seed in SEEDS:
                    data=dataset(task,seed);all_data[(task,seed)]=data
                    fields={k+'_'+suffix:value for k,(x,y,_) in data.items() for suffix,value in [('x',x),('y',y)]}
                    path=run/'datasets'/f'{task}-{seed}.npz';np.savez(path,**fields);dataset_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
                    digests=[{hashlib.sha256(row.tobytes()).hexdigest() for row in data[s][0]} for s in data]
                    if any(a&b for i,a in enumerate(digests) for b in digests[i+1:]):raise ValueError('Raw episode split overlap')
            write_json(run/'dataset-hashes.json',dataset_hashes)
            for task in TASKS:
                for seed in SEEDS:
                    data=all_data[(task,seed)]
                    for arm in ARMS:
                        job_start=time.perf_counter();job_deadline=min(deadline,job_start+120);job_id=f'{task}-{seed}-{arm}'
                        job=run/'jobs'/job_id;job.mkdir();status.update(status='TRAINING',task=task,seed=seed,arm=arm)
                        for budget in ('equal_updates','equal_cpu'):
                            model=model_for(seed,arm,data['train'][0]);status['budget']=budget
                            def progress(measurement):
                                event={'task':task,'seed':seed,'arm':arm,'budget':budget,**measurement}
                                events['learning'].append(event);events['learning']=events['learning'][-40:]
                                status.update(latest_learning=event,resources=monitor.report());publish();monitor.check()
                            training,history=train(model,*data['train'][:2],budget,job_deadline,progress)
                            np.savez(job/(budget+'-learning.npz'),history=history,columns=np.array(['step','loss','hidden_norm','readout_norm','train_cpu_seconds']))
                            metrics,predictions=evaluate(model,data);checkpoint=job/(budget+'-model.zip');checkpoint.write_bytes(model.dumps())
                            result={'task':task,'seed':seed,'arm':arm,'budget':budget,'training':training,'metrics':metrics,'predictions':predictions,
                                'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
                            write_json(job/(budget+'-result.json'),result);rows.append(result)
                            for i in range(8):
                                answer={'task':task,'seed':seed,'arm':arm,'budget':budget,'input':data['id'][2][i],
                                    'prediction':predictions['id']['direct'][i],'target':predictions['id']['target'][i],'qwen_output':None,
                                    'scope':'Actual binary event-task neural output, not natural-language QA'}
                                events['answers'].append(answer)
                            events['answers']=events['answers'][-32:];status['latest_metrics']=metrics;publish();monitor.check()
                            append(run,'learning',{'task':task,'seed':seed,'condition':arm,'budget':budget,**training,'metrics':metrics,'actual':True})
                            if time.perf_counter()>job_deadline:raise TimeoutError('Registered job wall limit exceeded')
                        status['completed_jobs']+=1;publish();print(json.dumps({'completed':status['completed_jobs'],'task':task,'seed':seed,'arm':arm}),flush=True)
            status['status']='COMPLETE_DEV_ONLY'
        except Exception as exc:error=type(exc).__name__+': '+str(exc);status.update(status='STOPPED',error=error)
    summary=summarize(rows);result={'summary':summary,'row_count':len(rows),'completed_jobs':status['completed_jobs'],'error':error,
        'resources':monitor.report(),'cap':cap,'p0_test_command':test_command,'new_confirmation':False,'main_or_live_change':False,
        'main_sha_before':main_before,'main_sha_after':subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip(),
        'decision':'DEV_COMPLETE_NOT_MASTER_P1_GO' if error is None else 'STOPPED_PRESERVE_PARTIAL'}
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/r8-matched-credit-dev-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    status.update(decision=result['decision'],resources=result['resources']);publish()
    (run/'report.md').write_text('# Registered R8 matched DEV\n\nNo fresh confirmation or production promotion.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    activity(run,'Hoàn tất DEV causal',result['decision'],status='completed' if error is None else 'failed',evidence='result.json')
    print(json.dumps({'run':run.name,'decision':result['decision'],'error':error,'observer_pid':observer.pid}),flush=True)
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
