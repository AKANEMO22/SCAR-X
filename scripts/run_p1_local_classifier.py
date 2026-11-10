"""Registered local-loss adaptation, with live status published to localhost."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,time,json,itertools
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from scarx.neural.core import Config,SpikingCore
from scarx.neural.local_classifier import LocalClassifierCore
from scarx.neural.local_target import LocalTargetCore
from scarx.evaluation.readout_conditioning import ConditionedReadout
from scarx.evaluation.tasks21 import representation,ridge_probe
from scarx.evaluation.binding_audit21 import strict_binding
from scarx.evaluation.phase21 import interval,signflip,holm
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.experiment_status import progress

TASKS=('order','xor','strict_binding')
DEVELOPMENT=(11113,11117,11131)
CONFIRMATION=(11503,11519,11527,11549,11551,11579,11587,11617)


def dataset(task,seed,count,ood=False):
    if task=='strict_binding':return (None,None) if ood else strict_binding(seed,count)
    return representation(task,seed,count,ood)


def feature_view(model):
    view=LocalTargetCore(model.config)
    for n in ('w_in','w_rec','indices','w_out','bias'):getattr(view,n)[:]=getattr(model,n)
    return view


def trial(job):
    phase,task,seed,condition,lr,runpath,artpath=job
    cap=apply_memory_limit()
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    start=time.monotonic();c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed,anchor_strength=0,hidden_lr=lr)
    model=LocalClassifierCore(c) if condition in ('aux','shuffled') else SpikingCore(c)
    initial=feature_view(model);cal,_=dataset(task,seed+3200000,128)
    head=ConditionedReadout(model.w_out,model.bias,'normalized')
    for f in initial.features_many(cal):head.observe(f)
    head.freeze();x,y=dataset(task,seed+3300000,2048);fit,fy=dataset(task,seed+3400000,128);test,ty=dataset(task,seed+3500000,256 if phase=='development' else 512)
    ood,oy=dataset(task,seed+3600000,128,True)
    initial_probe=ridge_probe(initial.features_many(fit),fy,initial.features_many(test),ty)
    wi=model.w_in.copy();wr=model.w_rec.copy();losses=[];auxloss=[]
    with ResourceMonitor(interval=.25) as monitor:
        for index,(sequence,label) in enumerate(zip(x,y)):
            model.reset_state()
            for tick,value in enumerate(sequence):
                model.step(value,track=True)
                if condition in ('aux','shuffled') and tick>=8:auxloss.append(model.learn_auxiliary(int(label),condition=='shuffled'))
            p=head.probabilities(model.features);losses.append(head.learn(model.features,int(label)))
            if condition=='scalar':
                reward=float(p[int(label)]);advantage=float(np.clip(reward-model.reward_mean,-1,1));model.reward_mean=.95*model.reward_mean+.05*reward
                for name,trace in (('in',model.elig_in),('rec',model.elig_rec)):
                    w=getattr(model,'w_'+name);w[model.rows]+=np.clip(c.hidden_lr*advantage*trace,-c.update_clip,c.update_clip);np.clip(w,-c.weight_clip,c.weight_clip,out=w)
            model.updates+=1
            if index%256==0:
                monitor.check()
                if time.monotonic()-start>180:raise TimeoutError('Registered180s trial limit')
        train_seconds=time.monotonic()-start;view=feature_view(model);ff=view.features_many(fit);tf=view.features_many(test)
        predictions=np.argmax(head.probabilities(tf),axis=1);probe=ridge_probe(ff,fy,tf,ty)
        identifier=f'{phase}-{task}-{seed}-{condition}-{lr}';artifact=Path(artpath)
        model_path=artifact/(identifier+'.core.npz');head_path=artifact/(identifier+'.head.npz');model_path.write_bytes(model.dumps());head_path.write_bytes(head.dumps())
        restore=LocalClassifierCore.loads(model_path.read_bytes()) if condition in ('aux','shuffled') else SpikingCore.loads(model_path.read_bytes())
        for name,array in model.arrays().items():np.testing.assert_array_equal(array,restore.arrays()[name])
        restored_head=ConditionedReadout.loads(head_path.read_bytes());assert np.array_equal(restored_head.probabilities(tf),head.probabilities(tf))
        record={'phase':phase,'task':task,'seed':seed,'condition':condition,'lr':lr,'direct':float(np.mean(predictions==ty)),'probe':probe,
          'initial_probe':initial_probe,'probe_gain':probe-initial_probe,'initial_headroom':1-initial_probe,
          'probe_time_permuted':ridge_probe(ff,fy,view.features_many(test,'permute'),ty),'probe_state_reset':ridge_probe(ff,fy,view.features_many(test,'reset'),ty),
          'ood_probe':ridge_probe(ff,fy,view.features_many(ood),oy) if ood is not None else None,'bag_only':ridge_probe(fit.sum(1),fy,test.sum(1),ty),
          'main_updates':head.updates,'auxiliary_hidden_updates':getattr(model,'aux_updates',0),'scalar_hidden_updates':len(x) if condition=='scalar' else 0,
          'hidden_delta_in':float(np.linalg.norm(model.w_in-wi)),'hidden_delta_rec':float(np.linalg.norm(model.w_rec-wr)),'mean_loss':float(np.mean(losses)),
          'mean_aux_loss':float(np.mean(auxloss)) if auxloss else None,'train_seconds':train_seconds,'seconds':time.monotonic()-start,
          'checkpoint':model_path.relative_to(root).as_posix(),'head_checkpoint':head_path.relative_to(root).as_posix(),'checkpoint_exact':True,'resources':monitor.report(),'cap':cap}
        trace={'computed':48,'allocated':48,'tick':len(x)*12,'spikes':model.spikes.astype(int).tolist(),'voltage':np.round(model.voltage,4).tolist(),
          'input_synapses_learned':condition!='readout','auxiliary_updates':getattr(model,'aux_updates',0),'task':task,'seed':seed,'sample_kind':'last_actual_training_tick'}
        write_json(Path(runpath)/(identifier+'.json'),{'summary':record,'targets':ty.tolist(),'predictions':predictions.tolist(),'neural_sample':trace})
        return record,trace


def run_jobs(jobs,run,monitor,deadline,phase):
    todo=iter(jobs);pending={};out=[];total=len(jobs)
    progress(run,f'{phase}: bắt đầu',0,total,metrics={'fixed_neurons':48,'hidden_fan_in':8,'plastic_rows':12})
    with ProcessPoolExecutor(max_workers=2) as pool:
        for _ in range(2):
            try:job=next(todo);pending[pool.submit(trial,job)]=job
            except StopIteration:break
        while pending:
            ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED);monitor.check()
            if time.monotonic()>deadline:raise TimeoutError('Registered1200s round limit')
            for future in ready:
                pending.pop(future);record,trace=future.result();out.append(record)
                progress(run,f'{phase}: {record["task"]} · {record["condition"]} · seed {record["seed"]}',len(out),total,
                  metrics={k:record[k] for k in ('direct','probe_gain','initial_probe','auxiliary_hidden_updates','train_seconds')},neural=trace,task=record['task'])
                print(json.dumps({k:record[k] for k in ('phase','task','condition','seed','lr','direct','probe_gain','train_seconds')}),flush=True)
                try:job=next(todo);pending[pool.submit(trial,job)]=job
                except StopIteration:pass
    return out


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();start=time.monotonic()
    run=new_run(root,'p1-local-classifier-v1',{'protocol':'experiments/p1-local-classifier-v1.md','dev_seeds':DEVELOPMENT,'confirmation_seeds':CONFIRMATION,'hidden':48,'workers':2,'main_updates':2048,'aux_updates_per_episode':4},data_description='Registered existing order/XOR/strict-binding tasks, new seeds and auxiliary local loss; no private data')
    artifact=root/'artifacts/local-classifier'/run.name;artifact.mkdir(parents=True)
    conditions=(('readout',.002),('scalar',.002),('aux',.0002),('aux',.002),('shuffled',.0002),('shuffled',.002))
    with ResourceMonitor(limit=2000000000,interval=.1) as monitor:
        jobs=[('development',t,s,c,lr,str(run),str(artifact)) for t,s,(c,lr) in itertools.product(TASKS,DEVELOPMENT,conditions)]
        dev=run_jobs(jobs,run,monitor,start+1200,'Phát triển');write_json(run/'development.json',dev)
        means={str(lr):float(np.mean([r['probe_gain'] for r in dev if r['condition']=='aux' and r['lr']==lr])) for lr in (.0002,.002)}
        selected=max((.0002,.002),key=lambda lr:(means[str(lr)],-lr))
        summary={t:{'probe_gain':float(np.mean([r['probe_gain'] for r in dev if r['task']==t and r['condition']=='aux' and r['lr']==selected])),
          'direct':float(np.mean([r['direct'] for r in dev if r['task']==t and r['condition']=='aux' and r['lr']==selected])),
          'initial_probe':float(np.mean([r['initial_probe'] for r in dev if r['task']==t and r['condition']=='aux' and r['lr']==selected]))} for t in TASKS}
        screened=sum(v['probe_gain']>=.05 and v['direct']>=.65 for v in summary.values())>=2
        selection={'selected_lr':selected,'scores':means,'tasks':summary,'screen_pass':screened,'confirmation_opened':False};write_json(run/'selection-frozen.json',selection)
        confirmed=[];stats={}
        if screened:
            progress(run,'Đã khóa lựa chọn; chuyển sang 8 seed xác nhận',0,96,metrics=selection)
            jobs=[('confirmation',t,s,c,selected if c in ('aux','shuffled') else .002,str(run),str(artifact)) for t,s,c in itertools.product(TASKS,CONFIRMATION,('readout','scalar','aux','shuffled'))]
            confirmed=run_jobs(jobs,run,monitor,start+1200,'Xác nhận');write_json(run/'confirmation.json',confirmed)
            for task in TASKS:
                rs=[r for r in confirmed if r['task']==task and r['condition']=='aux']
                diff=[r['probe']-next(b['probe'] for b in confirmed if b['task']==task and b['seed']==r['seed'] and b['condition']=='readout') for r in rs]
                stats[task]={'probe_gain':interval([r['probe_gain'] for r in rs]),'direct':interval([r['direct'] for r in rs]),'vs_readout':interval(diff),
                  'gain_p':signflip([r['probe_gain'] for r in rs]),'vs_readout_p':signflip(diff)}
            gp=holm([stats[t]['gain_p'] for t in TASKS]);rp=holm([stats[t]['vs_readout_p'] for t in TASKS])
            for i,t in enumerate(TASKS):
                v=stats[t];v['holm_gain_p']=gp[i];v['holm_vs_readout_p']=rp[i]
                v['pass']=v['probe_gain']['lower95']>=.05 and v['direct']['mean']>=.65 and v['vs_readout']['mean']>0 and gp[i]<=.05 and rp[i]<=.05
        gate=sum(v.get('pass',False) for v in stats.values())>=2
        result={'run':run.relative_to(root).as_posix(),'selection':selection,'confirmation_statistics':stats,'primary_representation_gate':gate,
          'screen_trials':len(dev),'confirmation_trials':len(confirmed),'resources':monitor.report(),'cap':cap,'deployed':False,
          'normal_chat_or_vision_proven':False,'retention_proven':False,'published_paper_reproduction':False,
          'decision':'GO_P2_RESEARCH_ONLY' if gate else 'PIVOT_LOCAL_CLASSIFIER_V1','learnability_limit':'No new matched BPTT control for these tasks; failure does not refute published DECOLLE or every SNN.'}
        write_json(run/'result.json',result);write_json(root/'artifacts/p1-local-classifier-latest.json',result)
        lines=['# P1 local auxiliary classifier adaptation','','Manual local-loss adaptation under unchanged48ALIF topology; not a faithful DECOLLE reproduction, deployment or normalchat/vision claim.','',
          '| Task | Initial probe | Selected gain | Direct |','|---|---:|---:|---:|']
        for t,v in summary.items():lines.append(f"| {t} | {v['initial_probe']:.4f} | {v['probe_gain']*100:.2f}pp | {v['direct']:.4f} |")
        lines+=['',json.dumps(result,indent=2),'','Main output updates matched; auxiliary hidden updates are4pertrainingepisode vs scalar1, with different compute. No equal-time superiority claim.']
        (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('RESULT '+str(run),flush=True)


if __name__=='__main__':main()
