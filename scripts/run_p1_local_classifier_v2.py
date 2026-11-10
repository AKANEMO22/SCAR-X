"""Final registered auxiliary-classifier adjustment; actual traces to localhost."""
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
from scarx.neural.local_classifier_conditioned import ConditionedLocalClassifierCore
from scarx.neural.local_target import LocalTargetCore
from scarx.evaluation.readout_conditioning import ConditionedReadout
from scarx.evaluation.tasks21 import representation,ridge_probe
from scarx.evaluation.binding_audit21 import strict_binding
from scarx.evaluation.phase21 import interval,signflip,holm
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append

TASKS=('order','xor','strict_binding');DEV=(12101,12109,12113);CONFIRM=(12503,12511,12517,12527,12539,12541,12547,12553)


def dataset(task,seed,count,ood=False):
    if task=='strict_binding':return (None,None) if ood else strict_binding(seed,count)
    return representation(task,seed,count,ood)


def feature_view(model):
    view=LocalTargetCore(model.config)
    for name in ('w_in','w_rec','indices','w_out','bias'):getattr(view,name)[:]=getattr(model,name)
    return view


def setup(task,seed):
    config=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed,anchor_strength=0)
    initial=feature_view(SpikingCore(config));head=ConditionedReadout(initial.w_out,initial.bias,'normalized')
    cal,_=dataset(task,seed+4200000,128)
    for f in initial.features_many(cal):head.observe(f)
    head.freeze();return config,initial,head


def snapshot(model,head):return {'w_in':model.w_in.copy(),'w_rec':model.w_rec.copy(),'w_out':head.weights.copy()}


def block_trace(run,task,seed,condition,lr,losses,auxloss,model,head,before,hidden_updates,control=False):
    now={'w_in':model.w_in,'w_rec':model.w_rec,'w_out':head.weights}
    delta={k:float(np.linalg.norm(now[k]-old)) for k,old in before.items()}
    record={'scope':('BPTT đối chứng cô lập' if control else 'P1 bản thử nghiệm; không thay lõi đang dùng'),
        'prompt':f'{task} · seed{seed} · {condition} · LR{lr} · examples{head.updates-511}–{head.updates}',
        'passes':1,'sample_count':512,'losses':[float(np.mean(losses))],'loss_mean':float(np.mean(losses)),
        'auxiliary_loss_mean':float(np.mean(auxloss)) if auxloss else None,'update_steps':512,'update_unit':'episode/head',
        'hidden_update_steps':hidden_updates,'changed_synapses':{k:int(np.count_nonzero(now[k]!=old)) for k,old in before.items()},
        'weight_delta':delta,'attempted_weight_delta':delta,'accepted':True,'rolled_back':False,
        'guard_checked':False,'guard_probe_count':0,'reason':'Research clone; no serving guard or deployment. These measured updates remain in this trial only.'}
    append(run,'learning',record);return record


def measure(model,head,task,seed,phase,condition,lr,run,artifact,initial,metrics):
    fit,fy=dataset(task,seed+4400000,128);test,ty=dataset(task,seed+4500000,256 if phase!='confirmation' else 512)
    ood,oy=dataset(task,seed+4600000,128,True);view=feature_view(model)
    ff=view.features_many(fit);tf=view.features_many(test);probabilities=head.probabilities(tf);predictions=probabilities.argmax(1)
    original=ridge_probe(initial.features_many(fit),fy,initial.features_many(test),ty);probe=ridge_probe(ff,fy,tf,ty)
    identifier=f'{phase}-{task}-{seed}-{condition}-{lr}';path=artifact/(identifier+'.core.npz');hp=artifact/(identifier+'.head.npz')
    path.write_bytes(model.dumps());hp.write_bytes(head.dumps())
    restore=type(model).loads(path.read_bytes());restored_head=ConditionedReadout.loads(hp.read_bytes())
    for name,value in model.arrays().items():np.testing.assert_array_equal(value,restore.arrays()[name])
    assert np.array_equal(restored_head.probabilities(tf),probabilities)
    row={'phase':phase,'task':task,'seed':seed,'condition':condition,'lr':lr,'direct':float(np.mean(predictions==ty)),
        'probe':probe,'initial_probe':original,'probe_gain':probe-original,'initial_headroom':1-original,
        'probe_time_permuted':ridge_probe(ff,fy,view.features_many(test,'permute'),ty),
        'probe_state_reset':ridge_probe(ff,fy,view.features_many(test,'reset'),ty),
        'ood_probe':ridge_probe(ff,fy,view.features_many(ood),oy) if ood is not None else None,
        'bag_only':ridge_probe(fit.sum(1),fy,test.sum(1),ty),'checkpoint':path.relative_to(root).as_posix(),
        'head_checkpoint':hp.relative_to(root).as_posix(),'checkpoint_exact':True,'array_bytes':model.memory_bytes(),**metrics}
    for i in range(3):
        append(run,'answer',{'scope':'BPTT đối chứng cô lập' if phase=='control' else 'P1 bản thử nghiệm; không thay lõi đang dùng',
            'case_id':identifier+f'/{i}','prompt':json.dumps(test[i].tolist()),
            'input_scope':'Actual float32 tensor [12,8]; order/XOR/binding classification, not text conversation. Batched inference, individual latency not measured.',
            'generated':str(int(predictions[i])),'target':str(int(ty[i])),'probabilities':probabilities[i].tolist(),
            'qwen':None,'active':48,'allocated':48,'neural_ticks':12,'checkpoint':row['checkpoint']})
    trace={'computed':48,'allocated':48,'tick':2048*12,'spikes':model.spikes.astype(int).tolist(),
        'voltage':np.round(model.voltage,4).tolist(),'task':task,'seed':seed,'sample_kind':'last actual training tick'}
    write_json(run/(identifier+'.json'),{'summary':row,'targets':ty.tolist(),'predictions':predictions.tolist(),'neural_sample':trace})
    return row,trace


def trial(job):
    phase,task,seed,condition,lr,runpath,artpath=job;run=Path(runpath);artifact=Path(artpath)
    cap=apply_memory_limit()
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.perf_counter();config,initial,head=setup(task,seed)
    normalized=condition.startswith(('norm','shuffle'));local=condition not in ('readout','scalar')
    model=(ConditionedLocalClassifierCore(config) if normalized else LocalClassifierCore(config)) if local else SpikingCore(config)
    # Config field affects only hidden updates; all heads retain fixed .08 delta rule.
    from dataclasses import replace
    model.config=replace(model.config,hidden_lr=lr)
    if normalized:model.condition(head.mean,np.maximum(np.sqrt(head.m2/(head.count-1)),.05))
    x,y=dataset(task,seed+4300000,2048);start_weights=snapshot(model,head);before=snapshot(model,head)
    losses=[];auxloss=[];allloss=[];allaux=[];blocks=[]
    with ResourceMonitor(interval=.25) as monitor:
        for index,(sequence,label) in enumerate(zip(x,y)):
            model.reset_state()
            for tick,value in enumerate(sequence):
                model.step(value,track=True)
                updates=4 if condition.endswith('final4') and tick==11 else 1 if condition.endswith('last4') and tick>=8 else 0
                if local:
                    for _ in range(updates):auxloss.append(model.learn_auxiliary(int(label),condition.startswith('shuffle')))
            p=head.probabilities(model.features);losses.append(head.learn(model.features,int(label)))
            if condition=='scalar':
                reward=float(p[int(label)]);advantage=float(np.clip(reward-model.reward_mean,-1,1));model.reward_mean=.95*model.reward_mean+.05*reward
                for name,elig in (('in',model.elig_in),('rec',model.elig_rec)):
                    w=getattr(model,'w_'+name);w[model.rows]+=np.clip(lr*advantage*elig,-config.update_clip,config.update_clip);np.clip(w,-config.weight_clip,config.weight_clip,out=w)
            model.updates+=1
            if (index+1)%512==0:
                blocks.append(block_trace(run,task,seed,condition,lr,losses,auxloss,model,head,before,2048 if local else 512 if condition=='scalar' else 0))
                allloss+=losses;allaux+=auxloss;losses=[];auxloss=[];before=snapshot(model,head);monitor.check()
                if time.perf_counter()-began>180:raise TimeoutError('Registered180s trial limit')
        row,trace=measure(model,head,task,seed,phase,condition,lr,run,artifact,initial,{
            'main_updates':head.updates,'auxiliary_hidden_updates':getattr(model,'aux_updates',0),'scalar_hidden_updates':2048 if condition=='scalar' else 0,
            'hidden_delta_in':float(np.linalg.norm(model.w_in-start_weights['w_in'])),'hidden_delta_rec':float(np.linalg.norm(model.w_rec-start_weights['w_rec'])),
            'mean_loss':float(np.mean(allloss)),'mean_aux_loss':float(np.mean(allaux)) if allaux else None,
            'train_seconds':time.perf_counter()-began,'resources':monitor.report(),'cap':cap,'learning_blocks':blocks})
        return row,trace


def jobs_run(jobs,run,monitor,deadline,label):
    todo=iter(jobs);pending={};records=[]
    progress(run,label+': bắt đầu',0,len(jobs))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for _ in range(2):
            try:job=next(todo);pending[pool.submit(trial,job)]=job
            except StopIteration:break
        while pending:
            ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED);monitor.check()
            if time.perf_counter()>deadline:raise TimeoutError('Registered2700s round limit')
            for future in ready:
                pending.pop(future);row,trace=future.result();records.append(row)
                progress(run,f'{label}: {row["task"]} · {row["condition"]} · seed{row["seed"]}',len(records),len(jobs),
                    metrics={k:row[k] for k in ('direct','probe_gain','mean_loss','mean_aux_loss','train_seconds')},neural=trace,task=row['task'])
                print(json.dumps({k:row[k] for k in ('phase','task','seed','condition','lr','direct','probe_gain')}),flush=True)
                try:job=next(todo);pending[pool.submit(trial,job)]=job
                except StopIteration:pass
    return records


def controls_run(run,artifact,monitor,deadline):
    # Imports autograd ONLY into the separate evaluation-control phase.
    from scarx.evaluation.controls import ClassifierLearnabilityControl,torch
    torch.set_num_threads(1);rows=[]
    for task,seed,frozen in itertools.product(TASKS,DEV,(True,False)):
        began=time.perf_counter();config,initial,head=setup(task,seed);control=ClassifierLearnabilityControl(config,head,frozen)
        x,y=dataset(task,seed+4300000,2048);losses=[];allloss=[];before=snapshot(control.copied_core(),head)
        np.testing.assert_array_equal(control.features(x[:3],False).detach().numpy(),initial.features_many(x[:3]))
        condition='bptt-readout' if frozen else 'bptt-full'
        for index,(seq,target) in enumerate(zip(x,y)):
            losses.append(control.learn(seq,int(target)))
            if (index+1)%512==0:
                model=control.copied_core();block_trace(run,task,seed,condition,.002,losses,[],model,head,before,0 if frozen else 512,True)
                allloss+=losses;losses=[];before=snapshot(model,head);monitor.check()
                progress(run,f'Đối chứng cô lập: {task} · seed{seed} · {condition}',len(rows),18,
                    metrics={'examples':index+1,'loss':float(np.mean(allloss[-512:]))},task=task)
                if time.perf_counter()-began>180 or time.perf_counter()>deadline:raise TimeoutError('Control budget')
        model=control.copied_core();fit,_=dataset(task,seed+4400000,128)
        expected=initial.features_many(fit) if frozen else feature_view(model).features_many(fit)
        np.testing.assert_array_equal(control.features(fit,False).detach().numpy(),expected)
        row,trace=measure(model,head,task,seed,'control',condition,.002,run,artifact,initial,{
            'main_updates':head.updates,'hidden_bptt_updates':control.hidden_updates,'optimizer_bytes':control.optimizer_bytes(),
            'mean_loss':float(np.mean(allloss)),'train_seconds':time.perf_counter()-began,
            'resources':monitor.report(),'control_only':True,'forward_parity':True,'optimizer_resume_claim':False})
        rows.append(row);progress(run,f'Đối chứng xong: {task} · {condition}',len(rows),18,metrics={k:row[k] for k in ('direct','probe_gain','train_seconds')},neural=trace,task=task)
        print(json.dumps({k:row[k] for k in ('phase','task','seed','condition','direct','probe_gain')}),flush=True)
    return rows


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();deadline=began+2700
    run=new_run(root,'p1-local-classifier-v2',{'protocol':'experiments/p1-local-classifier-v2.md',
        'display_title':'P1 · conditioning + thời điểm dạy hidden · vòng cuối',
        'development_seeds':DEV,'confirmation_seeds':CONFIRM,'hidden':48,'workers':2,'examples':2048,
        'local_backward':False,'isolated_bptt_control':True,'final_adjustment_of_family':True})
    artifact=root/'artifacts/local-classifier'/run.name;artifact.mkdir(parents=True)
    conditions=[('readout',.002),('scalar',.002),('raw-last4',.002),('raw-final4',.002)]
    conditions += [(c,lr) for c,lr in itertools.product(('norm-last4','norm-final4','shuffle-last4','shuffle-final4'),(.0002,.002))]
    with ResourceMonitor(interval=.1) as monitor:
        jobs=[('development',t,s,c,lr,str(run),str(artifact)) for t,s,(c,lr) in itertools.product(TASKS,DEV,conditions)]
        dev=jobs_run(jobs,run,monitor,deadline,'Phát triển');write_json(run/'development.json',dev)
        options=list(itertools.product(('norm-last4','norm-final4'),(.0002,.002)))
        selected=max(options,key=lambda a:(np.mean([r['probe_gain'] for r in dev if (r['condition'],r['lr'])==a]),-a[1],a[0]=='norm-last4'))
        summary={t:{'probe_gain':float(np.mean([r['probe_gain'] for r in dev if r['task']==t and (r['condition'],r['lr'])==selected])),
            'direct':float(np.mean([r['direct'] for r in dev if r['task']==t and (r['condition'],r['lr'])==selected])),
            'initial_probe':float(np.mean([r['initial_probe'] for r in dev if r['task']==t and (r['condition'],r['lr'])==selected]))} for t in TASKS}
        screen=sum(v['probe_gain']>=.05 and v['direct']>=.65 for v in summary.values())>=2
        selection={'selected_condition':selected[0],'selected_lr':selected[1],'tasks':summary,'screen_pass':screen,'confirmation_opened':screen}
        write_json(run/'selection-frozen.json',selection);confirmed=[];stats={}
        if screen:
            conditions=[selected,('raw-'+selected[0].split('-')[1],.002),('shuffle-'+selected[0].split('-')[1],selected[1]),('readout',.002),('scalar',.002)]
            jobs=[('confirmation',t,s,c,lr,str(run),str(artifact)) for t,s,(c,lr) in itertools.product(TASKS,CONFIRM,conditions)]
            confirmed=jobs_run(jobs,run,monitor,deadline,'Xác nhận');write_json(run/'confirmation.json',confirmed)
            for t in TASKS:
                rs=[r for r in confirmed if r['task']==t and (r['condition'],r['lr'])==selected]
                diff=[r['probe']-next(b['probe'] for b in confirmed if b['task']==t and b['seed']==r['seed'] and b['condition']=='readout') for r in rs]
                gains=[r['probe_gain'] for r in rs]
                stats[t]={'probe_gain':interval(gains),'direct':interval([r['direct'] for r in rs]),'vs_readout':interval(diff),'gain_p':signflip(gains),'vs_readout_p':signflip(diff)}
            gp=holm([stats[t]['gain_p'] for t in TASKS]);rp=holm([stats[t]['vs_readout_p'] for t in TASKS])
            for i,t in enumerate(TASKS):
                v=stats[t];v['holm_gain_p']=gp[i];v['holm_vs_readout_p']=rp[i]
                v['pass']=v['probe_gain']['lower95']>=.05 and v['direct']['mean']>=.65 and v['vs_readout']['mean']>0 and gp[i]<=.05 and rp[i]<=.05
        controls=controls_run(run,artifact,monitor,deadline);write_json(run/'controls.json',controls)
        gate=sum(v.get('pass',False) for v in stats.values())>=2
        result={'run':run.relative_to(root).as_posix(),'selection':selection,'confirmation_statistics':stats,'primary_representation_gate':gate,
            'screen_trials':len(dev),'confirmation_trials':len(confirmed),'control_trials':len(controls),
            'control_summary':{t:{c:float(np.mean([r['direct'] for r in controls if r['task']==t and r['condition']==c])) for c in ('bptt-readout','bptt-full')} for t in TASKS},
            'resources':monitor.report(),'cap':cap,'deployed':False,'normal_chat_or_vision_proven':False,'retention_proven':False,
            'decision':'GO_P2_RESEARCH_ONLY' if gate else 'NO_GO_LOCAL_CLASSIFIER_FAMILY',
            'family_rounds_exhausted':not gate,'scope':'Same ALIF48, no deployment; actual controls use backward only in evaluation/controls.py; no equal-time efficiency claim'}
        (run/'report.md').write_text('# Final local-classifier adjustment\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/p1-local-classifier-v2-latest.json',result)
        print('RESULT '+str(run),flush=True)


if __name__=='__main__':main()
