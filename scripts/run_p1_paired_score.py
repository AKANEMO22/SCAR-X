"""Final registered Phase2.1-score adjustment, local updates only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,itertools
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from scarx.neural.core import Config
from scarx.neural.paired_score import PairedScoreCore
from scarx.evaluation.tasks21 import ridge_probe
from scarx.evaluation.phase21 import interval,signflip,holm
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from run_p1_local_classifier_v2 import dataset

TASKS=('order','xor','strict_binding');DEV=(14207,14221,14243);CONFIRM=(14503,14519,14533,14537,14543,14549,14551,14557)


def make(task,seed,condition,lr):
    c=Config(inputs=8,hidden=48,outputs=2,fan_in=8,plastic_rows=12,seed=seed,anchor_strength=0,hidden_lr=lr)
    m=PairedScoreCore(c,'antithetic' if condition.endswith('antithetic') else 'independent',
        'sgd' if '-sgd-' in condition else 'adam',condition.startswith('clipped'),condition.startswith('frozen'),condition.startswith('shuffle'))
    cal,_=dataset(task,seed+6200000,128);m.calibrate(m.features_many(cal,seed+6700000));return m


def weights(m):return {name:getattr(m,name).copy() for name in ('w_in','w_rec','w_out')}


def trial(job):
    phase,task,seed,condition,lr,runpath,artpath=job;run=Path(runpath);artifact=Path(artpath)
    cap=apply_memory_limit()
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.perf_counter();m=make(task,seed,condition,lr);initial=PairedScoreCore.loads(m.dumps())
    x,y=dataset(task,seed+6300000,2048);before=weights(m);original=weights(m);losses=[];pathlosses=[];blocks=[]
    with ResourceMonitor(interval=.2) as monitor:
        for index,(sequence,target) in enumerate(zip(x,y)):
            measured=m.learn_example(sequence,int(target));losses.append(measured['loss']);pathlosses.append(measured['trajectory_loss'])
            if (index+1)%512==0:
                if any(not np.isfinite(a).all() for a in m.arrays().values()):raise ValueError('Nonfinite paired-score state')
                delta={name:float(np.linalg.norm(getattr(m,name)-old)) for name,old in before.items()}
                event={'scope':'P1 local score-v2 · bản thử nghiệm; không thay lõi đang dùng',
                    'prompt':f'{task} · seed{seed} · {condition} · LR{lr} · examples{index-510}–{index+1}',
                    'passes':1,'sample_count':512,'loss_mean':float(np.mean(losses[-512:])),
                    'losses':[float(np.mean(losses[-512:]))],'trajectory_loss_mean':float(np.mean(pathlosses[-512:])),
                    'update_steps':512,'update_unit':'episode/head','hidden_update_steps':0 if m.frozen else 2048,
                    'rollouts':4096,'neural_ticks':49152,'weight_delta':delta,'attempted_weight_delta':delta,
                    'changed_synapses':{name:int(np.count_nonzero(getattr(m,name)!=old)) for name,old in before.items()},
                    'accepted':True,'rolled_back':False,'guard_checked':False,'guard_probe_count':0,
                    'reason':'Research clone, no serving guard. Scalar return × local pre/post score; no network backward.'}
                append(run,'learning',event);blocks.append(event);before=weights(m);monitor.check()
                if time.perf_counter()-began>180:raise TimeoutError('Registered180s paired-score trial limit')
        train_seconds=time.perf_counter()-began
        fit,fy=dataset(task,seed+6400000,128);test,ty=dataset(task,seed+6500000,256 if phase=='development' else 512);ood,oy=dataset(task,seed+6600000,128,True)
        ff=m.features_many(fit,seed+6800000);tf=m.features_many(test,seed+6900000)
        original_probe=ridge_probe(initial.features_many(fit,seed+6800000),fy,initial.features_many(test,seed+6900000),ty)
        probe=ridge_probe(ff,fy,tf,ty);p=m.probabilities(tf);predictions=p.argmax(1)
        identifier=f'{phase}-{task}-{seed}-{condition}-{lr}';path=artifact/(identifier+'.npz');path.write_bytes(m.dumps())
        restored=PairedScoreCore.loads(path.read_bytes())
        for name,a in m.arrays().items():np.testing.assert_array_equal(a,restored.arrays()[name])
        np.testing.assert_array_equal(p,restored.probabilities(restored.features_many(test,seed+6900000)))
        # Continuation only on two discarded clones; benchmark weights remain2048updates.
        second=PairedScoreCore.loads(m.dumps());a=restored.learn_example(x[-1],int(y[-1]));b=second.learn_example(x[-1],int(y[-1]));assert a==b and restored.dumps()==second.dumps()
        row={'phase':phase,'task':task,'seed':seed,'condition':condition,'lr':lr,'direct':float(np.mean(predictions==ty)),
            'probe':probe,'initial_probe':original_probe,'probe_gain':probe-original_probe,
            'probe_time_permuted':ridge_probe(ff,fy,m.features_many(test,seed+6900000,'permute'),ty),
            'probe_state_reset':ridge_probe(ff,fy,m.features_many(test,seed+6900000,'reset'),ty),
            'ood_probe':ridge_probe(ff,fy,m.features_many(ood,seed+7000000),oy) if ood is not None else None,
            'bag_only':ridge_probe(fit.sum(1),fy,test.sum(1),ty),'main_updates':m.updates,'hidden_updates':m.hidden_updates,
            'training_rollouts':2048*8,'training_ticks':2048*96,'array_bytes':m.memory_bytes(),
            'hidden_delta_in':float(np.linalg.norm(m.w_in-original['w_in'])),'hidden_delta_rec':float(np.linalg.norm(m.w_rec-original['w_rec'])),
            'mean_loss':float(np.mean(losses)),'mean_path_loss':float(np.mean(pathlosses)),
            'checkpoint':path.relative_to(root).as_posix(),'checkpoint_exact':True,'continuation_exact':True,
            'train_seconds':train_seconds,'seconds':time.perf_counter()-began,'resources':monitor.report(),'cap':cap,'learning_blocks':blocks}
        for i in range(3):append(run,'answer',{'scope':'P1 local score-v2 · không phải hội thoại model live',
            'case_id':identifier+f'/{i}','prompt':json.dumps(test[i].tolist()),
            'input_scope':'Actual float32 [12,8] diagnostic tensor. Prediction uses mean of8stochastic feature rollouts; individual latency unmeasured.',
            'generated':str(int(predictions[i])),'target':str(int(ty[i])),'probabilities':p[i].tolist(),'qwen':None,
            'active':48,'allocated':48,'neural_ticks':96,'checkpoint':row['checkpoint']})
        neural={'computed':48,'allocated':48,'tick':m.ticks,'spikes':m.spikes.astype(int).tolist(),'voltage':np.round(m.voltage,4).tolist(),
            'sample_kind':'last actual training trajectory; score-local, no backward','task':task}
        write_json(run/(identifier+'.json'),{'summary':row,'predictions':predictions.tolist(),'targets':ty.tolist(),'neural_sample':neural})
        return row,neural


def jobs_run(jobs,run,monitor,deadline,label):
    todo=iter(jobs);pending={};out=[];progress(run,label+': bắt đầu',0,len(jobs))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for _ in range(2):
            try:job=next(todo);pending[pool.submit(trial,job)]=job
            except StopIteration:break
        while pending:
            ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED);monitor.check()
            if time.perf_counter()>deadline:raise TimeoutError('Registered3600s round limit')
            for future in ready:
                pending.pop(future);row,neural=future.result();out.append(row)
                progress(run,f'{label}: {row["task"]} · {row["condition"]} · seed{row["seed"]}',len(out),len(jobs),
                    metrics={k:row[k] for k in ('direct','probe_gain','mean_loss','train_seconds')},neural=neural,task=row['task'])
                print(json.dumps({k:row[k] for k in ('phase','task','seed','condition','lr','direct','probe_gain','train_seconds')}),flush=True)
                try:job=next(todo);pending[pool.submit(trial,job)]=job
                except StopIteration:pass
    return out


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    run=new_run(root,'p1-paired-score-v2',{'protocol':'experiments/p1-paired-score-v2.md',
        'display_title':'P1 · paired score eligibility · local learning · vòng cuối',
        'model_scope':'Local score research clone; không đưa backward hoặc weights vào lõi đang dùng',
        'development_seeds':DEV,'confirmation_seeds':CONFIRM,'hidden':48,'plastic_rows':12,'rollouts_per_example':8,
        'new_network_backward':False,'score_family_final_adjustment':True,'workers':2})
    artifact=root/'artifacts/paired-score'/run.name;artifact.mkdir(parents=True)
    conditions=[('frozen-independent',.002),('frozen-antithetic',.002),('clipped-sgd-independent',.002),('clipped-adam-independent',.002)]
    conditions += [(name,lr) for name,lr in itertools.product(('raw-independent','raw-antithetic','shuffle-independent','shuffle-antithetic'),(.0002,.002))]
    with ResourceMonitor(interval=.1) as monitor:
        jobs=[('development',t,s,c,lr,str(run),str(artifact)) for t,s,(c,lr) in itertools.product(TASKS,DEV,conditions)]
        dev=jobs_run(jobs,run,monitor,began+3600,'Phát triển');write_json(run/'development.json',dev)
        options=list(itertools.product(('raw-independent','raw-antithetic'),(.0002,.002)))
        selected=max(options,key=lambda a:(np.mean([r['probe_gain'] for r in dev if (r['condition'],r['lr'])==a]),-a[1],a[0]=='raw-independent'))
        summary={t:{'probe_gain':float(np.mean([r['probe_gain'] for r in dev if r['task']==t and (r['condition'],r['lr'])==selected])),
            'direct':float(np.mean([r['direct'] for r in dev if r['task']==t and (r['condition'],r['lr'])==selected])),
            'initial_probe':float(np.mean([r['initial_probe'] for r in dev if r['task']==t and (r['condition'],r['lr'])==selected]))} for t in TASKS}
        screen=sum(v['probe_gain']>=.05 and v['direct']>=.65 for v in summary.values())>=2
        selection={'selected_condition':selected[0],'selected_lr':selected[1],'tasks':summary,'screen_pass':screen,'confirmation_opened':screen}
        write_json(run/'selection-frozen.json',selection);confirmed=[];stats={}
        frozen='frozen-'+selected[0].split('-')[1];shuffled='shuffle-'+selected[0].split('-')[1]
        if screen:
            conditions=[selected,(frozen,.002),('clipped-sgd-independent',.002),('clipped-adam-independent',.002),(shuffled,selected[1])]
            jobs=[('confirmation',t,s,c,lr,str(run),str(artifact)) for t,s,(c,lr) in itertools.product(TASKS,CONFIRM,conditions)]
            confirmed=jobs_run(jobs,run,monitor,began+3600,'Xác nhận');write_json(run/'confirmation.json',confirmed)
            for task in TASKS:
                rs=[r for r in confirmed if r['task']==task and (r['condition'],r['lr'])==selected]
                gains=[r['probe_gain'] for r in rs];diff=[r['probe']-next(b['probe'] for b in confirmed if b['task']==task and b['seed']==r['seed'] and b['condition']==frozen) for r in rs]
                stats[task]={'probe_gain':interval(gains),'direct':interval([r['direct'] for r in rs]),'vs_frozen':interval(diff),'gain_p':signflip(gains),'vs_frozen_p':signflip(diff)}
            gp=holm([stats[t]['gain_p'] for t in TASKS]);fp=holm([stats[t]['vs_frozen_p'] for t in TASKS])
            for i,t in enumerate(TASKS):
                v=stats[t];v['holm_gain_p']=gp[i];v['holm_vs_frozen_p']=fp[i]
                v['pass']=v['probe_gain']['lower95']>=.05 and v['direct']['mean']>=.65 and v['vs_frozen']['mean']>0 and gp[i]<=.05 and fp[i]<=.05
        gate=sum(v.get('pass',False) for v in stats.values())>=2
        result={'run':run.relative_to(root).as_posix(),'decision':'GO_P2_RESEARCH_ONLY' if gate else 'STOP_SCORE_FAMILY_V2',
            'selection':selection,'confirmation_statistics':stats,'primary_representation_gate':gate,
            'screen_trials':len(dev),'confirmation_trials':len(confirmed),'resources':monitor.report(),'cap':cap,
            'deployed':False,'normal_chat_or_vision_proven':False,'retention_proven':False,'network_backward':False,
            'limitations':['Eight rollouts vs historical one; no equal-cost advantage to old scores or deterministic BPTT.',
                'No new matched-stochastic BPTT control; negative result does not prove universal local-learning impossibility.',
                'Raw-estimator math does not establish unbiased floating-point clipped Adam learner or useful representation.']}
        (run/'report.md').write_text('# Final paired-score adjustment\n\nLocal scalar return / synaptic eligibility, no network backward; quality gate unchanged.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/p1-paired-score-v2-latest.json',result)
        print('RESULT '+str(run),flush=True)


if __name__=='__main__':main()
