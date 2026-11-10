"""Locked P1 screen/confirmation; no serving code or private model mutation."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,time,json,itertools,argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from scarx.neural.core import Config,SpikingCore
from scarx.neural.local_target import LocalTargetCore
from scarx.evaluation.tasks21 import representation,ridge_probe
from scarx.evaluation.binding_audit21 import strict_binding
from scarx.evaluation.phase21 import interval,signflip,holm
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor

TASKS=('order','xor','strict_binding')
DEVELOPMENT_SEEDS=(1201,1213,1229)
CONFIRMATION_SEEDS=(1709,1723,1741,1759,1777,1789,1801,1811)


def data(task,seed,count,ood=False):
    if task=='strict_binding':
        if ood:return None,None
        return strict_binding(seed,count)
    return representation(task,seed,count,ood)


def trial(job):
    phase,task,seed,condition,lr,run_path,artifact_path=job
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cap=apply_memory_limit();started=time.monotonic()
    c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed,anchor_strength=0,hidden_lr=lr)
    model=SpikingCore(c) if condition=='scalar' else LocalTargetCore(c)
    initial=LocalTargetCore(c)
    x,y=data(task,seed+1100000,2048)
    fit,fit_y=data(task,seed+1200000,128)
    test,test_y=data(task,seed+1300000,256 if phase=='development' else 512)
    ood,ood_y=data(task,seed+1400000,128,True)
    initial_probe=ridge_probe(initial.features_many(fit),fit_y,initial.features_many(test),test_y)
    checkpoint_initial_in=model.w_in.copy();checkpoint_initial_rec=model.w_rec.copy()
    losses=[]
    with ResourceMonitor(interval=.25) as monitor:
        for index,(sequence,label) in enumerate(zip(x,y)):
            model.forward(sequence,track=True)
            losses.append(model.learn(int(label),'full' if condition=='scalar' else condition))
            if index%256==0:
                monitor.check()
                if time.monotonic()-started>180:raise TimeoutError('Registered180s trial budget')
        trained_seconds=time.monotonic()-started
        # Same deterministic forward for every condition; diagnostic view has
        # no target-driven hidden updates and does not mutate the trained core.
        view=LocalTargetCore(c)
        for name in ('w_in','w_rec','indices','w_out','bias'):
            getattr(view,name)[:]=getattr(model,name)
        ff=view.features_many(fit);tf=view.features_many(test)
        logits=tf@model.w_out.T+model.bias;predictions=np.argmax(logits,axis=1)
        probe=ridge_probe(ff,fit_y,tf,test_y)
        evaluation={'direct':float(np.mean(predictions==test_y)),'probe':probe,'initial_probe':initial_probe,'probe_gain':probe-initial_probe,
          'probe_time_permuted':ridge_probe(ff,fit_y,view.features_many(test,'permute'),test_y),
          'probe_state_reset':ridge_probe(ff,fit_y,view.features_many(test,'reset'),test_y),
          'ood_probe':ridge_probe(ff,fit_y,view.features_many(ood),ood_y) if ood is not None else None,
          'bag_only_probe':ridge_probe(fit.sum(1),fit_y,test.sum(1),test_y)}
        identifier=f'{phase}-{task}-{seed}-{condition}-{lr}'
        checkpoint=Path(artifact_path)/(identifier+'.npz');checkpoint.write_bytes(model.dumps())
        restored=SpikingCore.loads(checkpoint.read_bytes()) if condition=='scalar' else LocalTargetCore.loads(checkpoint.read_bytes())
        for name,array in model.arrays().items():np.testing.assert_array_equal(array,restored.arrays()[name])
        result={'phase':phase,'task':task,'seed':seed,'condition':condition,'lr':lr,**evaluation,
          'train_examples':len(x),'updates':model.updates,'train_seconds':trained_seconds,'total_seconds':time.monotonic()-started,
          'train_loss':float(np.mean(losses)),'delta_input':float(np.linalg.norm(model.w_in-checkpoint_initial_in)),
          'delta_recurrent':float(np.linalg.norm(model.w_rec-checkpoint_initial_rec)),
          'checkpoint':checkpoint.relative_to(root).as_posix(),'checkpoint_exact':True,
          'resources':monitor.report(),'cap':cap,'array_bytes':model.memory_bytes()}
        write_json(Path(run_path)/(identifier+'.json'),{'summary':result,'targets':test_y.tolist(),'direct_predictions':predictions.tolist()})
        return result


def run_jobs(jobs,monitor,deadline):
    results=[];pending={};todo=iter(jobs)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for _ in range(2):
            try:job=next(todo);pending[pool.submit(trial,job)]=job
            except StopIteration:break
        while pending:
            ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED)
            monitor.check()
            if time.monotonic()>deadline:raise TimeoutError('Registered1200s whole-round budget')
            for future in ready:
                pending.pop(future);result=future.result();results.append(result)
                print(json.dumps({k:result[k] for k in ('phase','task','seed','condition','lr','direct','probe_gain','train_seconds')}),flush=True)
                try:job=next(todo);pending[pool.submit(trial,job)]=job
                except StopIteration:pass
    return results


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--random-feedback',action='store_true');args=parser.parse_args()
    dev_seeds=(2203,2213,2237) if args.random_feedback else DEVELOPMENT_SEEDS
    confirm_seeds=(2707,2711,2729,2731,2741,2749,2753,2767) if args.random_feedback else CONFIRMATION_SEEDS
    candidate='feedback' if args.random_feedback else 'target';shuffled='feedback_shuffled' if args.random_feedback else 'shuffled'
    protocol='p1-random-feedback-v2' if args.random_feedback else 'p1-local-target-v1'
    cap=apply_memory_limit();began=time.monotonic()
    run=new_run(root,protocol,{'protocol':'experiments/'+protocol+'.md','tasks':TASKS,'dev_seeds':dev_seeds,'confirmation_seeds':confirm_seeds,'workers':2,'hidden':48,'updates':2048},data_description='Same temporal generators as Phase2.1, strict binding without bag shortcut; no private data')
    artifact=root/'artifacts/p1-local-target'/run.name;artifact.mkdir(parents=True)
    candidates=(('readout',.002),('scalar',.002),(candidate,.002),(candidate,.02),(shuffled,.002),(shuffled,.02))
    with ResourceMonitor(limit=2_000_000_000,interval=.1) as monitor:
        jobs=[('development',t,s,c,lr,str(run),str(artifact)) for t,s,(c,lr) in itertools.product(TASKS,dev_seeds,candidates)]
        dev=run_jobs(jobs,monitor,began+1200);write_json(run/'development.json',dev)
        means={str(lr):float(np.mean([r['probe_gain'] for r in dev if r['condition']==candidate and r['lr']==lr])) for lr in (.002,.02)}
        selected=max((.002,.02),key=lambda lr:(means[str(lr)],-lr))
        task_summary={t:{'gain':float(np.mean([r['probe_gain'] for r in dev if r['task']==t and r['condition']==candidate and r['lr']==selected])),
          'direct':float(np.mean([r['direct'] for r in dev if r['task']==t and r['condition']==candidate and r['lr']==selected]))} for t in TASKS}
        screened=sum(v['gain']>=.05 and v['direct']>=.65 for v in task_summary.values())>=2
        selection={'selected_lr':selected,'development_mean_gain':means,'tasks':task_summary,'screen_pass':screened,'confirmation_opened':False}
        write_json(run/'selection-frozen.json',selection)
        confirmed=[];statistics={}
        if screened:
            jobs=[('confirmation',t,s,c,selected if c in (candidate,shuffled) else .002,str(run),str(artifact)) for t,s,c in itertools.product(TASKS,confirm_seeds,('readout','scalar',candidate,shuffled))]
            confirmed=run_jobs(jobs,monitor,began+1200);write_json(run/'confirmation.json',confirmed)
            for task in TASKS:
                candidates=[r for r in confirmed if r['task']==task and r['condition']==candidate]
                paired=[r['probe']-next(b['probe'] for b in confirmed if b['task']==task and b['seed']==r['seed'] and b['condition']=='readout') for r in candidates]
                statistics[task]={'probe_gain':interval([r['probe_gain'] for r in candidates]),'direct':interval([r['direct'] for r in candidates]),
                  'vs_readout':interval(paired),'gain_signflip_p':signflip([r['probe_gain'] for r in candidates]),'vs_readout_signflip_p':signflip(paired)}
            gain_p=holm([statistics[t]['gain_signflip_p'] for t in TASKS]);readout_p=holm([statistics[t]['vs_readout_signflip_p'] for t in TASKS])
            for i,t in enumerate(TASKS):
                q=statistics[t];q['gain_holm_p']=gain_p[i];q['vs_readout_holm_p']=readout_p[i]
                q['pass']=q['probe_gain']['lower95']>=.05 and q['direct']['mean']>=.65 and q['vs_readout']['mean']>0 and gain_p[i]<=.05 and readout_p[i]<=.05
        gate=sum(v.get('pass',False) for v in statistics.values())>=2
        result={'run':run.relative_to(root).as_posix(),'selection':selection,'confirmation_statistics':statistics,'screen_trials':len(dev),
          'confirmation_trials':len(confirmed),'primary_representation_gate':gate,'decision':'GO_P2_RESEARCH_ONLY' if gate else 'NO_GO_LOCAL_TARGET_FAMILY' if args.random_feedback else 'PIVOT_LOCAL_TARGET_V1',
          'resources':monitor.report(),'cap':cap,'elapsed_seconds':time.monotonic()-began,'deployed':False,
          'retention_tested':False,'normal_chat_vision_proven':False,'learnability_control_limitation':'No new BPTT control is executed; existing task controls differ in budget/dynamics. Failure does not establish impossibility.'}
        write_json(run/'result.json',result);write_json(root/('artifacts/p1-random-feedback-latest.json' if args.random_feedback else 'artifacts/p1-local-target-latest.json'),result)
        lines=['# P1 neuron-local target eligibility','','Same deterministic48-neuron ALIF, three temporal tasks. No core scale-up or live weights change.','',
          '| Task | SelectedLR | Development gain | Direct | Screen task pass |','|---|---:|---:|---:|---|']
        for t,v in task_summary.items():lines.append(f"| {t} | {selected} | {v['gain']*100:.2f}pp | {v['direct']*100:.2f}% | {v['gain']>=.05 and v['direct']>=.65} |")
        lines+=['',json.dumps(result,indent=2),'','This rule uses an explicit supervised target-code third factor and local pseudo-derivative, not unsupervised STDP, exact e-prop, or whole-network backprop. Missing temporal learnability control limits negative conclusions. P2 retention and chat/vision quality are not established.']
        (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('RESULT '+str(run),flush=True)


if __name__=='__main__':main()
