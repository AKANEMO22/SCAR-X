"""Continuous mixture/noisy feedback study. No task IDs reach the learner."""
import json
from pathlib import Path
import time
import numpy as np
from ..neural.core import Config
from ..learning.lifelong import LifelongLearner
from ..learning.replay import ReplayBuffer
from ..runtime.resources import ResourceMonitor
from .tasks21 import representation
from .runs import new_run,write_json
from .phase21 import interval,signflip,holm

TASKS=('xor','order','binding')


def continuous_stream(seed,n=1280):
    rng=np.random.default_rng(seed+110000)
    pools=[representation(task,seed+120000+i,n) for i,task in enumerate(TASKS)]
    for i in range(n):
        phase=2*np.pi*i/512
        weights=np.exp(2*np.cos(phase-np.arange(3)*2*np.pi/3)); weights/=weights.sum()
        family=int(rng.choice(3,p=weights)); x=pools[family][0][i]; clean=int(pools[family][1][i])
        corrupt=bool(rng.random()<.2)
        observed=clean^int(corrupt)
        rejected_flag=bool(rng.random()<(.7 if corrupt else .1))
        # Return clean/family only for evaluator diagnostics; never passed to update.
        yield x,observed,(.2 if rejected_flag else .9),clean,family


def feature_rank(model,samples):
    f=model.features_many(samples,617)
    singular=np.linalg.svd(f-f.mean(0),compute_uv=False)
    probabilities=singular/(singular.sum()+1e-12)
    return {"effective_rank":float(np.exp(-np.sum(probabilities*np.log(probabilities+1e-12)))),
            "feature_std_mean":float(f.std(0).mean()),"near_constant_fraction":float(np.mean(f.std(0)<.005))}


def trial(seed,mode,output):
    c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed)
    learner=LifelongLearner(c,mode); fresh=LifelongLearner(c,mode)
    buffer=ReplayBuffer(128,12,8,'reservoir',seed+140000)
    reject_rng=np.random.default_rng(seed+150000)
    tests=[representation(task,seed+130000+i,256) for i,task in enumerate(TASKS)]
    rank_initial=feature_rank(learner.model,tests[2][0]); scores=[]; accepted=rejected=accepted_noise=0; wall=0.; counts=[]
    for i,(x,label,reliability,clean,family) in enumerate(continuous_stream(seed)):
        start=time.perf_counter()
        reject=(reliability<.7) if mode=='selective' else (reject_rng.random()<.22 if mode=='random_reject' else False)
        if reject:
            rejected+=1
            replacement=buffer.sample(1)
            if replacement: learner.update(*replacement[0])
        else:
            accepted+=1; accepted_noise+=int(label!=clean)
            learner.update(x,label); buffer.add(x,label,0)
        if i%2==1:
            old=buffer.sample(1)
            if old: learner.update(*old[0])
        wall+=time.perf_counter()-start
        if (i+1)%64==0:
            scores.append([float(np.mean(learner.model.predict_many(tx,620+j)==ty)) for j,(tx,ty) in enumerate(tests)])
            counts.append(learner.model.updates)
    final=np.array(scores[-1]); forgetting=np.max(np.asarray(scores),axis=0)-final
    rank_final=feature_rank(learner.model,tests[2][0])
    weights=learner.model.w_in.copy()
    before=float(np.mean(learner.model.predict_many(tests[2][0],622)==tests[2][1]))
    novel_x,novel_y=representation('binding',seed+160000,64)
    # Novel target is a reversal with a familiar query input: measure adaptation, disclose conflict.
    novel_y=1-novel_y; probe_test_y=1-tests[2][1]
    old_novel=float(np.mean(learner.model.predict_many(tests[2][0],622)==probe_test_y))
    fresh_before=float(np.mean(fresh.model.predict_many(tests[2][0],622)==probe_test_y))
    for x,y in zip(novel_x,novel_y): learner.update(x,int(y)); fresh.update(x,int(y))
    learned=float(np.mean(learner.model.predict_many(tests[2][0],622)==probe_test_y))
    fresh_after=float(np.mean(fresh.model.predict_many(tests[2][0],622)==probe_test_y))
    location=output/f'{seed}-{mode}'; learner.save(location); buffer.save(location/'replay.npz')
    restored=LifelongLearner.load(location)
    for x,y in zip(novel_x[:4],novel_y[:4]): learner.update(x,int(y)); restored.update(x,int(y))
    resumed=all(np.array_equal(v,restored.model.arrays()[k]) for k,v in learner.model.arrays().items()) and all(np.array_equal(v,restored.si[k]) for k,v in learner.si.items())
    return {"seed":seed,"mode":mode,"matrix":scores,"final_accuracy":float(final.mean()),"per_family_final":final.tolist(),
            "forgetting":float(forgetting.mean()),"worst_forgetting":float(forgetting.max()),"accepted":accepted,"rejected":rejected,"accepted_noise":accepted_noise,
            "updates":counts[-1],"updates_per_window":counts,"train_seconds":wall,"rank_initial":rank_initial,"rank_final":rank_final,
            "new_rule_gain":learned-old_novel,"fresh_new_rule_gain":fresh_after-fresh_before,"new_rule_accuracy":learned,
            "post_probe_old_binding":1-learned,"before_probe_old_binding":before,"hidden_probe_drift":float(np.linalg.norm(weights-learner.model.w_in)),
            "importance_mean":float(learner.model.importance_in.mean()),"restart_equal":resumed,
            "model_arrays_bytes":learner.model.memory_bytes(),"si_arrays_bytes":sum(v.nbytes for v in learner.si.values()),"replay_bytes":buffer.x.nbytes+buffer.y.nbytes+buffer.context.nbytes}


def run(root):
    root=Path(root); settings=json.loads((root/'configs/phase21.json').read_text())
    output=new_run(root,'phase21-lifelong-v1',settings); rows=[]; deadline=time.monotonic()+settings['max_seconds']
    try:
        with ResourceMonitor(settings['ram_limit']) as monitor:
            for seed in settings['confirmation_seeds']:
                for mode in LifelongLearner.modes:
                    row=trial(seed,mode,output); rows.append(row)
                    with (output/'trials.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                    print(f'lifelong seed={seed} mode={mode} final={row["final_accuracy"]:.3f} forget={row["forgetting"]:.3f}',flush=True)
                    monitor.check()
                    if time.monotonic()>deadline: raise TimeoutError('Experiment time budget')
            summaries={mode:{key:interval([v[key] for v in rows if v['mode']==mode]) for key in ('final_accuracy','forgetting','worst_forgetting','train_seconds','new_rule_gain','fresh_new_rule_gain','accepted_noise','updates')} for mode in LifelongLearner.modes}
            comparisons={}; pvalues=[]
            for name,a,b in [('metaplasticity','score_meta','score_ema'),('importance_decay','meta_decay','score_meta'),('selective_quality','selective','random_reject')]:
                aa=[v for v in rows if v['mode']==a]; bb=[v for v in rows if v['mode']==b]
                delta=[x['final_accuracy']-y['final_accuracy'] for x,y in zip(aa,bb)]
                comparisons[name]={"accuracy_gain":interval(delta),"forgetting_reduction":interval([y['forgetting']-x['forgetting'] for x,y in zip(aa,bb)])}
                pvalues.append(signflip(delta))
            for name,p in zip(comparisons,holm(pvalues)): comparisons[name]['holm_p']=p
            passes={mode:all(v['forgetting']<=.03 and v['worst_forgetting']<=.05 and v['final_accuracy']>=.65 and v['restart_equal'] for v in rows if v['mode']==mode) for mode in LifelongLearner.modes}
            result={"summaries":summaries,"comparisons":comparisons,"absolute_gates":passes,"trials":rows,"resources":monitor.report(),
                    "interpretation":"Finite stream, observable input families but no task IDs/boundaries. Final 64-example reversal is a plasticity diagnostic, not simultaneous retention of contradictory labels. SI readout adaptation only."}
            write_json(output/'result.json',result)
            (output/'report.md').write_text('# Lifelong task-agnostic stream\n\n```json\n'+json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2)+'\n```\n')
            print('RESULT '+str(output),flush=True)
            return output,result
    except BaseException as exc:
        write_json(output/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        (output/'report.md').write_text(f'# Incomplete lifelong run\n\n{exc}\n')
        raise


if __name__=='__main__': run(Path.cwd())
