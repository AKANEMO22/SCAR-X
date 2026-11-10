"""Pre-registered Phase 2.1 small experiments. No optimizer/backprop dependency."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
from ..neural.core import Config
from ..neural.plasticity import PlasticCore
from ..learning.replay import ReplayBuffer
from ..runtime.resources import ResourceMonitor
from .runs import new_run,write_json
from .tasks21 import representation,context_task,ridge_probe
from .audit21 import audit_phase2


def interval(values):
    v=np.asarray(values,float); rng=np.random.default_rng(2101)
    means=rng.choice(v,(10000,len(v)),replace=True).mean(1)
    return {"mean":float(v.mean()),"lower95":float(np.quantile(means,.025)),"upper95":float(np.quantile(means,.975)),"n_seeds":len(v)}


def signflip(values):
    v=np.asarray(values,float); observed=abs(float(v.mean()))
    signs=np.array(list(itertools.product([-1,1],repeat=len(v))))
    return float(np.mean(np.abs((signs*v).mean(1))>=observed-1e-12))


def holm(values):
    order=np.argsort(values); adjusted=np.zeros(len(values)); last=0
    for rank,i in enumerate(order):
        last=max(last,min(1.,values[i]*(len(values)-rank))); adjusted[i]=last
    return adjusted.tolist()


def probe_trial(seed,task,rule,settings,shuffle=False):
    c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed)
    m=PlasticCore(c,rule); m.shuffle=shuffle
    initial=PlasticCore(c,"readout")
    x,y=representation(task,10_000+seed,settings['representation_train'])
    start=time.perf_counter()
    for i,(seq,label) in enumerate(zip(x,y)):
        m.forward(seq,track=True); m.learn(int(label))
        if (i+1)%64==0: m.consolidate()
    train_seconds=time.perf_counter()-start
    fit_x,fit_y=representation(task,20_000+seed,settings['probe_train'])
    test_x,test_y=representation(task,30_000+seed,settings['test_examples'])
    ood_x,ood_y=representation(task,40_000+seed,settings['test_examples'],ood=True)
    fit=m.features_many(fit_x,902); test=m.features_many(test_x,903)
    score=ridge_probe(fit,fit_y,test,test_y)
    original=ridge_probe(initial.features_many(fit_x,902),fit_y,initial.features_many(test_x,903),test_y)
    shuffled=m.features_many(test_x,903,lesion="permute")
    reset=m.features_many(test_x,903,lesion="reset")
    return {"seed":seed,"task":task,"rule":rule,"shuffle":shuffle,"probe":score,"initial_probe":original,"probe_gain":score-original,
            "accuracy":float(np.mean(m.predict_many(test_x,903)==test_y)),
            "frozen_accuracy":float(np.mean(initial.predict_many(test_x,903)==test_y)),
            "ood_probe":ridge_probe(fit,fit_y,m.features_many(ood_x,904),ood_y),
            "permuted_probe":ridge_probe(fit,fit_y,shuffled,test_y),"reset_probe":ridge_probe(fit,fit_y,reset,test_y),
            "train_seconds":train_seconds,"array_bytes":m.memory_bytes()},m


def retention_trial(seed,policy,consolidation,settings,rule="score",stage_budget=None,forced_updates=None):
    stages=settings['continual_stages']; c=Config(inputs=4+stages,outputs=4,hidden=48,fan_in=8,plastic_rows=12,seed=seed)
    m=PlasticCore(c,rule,consolidation)
    buffer=ReplayBuffer(settings['replay_capacity'],12,c.inputs,policy,seed+70000)
    rng=np.random.default_rng(seed+80000)
    test=[context_task(90000+seed*100+j,settings['retention_test_per_context'],j) for j in range(stages)]
    scores=[]; stage_seconds=[]; updates=[]
    for stage in range(stages):
        x,y=context_task(60000+seed*100+stage,settings['new_per_stage'],stage)
        fresh=list(zip(x,y)); old=buffer.sample(settings['replay_per_stage'])
        if not old: old=[fresh[int(j)] for j in rng.choice(len(fresh),settings['replay_per_stage'],replace=False)]
        mixed=fresh+old; rng.shuffle(mixed)
        start=time.perf_counter(); count=0
        for seq,label in mixed:
            if forced_updates is not None:
                if count>=forced_updates[stage]: break
            elif stage_budget is not None and count>0 and time.perf_counter()-start>=stage_budget: break
            m.forward(seq,track=True); m.learn(int(label)); count+=1
        m.consolidate()
        for seq,label in fresh: buffer.add(seq,int(label),stage)
        stage_seconds.append(time.perf_counter()-start); updates.append(count)
        scores.append([float(np.mean(m.predict_many(test[j][0],91000+j)==test[j][1])) for j in range(stage+1)])
    diagonal=np.array([scores[j][j] for j in range(stages)]); final=np.array(scores[-1])
    forgetting=np.array([max(scores[t][j] for t in range(j,stages))-final[j] for j in range(stages)])
    coverage=len(set(zip(buffer.context[:buffer.size].tolist(),buffer.y[:buffer.size].tolist())))
    return {"seed":seed,"policy":policy,"consolidation":consolidation,"rule":rule,"stage_budget":stage_budget,
            "acquisition":float(diagonal.mean()),"final_accuracy":float(final.mean()),"forgetting":float(forgetting.mean()),
            "worst_forgetting":float(forgetting.max()),"signed_drop":float(np.mean(diagonal-final)),"matrix":scores,
            "train_seconds":float(sum(stage_seconds)),"stage_seconds":stage_seconds,"updates":sum(updates),"updates_per_stage":updates,
            "array_bytes":m.memory_bytes(),"buffer_bytes":buffer.x.nbytes+buffer.y.nbytes+buffer.context.nbytes,"groups_covered":coverage},m,buffer


def run(root):
    root=Path(root); settings=json.loads((root/'configs/phase21.json').read_text())
    output=new_run(root,settings['protocol'],settings)
    deadline=time.monotonic()+settings['max_seconds']
    def emit(kind,row):
        with (output/'trials.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps({"kind":kind,**row})+'\n')
        print(f"{kind}: seed={row.get('seed')} rule={row.get('rule')} task={row.get('task',row.get('policy'))} score={row.get('probe',row.get('final_accuracy',0)):.3f}",flush=True)
        monitor.check()
        if time.monotonic()>deadline: raise TimeoutError("Pre-registered run deadline")
    try:
        with ResourceMonitor(settings['ram_limit']) as monitor:
            audit=audit_phase2(root); write_json(output/'phase2_audit.json',audit)
            print('Phase2 audit complete: 15 trials/checkpoints verified.',flush=True)
            dev_probe=[]; dev_ret=[]
            for seed in settings['development_seeds']:
                for task in settings['representation_tasks']:
                    for rule in settings['local_rules']:
                        row,_=probe_trial(seed,task,rule,settings); dev_probe.append(row); emit('development_representation',row)
                for policy in settings['replay_policies']:
                    for con in settings['consolidations']:
                        row,_,_=retention_trial(seed,policy,con,settings); dev_ret.append(row); emit('development_retention',row)
            rules=['stdp','score','score_meta']
            winner=max(rules,key=lambda r:np.mean([v['probe_gain'] for v in dev_probe if v['rule']==r]))
            baseline_acq=float(np.mean([v['acquisition'] for v in dev_ret if v['policy']=='reservoir' and v['consolidation']=='ema']))
            pairs=list(itertools.product(settings['replay_policies'],settings['consolidations']))
            def rows(pair): return [v for v in dev_ret if (v['policy'],v['consolidation'])==pair]
            eligible=[p for p in pairs if np.mean([v['acquisition'] for v in rows(p)])>=baseline_acq-settings['acquisition_noninferiority']]
            chosen=max(eligible,key=lambda p:(np.mean([v['final_accuracy'] for v in rows(p)]),-np.mean([v['forgetting'] for v in rows(p)]),-np.mean([v['train_seconds'] for v in rows(p)]))) if eligible else ('reservoir','ema')
            time_budget=float(np.median([v['train_seconds']/settings['continual_stages'] for v in rows(('reservoir','ema'))]))
            selection={"hidden_rule":winner,"replay":chosen[0],"consolidation":chosen[1],"matched_stage_seconds":time_budget,"made_before_confirmation":True}
            write_json(output/'selection.json',selection); print('LOCKED SELECTION '+json.dumps(selection),flush=True)
            confirmation_probe=[]; confirmation_ret=[]; checkpoint_checks=[]
            for seed in settings['confirmation_seeds']:
                for task in settings['representation_tasks']:
                    for name,rule,shuffle in [('selected',winner,False),('readout','readout',False),('shuffled',winner,True)]:
                        row,m=probe_trial(seed,task,rule,settings,shuffle); row['condition']=name
                        confirmation_probe.append(row); emit('confirmation_representation',row)
                        if task=='binding' and name=='selected':
                            path=output/f'plastic-{seed}.npz'; m.save(path); restored=PlasticCore.load(path)
                            x,y=representation('binding',50000+seed,4)
                            for seq,label in zip(x,y):
                                for obj in (m,restored): obj.forward(seq,track=True); obj.learn(int(label))
                            checkpoint_checks.append(all(np.array_equal(v,restored.arrays()[k]) for k,v in m.arrays().items()))
                for budget_name,budget in [('updates',None),('time',time_budget)]:
                    conditions=[('selected',chosen[0],chosen[1],'score'),('reference','reservoir','ema','score'),('no_hidden',chosen[0],chosen[1],'readout'),('no_consolidation',chosen[0],'none','score')]
                    for name,policy,con,rule in conditions:
                        row,m,buffer=retention_trial(seed,policy,con,settings,rule,budget); row.update(condition=name,budget=budget_name)
                        confirmation_ret.append(row); emit('confirmation_retention_'+budget_name,row)
                        if name=='selected' and budget_name=='updates':
                            buffer.save(output/f'replay-{seed}.npz'); restored=ReplayBuffer.load(output/f'replay-{seed}.npz')
                            checkpoint_checks.append(all(np.array_equal(a[0],b[0]) and a[1]==b[1] for a,b in zip(buffer.sample(16),restored.sample(16))))
            analysis=analyze(confirmation_probe,confirmation_ret,settings)
            analysis['checkpoint_continuation']=all(checkpoint_checks)
            analysis['resources']=monitor.report()
            analysis['verdict']='GO_SMALL_RESEARCH' if analysis['representation_pass'] and analysis['retention_pass'] and all(checkpoint_checks) and not monitor.exceeded else 'PIVOT_NO_SCALE'
            result={"settings":settings,"selection":selection,"analysis":analysis,"development_representation":dev_probe,"development_retention":dev_ret,"confirmation_representation":confirmation_probe,"confirmation_retention":confirmation_ret}
            write_json(output/'result.json',result)
            (output/'report.md').write_text('# Phase 2.1\n\n```json\n'+json.dumps(analysis,indent=2)+'\n```\n',encoding='utf-8')
            print('RESULT '+str(output),flush=True)
            return output,result
    except BaseException as exc:
        write_json(output/'failure.json',{"type":type(exc).__name__,"message":str(exc)})
        (output/'report.md').write_text(f'# Incomplete Phase 2.1\n\n{type(exc).__name__}: {exc}\n')
        raise


def analyze(probes,retention,s):
    contrasts={}; pvalues=[]
    for task in s['representation_tasks']:
        selected=[r for r in probes if r['condition']=='selected' and r['task']==task]
        readout=[r for r in probes if r['condition']=='readout' and r['task']==task]
        shuffled=[r for r in probes if r['condition']=='shuffled' and r['task']==task]
        gains=[r['probe_gain'] for r in selected]; pvalues.append(signflip(gains))
        contrasts[task]={"probe_gain":interval(gains),"vs_readout":interval([a['probe']-b['probe'] for a,b in zip(selected,readout)]),
                         "vs_shuffled":interval([a['probe']-b['probe'] for a,b in zip(selected,shuffled)]),
                         "accuracy":interval([r['accuracy'] for r in selected]),"probe":interval([r['probe'] for r in selected]),
                         "frozen_gain":interval([r['accuracy']-r['frozen_accuracy'] for r in selected]),
                         "ood_probe":interval([r['ood_probe'] for r in selected]),
                         "permutation_drop":interval([r['probe']-r['permuted_probe'] for r in selected]),
                         "reset_drop":interval([r['probe']-r['reset_probe'] for r in selected])}
    adjusted=holm(pvalues); passes=[]
    for task,p in zip(s['representation_tasks'],adjusted):
        c=contrasts[task]; c['holm_signflip_p']=p
        c['pass']=c['probe_gain']['lower95']>=s['representation_gain'] and c['accuracy']['mean']>=s['minimum_accuracy'] and c['vs_readout']['lower95']>0 and p<=.05
        passes.append(c['pass'])
    retention_analysis={}; absolute=True
    for budget in ('updates','time'):
        rows={name:[v for v in retention if v['condition']==name and v['budget']==budget] for name in ('selected','reference','no_hidden','no_consolidation')}
        summaries={name:{k:interval([v[k] for v in values]) for k in ('acquisition','final_accuracy','forgetting','worst_forgetting','train_seconds','updates')} for name,values in rows.items()}
        delta=interval([b['forgetting']-a['forgetting'] for a,b in zip(rows['selected'],rows['reference'])])
        acquisition=interval([a['acquisition']-b['acquisition'] for a,b in zip(rows['selected'],rows['reference'])])
        gate=all(r['forgetting']<=s['forgetting_mean_max'] and r['worst_forgetting']<=s['forgetting_worst_max'] and r['acquisition']>=s['minimum_accuracy'] and r['final_accuracy']>=r['acquisition']-.03 for r in rows['selected'])
        absolute &= gate
        retention_analysis[budget]={"conditions":summaries,"forgetting_improvement":delta,"acquisition_difference":acquisition,
                                    "relative_improvement":delta['lower95']>=s['retention_improvement'] and acquisition['lower95']>=-s['acquisition_noninferiority'],"absolute_pass":gate}
    return {"representation":contrasts,"representation_pass":sum(passes)>=2,"retention":retention_analysis,"retention_pass":absolute}


if __name__=='__main__':
    run(Path.cwd())
