"""Registered P2 baseline diagnosis, unchanged paired-score rule; no deployment."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,itertools,hashlib
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from scarx.neural.core import Config
from scarx.neural.paired_score import PairedScoreCore
from scarx.learning.replay import ReplayBuffer
from scarx.evaluation.tasks21 import context_task
from scarx.evaluation.paired_retention import mixed_stream,ridge_multiclass,component_intervention,retention_metrics
from scarx.evaluation.phase21 import interval
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

SEEDS=(15217,15227,15241)
CONDITIONS=('stream_full_repeat','stream_frozen_repeat','stream_full_reservoir','stream_frozen_reservoir','offline_joint_full','offline_joint_frozen')


def weights(model):return {name:getattr(model,name).copy() for name in ('w_in','w_rec','w_out')}


def evaluate(model,tests,seed):
    features=[model.features_many(x,seed+7600000+131*j) for j,(x,y) in enumerate(tests)]
    predictions=[model.probabilities(f).argmax(1) for f in features]
    return features,predictions,[float(np.mean(p==y)) for p,(x,y) in zip(predictions,tests)]


def log_block(run,seed,condition,window,losses,model,before,examples):
    delta={n:float(np.linalg.norm(getattr(model,n)-old)) for n,old in before.items()}
    append(run,'learning',{'scope':'P2 baseline diagnostic · research clone, not serving',
        'prompt':f'{condition} · seed{seed} · window{window+1} · updates{model.updates}',
        'passes':1,'sample_count':examples,'losses':[float(np.mean(losses))],'loss_mean':float(np.mean(losses)),
        'update_steps':examples,'update_unit':'episode/head','hidden_update_steps':0 if model.frozen else examples*4,
        'weight_delta':delta,'attempted_weight_delta':delta,
        'changed_synapses':{n:int(np.count_nonzero(getattr(model,n)!=old)) for n,old in before.items()},
        'accepted':True,'rolled_back':False,'guard_checked':False,'guard_probe_count':0,
        'reason':'Unchanged local score learner; no test-score rollback, no serving update.'})


def trial(job):
    seed,condition,runpath,artpath=job;run=Path(runpath);artifact=Path(artpath);cap=apply_memory_limit()
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.perf_counter();frozen='frozen' in condition;offline=condition.startswith('offline');replay=condition.endswith('reservoir')
    config=Config(inputs=24,outputs=4,hidden=48,fan_in=8,plastic_rows=12,seed=seed,anchor_strength=0,hidden_lr=.002)
    m=PairedScoreCore(config,'independent','adam',False,frozen,False)
    calibration,_=context_task(seed+7200000,128,0,20);m.calibrate(m.features_many(calibration,seed+7100000))
    x,y,groups=mixed_stream(seed);stream_sha=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
    tests=[context_task(seed+7500000+131*j,128,j,20) for j in range(20)]
    buffer=ReplayBuffer(128,12,24,'reservoir' if replay else 'none',seed+7700000)
    original=weights(m);before=weights(m);losses=[];scores=[];interventions=[];train_seconds=0.;stored_features=None
    initial_bytes=m.dumps();previous_head=(m.w_out.copy(),m.bias.copy());last_predictions=None
    with ResourceMonitor(interval=.2) as monitor:
        if offline:
            indices=np.concatenate((np.arange(len(x)),np.arange(1,len(x),2)))
            np.random.default_rng(seed+7800000).shuffle(indices)
            for i,index in enumerate(indices):
                start=time.perf_counter();losses.append(m.learn_example(x[index],int(y[index]))['loss']);train_seconds+=time.perf_counter()-start
                if (i+1)%288==0:
                    log_block(run,seed,condition,i//288,losses[-288:],m,before,288);before=weights(m);monitor.check()
                    if time.perf_counter()-began>180:raise TimeoutError('Registered180s P2trial limit')
            stored_features,last_predictions,final_scores=evaluate(m,tests,seed)
        else:
            for i,(sequence,label) in enumerate(zip(x,y)):
                start=time.perf_counter();losses.append(m.learn_example(sequence,int(label))['loss']);buffer.add(sequence,int(label),0)
                if (i+1)%2==0:
                    item=buffer.sample(1) if replay else []
                    old_x,old_y=item[0] if item else (sequence,int(label))
                    losses.append(m.learn_example(old_x,old_y)['loss'])
                train_seconds+=time.perf_counter()-start
                if (i+1)%192==0:
                    window=i//192;new_features,last_predictions,final_scores=evaluate(m,tests,seed);scores.append(final_scores)
                    current_head=(m.w_out.copy(),m.bias.copy())
                    if window:
                        labels=np.concatenate([tests[j][1] for j in range(window)])
                        interventions.append({'window':window+1,**component_intervention(np.concatenate(stored_features[:window]),np.concatenate(new_features[:window]),labels,
                            previous_head,current_head,m.head_mean,m.head_scale)})
                    stored_features=new_features;previous_head=current_head
                    checkpoint=artifact/f'{seed}-{condition}-window{window+1}.npz';checkpoint.write_bytes(m.dumps())
                    log_block(run,seed,condition,window,losses[-288:],m,before,288);before=weights(m)
                    monitor.check()
                    if any(not np.isfinite(a).all() for a in m.arrays().values()):raise ValueError('Nonfinite P2state')
                    if time.perf_counter()-began>180:raise TimeoutError('Registered180s P2trial limit')
        # Offline refit is a diagnostic only, never imported into runtime.
        fit=np.concatenate([m.features_many(x[i:i+512],seed+7900000+i) for i in range(0,len(x),512)])
        probe,probe_predictions=ridge_multiclass(fit,y,np.concatenate(stored_features),np.concatenate([t[1] for t in tests]))
        path=artifact/f'{seed}-{condition}.npz';replay_path=artifact/f'{seed}-{condition}.replay.npz'
        path.write_bytes(m.dumps());replay_path.write_bytes(buffer.dumps())
        state={'cursor':len(x),'updates':m.updates,'stream_sha256':stream_sha,'mode':condition,'replay':replay_path.relative_to(root).as_posix()}
        write_json(artifact/f'{seed}-{condition}.state.json',state)
        a=PairedScoreCore.loads(m.dumps());b=PairedScoreCore.loads(path.read_bytes());ra=ReplayBuffer.loads(buffer.dumps());rb=ReplayBuffer.loads(replay_path.read_bytes())
        sa=ra.sample(1);sb=rb.sample(1)
        if sa:
            np.testing.assert_array_equal(sa[0][0],sb[0][0]);assert sa[0][1]==sb[0][1];nx,ny=sa[0]
        else:nx,ny=x[0],int(y[0])
        assert a.learn_example(nx,ny)==b.learn_example(nx,ny) and a.dumps()==b.dumps() and ra.dumps()==rb.dumps()
        if frozen:
            for name in ('w_in','w_rec'):np.testing.assert_array_equal(getattr(m,name),original[name])
            assert all(v['hidden_only_drop']==0 for v in interventions)
        metrics=retention_metrics(scores) if not offline else {'acquisition':None,'mean_forgetting':None,'worst_forgetting':None,'signed_bwt':None,'joint_gate':False}
        checkpoint=path.relative_to(root).as_posix()
        row={'seed':seed,'condition':condition,'offline_future_labels':offline,'final':float(np.mean(final_scores)),
            'per_context_final':final_scores,'joint_refit_probe':probe,'updates':m.updates,'hidden_updates':m.hidden_updates,
            'train_seconds':train_seconds,'seconds':time.perf_counter()-began,'mean_loss':float(np.mean(losses)),
            'mean_head_only_drop':float(np.mean([v['head_only_drop'] for v in interventions])) if interventions else None,
            'mean_hidden_only_drop':float(np.mean([v['hidden_only_drop'] for v in interventions])) if interventions else None,
            'mean_interaction_drop':float(np.mean([v['interaction_drop'] for v in interventions])) if interventions else None,
            'model_arrays_bytes':m.memory_bytes(),'replay_arrays_bytes':buffer.x.nbytes+buffer.y.nbytes+buffer.context.nbytes,
            'raw_training_pool_bytes':x.nbytes+y.nbytes+groups.nbytes,'checkpoint':checkpoint,'resume_exact':True,
            'resources':monitor.report(),'cap':cap,**metrics}
        if not offline:row['matrix']=scores
        for j in (0,9,19):append(run,'answer',{'scope':'P2 diagnostic · offline future-label control' if offline else 'P2 stream diagnostic · no serving update',
            'case_id':f'{seed}-{condition}-context{j}','prompt':json.dumps(tests[j][0][0].tolist()),
            'input_scope':'Actual float32 [12,24], context cue observable;4class task,8rolloutmean. Not ordinary chat/vision.',
            'generated':str(int(last_predictions[j][0])),'target':str(int(tests[j][1][0])),'qwen':None,
            'active':48,'allocated':48,'neural_ticks':96,'checkpoint':checkpoint})
        neural={'computed':48,'allocated':48,'tick':m.ticks,'spikes':m.spikes.astype(int).tolist(),'voltage':np.round(m.voltage,4).tolist(),
            'sample_kind':'last actual training trajectory; local score, no backward'}
        write_json(run/f'{seed}-{condition}.json',{'summary':row,'component_interventions':interventions,
            'final_predictions':[p.tolist() for p in last_predictions],'targets':[t[1].tolist() for t in tests],
            'refit_predictions':probe_predictions.tolist(),'state':state,'neural_sample':neural})
        return row,neural


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    run=new_run(root,'p2-paired-score-diagnostic-v1',{'protocol':'experiments/p2-paired-score-diagnostic-v1.md',
        'display_title':'P2 · học mới, forgetting và head/hidden · baseline diagnostic',
        'model_scope':'Research clone; unchanged local score. Offlinejoint sees futurelabels only as diagnostic control.',
        'seeds':SEEDS,'conditions':CONDITIONS,'contexts':20,'fresh_examples':3840,'updates':5760,'hidden':48,'workers':2,
        'new_importance_candidate':False,'task_boundaries_given_to_learner':False,'observable_context_cues':True})
    artifact=root/'artifacts/paired-retention'/run.name;artifact.mkdir(parents=True)
    records=[];jobs=iter([(s,c,str(run),str(artifact)) for s,c in itertools.product(SEEDS,CONDITIONS)]);pending={}
    with ResourceMonitor(interval=.1) as monitor:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for _ in range(2):
                job=next(jobs);pending[pool.submit(trial,job)]=job
            while pending:
                ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED);monitor.check()
                if time.perf_counter()-began>1800:raise TimeoutError('Registered1800s P2round limit')
                for future in ready:
                    pending.pop(future);row,neural=future.result();records.append(row)
                    progress(run,f'{row["condition"]} · seed{row["seed"]}',len(records),18,
                        metrics={k:row[k] for k in ('final','acquisition','mean_forgetting','joint_refit_probe','train_seconds')},neural=neural)
                    write_json(run/'trials.json',records)
                    print(json.dumps({k:row[k] for k in ('seed','condition','final','acquisition','mean_forgetting','worst_forgetting','joint_refit_probe','train_seconds')}),flush=True)
                    try:job=next(jobs);pending[pool.submit(trial,job)]=job
                    except StopIteration:pass
        summaries={c:{k:(float(np.mean([r[k] for r in records if r['condition']==c])) if next(r[k] for r in records if r['condition']==c) is not None else None)
            for k in ('final','acquisition','mean_forgetting','worst_forgetting','joint_refit_probe','train_seconds','mean_head_only_drop','mean_hidden_only_drop')} for c in CONDITIONS}
        gates={c:all(r['joint_gate'] for r in records if r['condition']==c) for c in CONDITIONS if not c.startswith('offline')}
        comparisons={}
        for a,b in (('stream_full_reservoir','stream_full_repeat'),('stream_frozen_reservoir','stream_frozen_repeat'),
            ('stream_full_reservoir','stream_frozen_reservoir'),('offline_joint_full','offline_joint_frozen')):
            comparisons[a+' minus '+b]={key:interval([next(r[key] for r in records if r['seed']==s and r['condition']==a)-next(r[key] for r in records if r['seed']==s and r['condition']==b) for s in SEEDS])
                for key in ('final','joint_refit_probe','train_seconds')}
        result={'run':run.relative_to(root).as_posix(),'decision':'P2_BASELINE_DIAGNOSTIC_ONLY','summary':summaries,'absolute_gates':gates,
            'comparisons':comparisons,'trials':len(records),'resources':monitor.report(),'cap':cap,'deployed':False,
            'normal_chat_or_vision_proven':False,'retention_confirmed':False,'new_importance_candidate':False,
            'limits':['Three development seeds only, no confirmation/equal-time efficacy claim.','Observable context cues; episode resets; not an unmarked natural-world stream.',
                'Offlinejoint has futurelabels; refithead diagnostic never imported into runtime.','Same5760updates across conditions; no hidden rule retuning, no new scale.']}
        (run/'report.md').write_text('# P2 paired-score baseline diagnostic\n\nNo new method/deployment; preserve joint acquisition/forgetting gates.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/p2-paired-score-diagnostic-latest.json',result)
        print('RESULT '+str(run),flush=True)


if __name__=='__main__':main()
