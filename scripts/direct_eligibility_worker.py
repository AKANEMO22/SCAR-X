"""One registered proposal30 trial/cold check, real numeric outputs only."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,argparse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from direct_eligibility_common import make,load,reference,dataset,TASKS,METHODS,DEV,CONFIRM
from visual_word_control_common import atomic
from scarx.evaluation.tasks21 import ridge_probe
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.work_trace import append


def trial(job):
    phase,task,seed,method,runpath=job
    if phase not in ('development','confirmation') or task not in TASKS or method not in METHODS or seed not in (DEV if phase=='development' else CONFIRM):raise ValueError('Unregistered eligibility trial')
    run=Path(runpath);cap=apply_memory_limit();began=time.perf_counter()
    if method=='bptt_control':
        from scarx.evaluation.controls import torch
        torch.set_num_threads(1)
    identifier=f'{phase}-{task}-{seed}-{method}';m=make(task,seed,method);ref=reference(m);initial_payload=m.dumps();initial={n:getattr(ref.core,n).copy() for n in ('w_in','w_rec','w_out')};previous={n:a.copy() for n,a in initial.items()}
    train,labels=dataset(task,seed+1370000,1024);losses=[];ticks_before=ref.total_ticks
    with ResourceMonitor(interval=.1) as monitor:
        for i,(seq,label) in enumerate(zip(train,labels),1):
            losses.append(m.learn(seq,int(label)))
            if i%64==0:
                neural={'computed':48,'allocated':48,'tick':ref.total_ticks,'spikes':ref.core.spikes.astype(int).tolist(),'voltage':np.round(ref.core.voltage,4).tolist(),
                    'sample_kind':f'Actual {method} ALIF trajectory; research only, one-stepfeedbackcandidate or isolatedBPTTcontrol'}
                atomic(run/f'heartbeat-{identifier}.json',{'phase':phase,'task':task,'seed':seed,'method':method,'examples':i,'loss_mean':float(np.mean(losses[-64:])),
                    'hidden_updates':ref.hidden_updates,'elapsed_seconds':time.perf_counter()-began,'neural':neural});monitor.check()
            if i%256==0:
                delta={n:float(np.linalg.norm(getattr(ref.core,n)-a)) for n,a in previous.items()}
                append(run,'learning',{'scope':f'PROPOSAL30 RESEARCH · {phase}/{task}/seed{seed}/{method} · NOT MAIN',
                    'prompt':f'{task}: experiences{i-255}–{i}','passes':1,'sample_count':256,'loss_mean':float(np.mean(losses[-256:])),'losses':[float(np.mean(losses[-256:]))],
                    'update_steps':ref.core.updates,'update_unit':'supervised sequence/head cumulative','hidden_update_steps':ref.hidden_updates,
                    'neural_ticks':256*12,'weight_delta':delta,'attempted_weight_delta':delta,'changed_synapses':{n:int(np.count_nonzero(getattr(ref.core,n)!=a)) for n,a in previous.items()},
                    'accepted':True,'rolled_back':False,'guard_checked':False,
                    'reason':('ISOLATED BPTT control, globaltemporal backward; never main.' if method=='bptt_control' else 'Approved ONE readout-errorstep + own-neuron forwardeligibility; no network/time backward. Readout-only for zero/frozen.')+' Research labels/scope, no servingguard or mainweightschange.'})
                previous={n:getattr(ref.core,n).copy() for n in previous}
            if time.perf_counter()-began>180:raise TimeoutError('Registered180s eligibility trialcap')
        train_seconds=time.perf_counter()-began;training_ticks=ref.total_ticks-ticks_before;assert training_ticks==12288 and ref.core.updates==1024 and ref.head.updates==1024
        payload=m.dumps();artifact=root/'artifacts/direct-readout-eligibility'/run.name;artifact.mkdir(parents=True,exist_ok=True);checkpoint=artifact/(identifier+'.eligibility.zip');checkpoint.write_bytes(payload)
        view=load(method,payload);v=reference(view);original=reference(load(method,initial_payload))
        fit,fy=dataset(task,seed+1380000,128);test,ty=dataset(task,seed+1390000,256 if phase=='development' else 512)
        ff=view.features_many(fit);tf=view.features_many(test);probabilities=v.head.probabilities(tf,v.output_mask);pred=probabilities.argmax(1)
        probe=ridge_probe(ff,fy,tf,ty);initial_probe=ridge_probe(original.features_many(fit),fy,original.features_many(test),ty)
        rng=np.random.default_rng(seed+1410000);permuted=np.stack([seq[rng.permutation(len(seq))] for seq in test])
        time_permuted=ridge_probe(ff,fy,view.features_many(permuted),ty);state_reset=ridge_probe(ff,fy,view.features_many(test,True),ty)
        ood,oy=dataset(task,seed+1400000,128,True);ood_probe=ridge_probe(ff,fy,view.features_many(ood),oy) if ood is not None else None
        bag=ridge_probe(fit.sum(1),fy,test.sum(1),ty)
        assert m.dumps()==payload
        duplicate=load(method,payload);loss=duplicate.learn(train[0],int(labels[0]));reply=duplicate.predict(test[0]);draw=int(reference(duplicate).signal_rng.integers(10000))
        cold_expected={'loss':loss,'prediction':reply,'signal_rng_draw':draw,'full_state_sha256':hashlib.sha256(duplicate.dumps()).hexdigest()}
        outputs=[{'id':identifier+'/'+str(i),'input':test[i].tolist(),'prediction':int(pred[i]),'target':int(ty[i]),'probabilities':probabilities[i].tolist()} for i in range(len(test))]
        for r in outputs[:4]:append(run,'answer',{'scope':f'PROPOSAL30 {method} · {phase}/{task}/seed{seed} · NUMERIC, NOT CHAT/MAIN',
            'case_id':r['id'],'prompt':json.dumps(r['input']),'generated':str(r['prediction']),'target':str(r['target']),'probabilities':r['probabilities'],
            'input_scope':'Actual float32 [12,8] tensor, deterministic singleALIFforward. Prediction from learnedneuralhead, no goldinput/hostanswerlookup. Numericclassification, NOT normalconversation/vision.',
            'qwen':None,'qwen_missing_reason':'No comparableQwenmeasurement on numericneuralinput; existingtextreferences not substituted.',
            'active':48,'allocated':48,'neural_ticks':12,'checkpoint':checkpoint.relative_to(root).as_posix()})
        row={'phase':phase,'task':task,'seed':seed,'method':method,'direct':float(np.mean(pred==ty)),'probe':probe,'initial_probe':initial_probe,'probe_gain':probe-initial_probe,
            'probe_time_permuted':time_permuted,'probe_state_reset':state_reset,'ood_probe':ood_probe,'bag_only':bag,'experiences':1024,'head_updates':ref.core.updates,
            'hidden_updates':ref.hidden_updates,'training_ticks':training_ticks,'array_bytes':ref.memory_bytes(),'checkpoint_bytes':len(payload),
            'hidden_delta_in':float(np.linalg.norm(ref.core.w_in-initial['w_in'])),'hidden_delta_rec':float(np.linalg.norm(ref.core.w_rec-initial['w_rec'])),
            'mean_training_loss':float(np.mean(losses)),'train_seconds':train_seconds,'seconds':time.perf_counter()-began,
            'test_inputs_sha256':hashlib.sha256(test.tobytes()).hexdigest(),'test_targets_sha256':hashlib.sha256(ty.astype(np.int32).tobytes()).hexdigest(),
            'checkpoint':checkpoint.relative_to(root).as_posix(),'checkpoint_sha256':hashlib.sha256(payload).hexdigest(),'cold_expected':cold_expected,
            'core_weight_fingerprint':ref.core.fingerprint(),'head_sha256':hashlib.sha256(ref.head.dumps()).hexdigest(),'resources':monitor.report(),'cap':cap,
            'candidate_network_backward':False,'network_backward_in_this_trial':method=='bptt_control','control_only':method=='bptt_control','proposal30_approved':True}
        if method=='bptt_control':row.update(backward_passes=m.backward_passes,surrogate_forward_ticks=m.surrogate_forward_ticks,max_graph_state_error=m.max_graph_state_error,max_graph_feature_error=m.max_graph_feature_error)
        monitor.check()
        if time.perf_counter()-began>180:raise TimeoutError('Registered180s eligibility incl evaluation')
        atomic(run/(identifier+'.json'),{'record':row,'outputs':outputs,'neural_sample':neural});return row,neural


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args();path=Path(args.job).resolve()
    if not path.is_relative_to(root/'runs') or path.stat().st_size>65536:raise ValueError('Registered eligibilityjob required')
    job=json.loads(path.read_text(encoding='utf-8'));phase,task,seed,method,runpath=job['trial'];run=Path(runpath).resolve();assert run==path.parent
    if job.get('mode')=='cold':
        apply_memory_limit()
        if method=='bptt_control':
            from scarx.evaluation.controls import torch
            torch.set_num_threads(1)
        checkpoint=(root/job['checkpoint']).resolve()
        if not checkpoint.is_relative_to(root/'artifacts/direct-readout-eligibility') or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=job['sha256']:raise ValueError('Coldcheckpoint provenance')
        m=load(method,checkpoint.read_bytes());train,y=dataset(task,seed+1370000,1024);test,_=dataset(task,seed+1390000,256 if phase=='development' else 512)
        loss=m.learn(train[0],int(y[0]));reply=m.predict(test[0]);draw=int(reference(m).signal_rng.integers(10000))
        atomic(run/f'cold-{phase}-{task}-{seed}-{method}.json',{'loss':loss,'prediction':reply,'signal_rng_draw':draw,'full_state_sha256':hashlib.sha256(m.dumps()).hexdigest()});return
    row,_=trial(tuple(job['trial']));print(json.dumps({'task':task,'seed':seed,'method':method,'direct':row['direct'],'probe_gain':row['probe_gain']}),flush=True)


if __name__=='__main__':main()
