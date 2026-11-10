"""P1 diagnostic: separate optimizer/row coverage; no candidate or deployment."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,itertools
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from scarx.neural.core import SpikingCore,Config
from scarx.evaluation.readout_conditioning import ConditionedReadout
from scarx.evaluation.tasks21 import ridge_probe
from scarx.evaluation.phase21 import interval,signflip,holm
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from run_p1_local_classifier_v2 import dataset,feature_view,block_trace,snapshot

TASKS=('order','xor','strict_binding');SEEDS=(13217,13219,13229)
CONDITIONS=('frozen','adam-all','adam-rotating','sgd-all','sgd-rotating')


def setup(task,seed):
    config=Config(inputs=8,hidden=48,outputs=2,fan_in=8,plastic_rows=12,seed=seed,anchor_strength=0)
    initial=feature_view(SpikingCore(config));head=ConditionedReadout(initial.w_out,initial.bias,'normalized')
    calibration,_=dataset(task,seed+5200000,128)
    for f in initial.features_many(calibration):head.observe(f)
    head.freeze();return config,initial,head


def evaluate(control,initial,task,seed,condition,run,artifact,metrics):
    model=control.copied_core();head=control.head;view=feature_view(model)
    fit,fy=dataset(task,seed+5400000,128);test,ty=dataset(task,seed+5500000,256);ood,oy=dataset(task,seed+5600000,128,True)
    fit_features=view.features_many(fit);test_features=view.features_many(test)
    np.testing.assert_array_equal(control.features(test,False).detach().numpy(),test_features)
    p=head.probabilities(test_features);predictions=p.argmax(1)
    probe=ridge_probe(fit_features,fy,test_features,ty)
    original=ridge_probe(initial.features_many(fit),fy,initial.features_many(test),ty)
    name=f'{task}-{seed}-{condition}';core_path=artifact/(name+'.core.npz');head_path=artifact/(name+'.head.npz')
    core_path.write_bytes(model.dumps());head_path.write_bytes(head.dumps())
    restore=SpikingCore.loads(core_path.read_bytes());restored_head=ConditionedReadout.loads(head_path.read_bytes())
    for key,value in model.arrays().items():np.testing.assert_array_equal(value,restore.arrays()[key])
    np.testing.assert_array_equal(restored_head.probabilities(feature_view(restore).features_many(test)),p)
    row={'task':task,'seed':seed,'condition':condition,'direct':float(np.mean(predictions==ty)),
        'probe':probe,'initial_probe':original,'probe_gain':probe-original,
        'initial_headroom':1-original,'probe_state_reset':ridge_probe(fit_features,fy,view.features_many(test,'reset'),ty),
        'probe_time_permuted':ridge_probe(fit_features,fy,view.features_many(test,'permute'),ty),
        'ood_probe':ridge_probe(fit_features,fy,view.features_many(ood),oy) if ood is not None else None,
        'checkpoint':core_path.relative_to(root).as_posix(),'head_checkpoint':head_path.relative_to(root).as_posix(),
        'checkpoint_weights_state_head_exact':True,'optimizer_resume_claim':False,
        'array_bytes':model.memory_bytes(),'optimizer_bytes':control.optimizer_bytes(),**metrics}
    for i in range(3):
        append(run,'answer',{'scope':'P1 BPTT đối chứng cô lập · optimizer/coverage audit',
            'case_id':name+f'/{i}','prompt':json.dumps(test[i].tolist()),
            'input_scope':'Actual float32 tensor [12,8]; diagnostic classification, not text conversation. Batched inference; individual latency unmeasured.',
            'generated':str(int(predictions[i])),'target':str(int(ty[i])),'probabilities':p[i].tolist(),
            'qwen':None,'active':48,'allocated':48,'neural_ticks':12,'checkpoint':row['checkpoint']})
    neural={'computed':48,'allocated':48,'tick':2048*12,'spikes':model.spikes.astype(int).tolist(),
        'voltage':np.round(model.voltage,4).tolist(),'sample_kind':'last actual training tick; isolated control','task':task,'condition':condition}
    write_json(run/(name+'.json'),{'summary':row,'targets':ty.tolist(),'predictions':predictions.tolist(),'neural_sample':neural})
    return row,neural


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();deadline=began+1800
    run=new_run(root,'p1-control-factor-audit-v1',{'protocol':'experiments/p1-control-factor-audit-v1.md',
        'display_title':'P1 audit · optimizer và coverage · BPTT chỉ đối chứng',
        'model_scope':'Đối chứng cô lập, không sửa lõi chính hay mở lại family đã dừng',
        'seeds':SEEDS,'conditions':CONDITIONS,'examples':2048,'hidden':48,'new_local_candidate':False,'workers':1})
    artifact=root/'artifacts/control-factors'/run.name;artifact.mkdir(parents=True)
    from scarx.evaluation.controls import ClassifierLearnabilityControl,torch
    torch.set_num_threads(1)
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    records=[];comparisons={}
    with ResourceMonitor(interval=.1) as monitor:
        for task,seed,condition in itertools.product(TASKS,SEEDS,CONDITIONS):
            start=time.perf_counter();config,initial,head=setup(task,seed)
            frozen=condition=='frozen';optimizer,policy=('adam','all') if frozen else condition.split('-')
            model=ClassifierLearnabilityControl(config,head,frozen,optimizer,policy)
            x,y=dataset(task,seed+5300000,2048)
            np.testing.assert_array_equal(model.features(x[:3],False).detach().numpy(),initial.features_many(x[:3]))
            origin=snapshot(model.copied_core(),head);before=snapshot(model.copied_core(),head)
            losses=[];allloss=[];gradient_norms=[];clipped_elements=0
            for index,(sequence,target) in enumerate(zip(x,y)):
                losses.append(model.learn(sequence,int(target)))
                if model.last_gradient_norm is not None:gradient_norms.append(model.last_gradient_norm)
                clipped_elements+=model.last_clipped_elements
                if (index+1)%512==0:
                    core=model.copied_core();block_trace(run,task,seed,condition,.002,losses,[],core,head,before,0 if frozen else 512,True)
                    allloss+=losses;losses=[];before=snapshot(core,head);monitor.check()
                    progress(run,f'{task} · seed{seed} · {condition} · {index+1}/2048examples',len(records),45,
                        metrics={'examples':index+1,'loss':float(np.mean(allloss[-512:])),'optimizer':optimizer,'coverage':policy},
                        neural={'computed':48,'allocated':48,'tick':(index+1)*12,'spikes':core.spikes.astype(int).tolist(),'voltage':np.round(core.voltage,4).tolist()},task=task)
                    if time.perf_counter()-start>180 or time.perf_counter()>deadline:raise TimeoutError('Registered audit deadline')
            core=model.copied_core()
            metrics={'main_updates':head.updates,'hidden_updates':model.hidden_updates,
                'optimizer':optimizer if not frozen else None,'coverage':policy if not frozen else 'none',
                'scheduled_hidden_synapse_writes':2048*(768 if policy=='all' else 192) if not frozen else 0,
                'hidden_delta_in':float(np.linalg.norm(core.w_in-origin['w_in'])),'hidden_delta_rec':float(np.linalg.norm(core.w_rec-origin['w_rec'])),
                'mean_loss':float(np.mean(allloss)),'gradient_norm_mean':float(np.mean(gradient_norms)) if gradient_norms else None,
                'gradient_norm_max':max(gradient_norms) if gradient_norms else None,'update_clipped_elements':clipped_elements,
                'train_seconds':time.perf_counter()-start,'resources':monitor.report(),'cap':cap,
                'control_only':True,'forward_parity':True}
            row,neural=evaluate(model,initial,task,seed,condition,run,artifact,metrics);records.append(row)
            write_json(run/'trials.json',records)
            progress(run,f'Đã đo {task} · {condition} · seed{seed}',len(records),45,metrics={k:row[k] for k in ('direct','probe_gain','train_seconds')},neural=neural,task=task)
            print(json.dumps({k:row[k] for k in ('task','seed','condition','direct','probe_gain','train_seconds')}),flush=True)
        contrasts=(('adam-all','sgd-all'),('adam-rotating','sgd-rotating'),('adam-all','adam-rotating'),('sgd-all','sgd-rotating'))
        primary=[]
        for task in TASKS:
            comparisons[task]={}
            for a,b in contrasts:
                effects={}
                for metric in ('direct','probe_gain','mean_loss','train_seconds','hidden_delta_in','hidden_delta_rec'):
                    values=[next(r[metric] for r in records if r['task']==task and r['seed']==s and r['condition']==a)-next(r[metric] for r in records if r['task']==task and r['seed']==s and r['condition']==b) for s in SEEDS]
                    effects[metric]={'values':values,'bootstrap95':interval(values),'signflip_p':signflip(values)}
                effects['direct']['operationally_material_mean_5pp']=abs(effects['direct']['bootstrap95']['mean'])>=.05
                comparisons[task][a+' minus '+b]=effects;primary.append(effects['direct'])
        for value,p in zip(primary,holm([v['signflip_p'] for v in primary])):value['holm_12contrasts_p']=p
        result={'run':run.relative_to(root).as_posix(),'decision':'DIAGNOSTIC_ONLY','primary_representation_gate':False,
            'control_summary':{t:{c:{'direct':float(np.mean([r['direct'] for r in records if r['task']==t and r['condition']==c])),
                'probe_gain':float(np.mean([r['probe_gain'] for r in records if r['task']==t and r['condition']==c]))} for c in CONDITIONS} for t in TASKS},
            'comparisons':comparisons,'control_trials':len(records),'resources':monitor.report(),'cap':cap,'torch_version':torch.__version__,
            'family_reopened':False,'new_local_candidate':False,'deployed':False,'normal_chat_or_vision_proven':False,'retention_proven':False,
            'limitations':['Three development seeds; no confirmatory efficacy claim.','Global Adam clock; inactive weights/vector moments frozen.','Full BPTT graph computed for rotating controls; no sparse efficiency claim.','Equal LR is not equal effective step between optimizers.']}
        (run/'report.md').write_text('# P1 control factors\n\nIsolated diagnostic only; stopped local family stays stopped.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/p1-control-factors-latest.json',result)
        print('RESULT '+str(run),flush=True)


if __name__=='__main__':main()
