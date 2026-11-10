"""Immutable cue/time/noise diagnostic after the failed P2 baseline screen."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.paired_score import PairedScoreCore
from scarx.evaluation.paired_retention import mixed_stream,ridge_multiclass
from scarx.evaluation.tasks21 import context_task
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor


def features(model,x,seed,samples):
    output=[]
    for index in range(0,len(x),512):
        chunk=x[index:index+512];value=np.zeros((len(chunk),model.config.hidden),np.float32)
        for sample in range(0,samples,16):
            count=min(16,samples-sample)
            value+=model.features_many(chunk,seed+index*100+sample,samples=count)*(count/samples)
        output.append(value)
    return np.concatenate(output)


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    source=json.loads((root/'artifacts/p2-paired-score-diagnostic-latest.json').read_text(encoding='utf-8'))
    rows=json.loads((root/source['run']/'trials.json').read_text(encoding='utf-8'))
    run=new_run(root,'p2-cue-information-audit-v1',{'protocol':'experiments/p2-cue-information-audit-v1.md',
        'display_title':'P2 audit · cue theo thời gian và noise · không train core',
        'source_run':source['run'],'model_scope':'Read-only checkpoints; diagnostic probe only, no weight/RNG change',
        'prefixes':[2,6,12],'rollouts':[8,64],'conditions':['offline_joint_full','offline_joint_frozen'],'posthoc':True,'new_candidate':False})
    measured=[];controls=[]
    with ResourceMonitor(interval=.1) as monitor:
        selected=[r for r in rows if r['condition'] in ('offline_joint_full','offline_joint_frozen')]
        for index,row in enumerate(selected):
            start=time.perf_counter();seed=row['seed'];model=PairedScoreCore.loads((root/row['checkpoint']).read_bytes());before=model.dumps()
            train,ty,_=mixed_stream(seed);tests=[context_task(seed+7500000+131*j,128,j,20) for j in range(20)]
            test=np.concatenate([t[0] for t in tests]);labels=np.concatenate([t[1] for t in tests])
            if row['condition']=='offline_joint_frozen':
                def input_features(x):
                    cue=x[:,:2,:4].mean(1);context=x[:,:,4:].mean(1)
                    return np.concatenate((cue,context),axis=1),(cue[:,:,None]*context[:,None,:]).reshape(len(x),-1)
                tf=input_features(train);vf=input_features(test)
                for kind,a,b in zip(('raw_inputs','cue_context_product'),tf,vf):
                    accuracy,predictions=ridge_multiclass(a,ty,b,labels)
                    control={'seed':seed,'kind':kind,'accuracy':accuracy,'features':a.shape[1]};controls.append(control)
                    write_json(run/f'input-{seed}-{kind}.json',{'summary':control,'predictions':predictions.tolist(),'targets':labels.tolist()})
            for samples in (8,64):
                for prefix in (2,6,12):
                    a=features(model,train[:,:prefix],seed+8100000,samples);b=features(model,test[:,:prefix],seed+8200000,samples)
                    accuracy,predictions=ridge_multiclass(a,ty,b,labels)
                    record={'seed':seed,'condition':row['condition'],'prefix_ticks':prefix,'rollouts':samples,'accuracy':accuracy,
                        'extra_compute_diagnostic':samples>8,'core_updates':model.updates,'elapsed_model_seconds':time.perf_counter()-start}
                    measured.append(record);write_json(run/f'{seed}-{row["condition"]}-{samples}-{prefix}.json',{'summary':record,'predictions':predictions.tolist(),'targets':labels.tolist()})
                    progress(run,f'{row["condition"]} · seed{seed} · prefix{prefix} · MC{samples}',len(measured),36,
                        metrics={'probe_accuracy':accuracy,'prefix_ticks':prefix,'inference_rollouts':samples})
                    monitor.check()
                    if time.perf_counter()-start>180 or time.perf_counter()-began>900:raise TimeoutError('Cue audit budget')
            assert model.dumps()==before
        summary={c:{str(s):{str(p):float(np.mean([r['accuracy'] for r in measured if r['condition']==c and r['rollouts']==s and r['prefix_ticks']==p])) for p in (2,6,12)} for s in (8,64)} for c in ('offline_joint_full','offline_joint_frozen')}
        result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':'CUE_INFORMATION_DIAGNOSTIC_ONLY',
            'summary':summary,'input_controls':controls,'measurements':measured,'all_model_state_unchanged':True,
            'new_training':False,'new_candidate':False,'deployed':False,'normal_chat_or_vision_proven':False,'retention_confirmed':False,
            'resources':monitor.report(),'cap':cap,'limits':['Posthoc diagnostic only, no new gate/selection.','64rollouts add inference compute; not a deployed or equal-budget improvement.','Engineered input products are an informationcontrol, not neural skill or matchedcapacity algorithm.']}
        (run/'report.md').write_text('# P2 cue information diagnostic\n\nRead-only neural checkpoints, diagnostic heads; no core training or new efficacy gate.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/p2-cue-information-latest.json',result)
        print(json.dumps({'run':str(run),'summary':summary,'input_controls':controls}))


if __name__=='__main__':main()
