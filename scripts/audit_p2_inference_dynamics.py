"""Fixed-weight cold/stochastic inference diagnostic; no core training."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.paired_score import PairedScoreCore
from scarx.evaluation.tasks21 import context_task
from scarx.evaluation.paired_retention import mixed_stream,ridge_multiclass
from scarx.evaluation.inference_dynamics import inference_features
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def features(model,x,seed,temperature,samples):
    return np.concatenate([inference_features(model,x[i:i+512],seed+i*100,temperature,samples) for i in range(0,len(x),512)])


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    source=json.loads((root/'artifacts/p2-paired-score-diagnostic-latest.json').read_text(encoding='utf-8'))
    rows=json.loads((root/source['run']/'trials.json').read_text(encoding='utf-8'))
    selected=[r for r in rows if r['condition'] in ('offline_joint_full','offline_joint_frozen')]
    run=new_run(root,'p2-inference-dynamics-audit-v1',{'protocol':'experiments/p2-inference-dynamics-audit-v1.md',
        'display_title':'P2 · audit firing noise / deterministic ALIF · không train',
        'source_run':source['run'],'temperatures':[0.,.1,.4],'requested_rollouts':[1,8],'prefixes':[6,12],
        'model_scope':'Immutable checkpoint inference views; changed emission dynamics, not a deployed model or lifelong method',
        'new_training':False,'posthoc_exploratory':True})
    measurements=[]
    with ResourceMonitor(interval=.1) as monitor:
        for row in selected:
            start=time.perf_counter();seed=row['seed'];model=PairedScoreCore.loads((root/row['checkpoint']).read_bytes());original=model.dumps()
            x,y,_=mixed_stream(seed);tests=[context_task(seed+7500000+131*j,128,j,20) for j in range(20)]
            tx=np.concatenate([t[0] for t in tests]);ty=np.concatenate([t[1] for t in tests])
            parity=x[:4];np.testing.assert_array_equal(inference_features(model,parity,341,.4,8),model.features_many(parity,341,samples=8))
            for temperature in (0.,.1,.4):
                for samples in (1,8):
                    for prefix in (6,12):
                        a=features(model,x[:,:prefix],seed+8300000,temperature,samples);b=features(model,tx[:,:prefix],seed+8400000,temperature,samples)
                        probe,predictions=ridge_multiclass(a,y,b,ty);p=model.probabilities(b);direct=p.argmax(1)
                        effective=1 if temperature==0 else samples
                        record={'seed':seed,'condition':row['condition'],'temperature':temperature,'requested_rollouts':samples,
                            'effective_rollouts':effective,'prefix_ticks':prefix,'direct_original_head':float(np.mean(direct==ty)),
                            'refit_probe':probe,'core_state_unchanged':True,'elapsed_model_seconds':time.perf_counter()-start}
                        measurements.append(record)
                        name=f'{seed}-{row["condition"]}-T{temperature}-MC{samples}-L{prefix}'
                        write_json(run/(name+'.json'),{'summary':record,'original_head_predictions':direct.tolist(),
                            'probe_predictions':predictions.tolist(),'targets':ty.tolist()})
                        for i in range(2):append(run,'answer',{'scope':'P2 inference-dynamics diagnostic · original learned head',
                            'case_id':name+f'/{i}','prompt':json.dumps(tx[i,:prefix].tolist()),
                            'input_scope':f'Actual float32[{prefix},24], emission temperature{temperature}; original head/calibration fixed. Probe not used to answer.',
                            'generated':str(int(direct[i])),'target':str(int(ty[i])),'probabilities':p[i].tolist(),'qwen':None,
                            'active':48,'allocated':48,'neural_ticks':effective*prefix,'checkpoint':row['checkpoint']})
                        progress(run,f'{row["condition"]} · seed{seed} · T{temperature}/MC{samples}/L{prefix}',len(measurements),72,
                            metrics={k:record[k] for k in ('temperature','effective_rollouts','prefix_ticks','direct_original_head','refit_probe')})
                        monitor.check()
                        if time.perf_counter()-start>120 or time.perf_counter()-began>600:raise TimeoutError('Registered inference audit deadline')
            assert model.dumps()==original
        summary={c:{str(t):{str(s):{str(l):{'direct':float(np.mean([r['direct_original_head'] for r in measurements if r['condition']==c and r['temperature']==t and r['requested_rollouts']==s and r['prefix_ticks']==l])),
            'probe':float(np.mean([r['refit_probe'] for r in measurements if r['condition']==c and r['temperature']==t and r['requested_rollouts']==s and r['prefix_ticks']==l]))} for l in (6,12)} for s in (1,8)} for t in (0.,.1,.4)} for c in ('offline_joint_full','offline_joint_frozen')}
        result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':'INFERENCE_DYNAMICS_DIAGNOSTIC_ONLY',
            'summary':summary,'measurements':measurements,'all_state_unchanged':True,'reference_parity':True,
            'new_training':False,'deployed':False,'normal_chat_or_vision_proven':False,'retention_confirmed':False,
            'resources':monitor.report(),'cap':cap,'limits':['Changed inference dynamics vs training, no fresh validated deployment/selection.',
                'Refit head is diagnostic, never a runtime or hidden skill claim.','Offline joint checkpoints learned futurelabels; cannot be used as a lifelong starting point.']}
        (run/'report.md').write_text('# P2 inference dynamics diagnostic\n\nImmutable core/oldhead, exploratory new inference views; no training, selection or deployment.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/p2-inference-dynamics-latest.json',result)
        print(json.dumps({'run':str(run),'summary':summary,'all_state_unchanged':True}))


if __name__=='__main__':main()
