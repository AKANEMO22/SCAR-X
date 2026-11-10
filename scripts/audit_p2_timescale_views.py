"""Fixed ALIF timescale inference views; no model training or deployment."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.paired_score import PairedScoreCore
from scarx.evaluation.inference_dynamics import inference_features,timescale_view,TIMESCALE_VIEWS
from scarx.evaluation.paired_retention import mixed_stream,ridge_multiclass
from scarx.evaluation.tasks21 import context_task
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor


def features(model,x,seed,dynamics):
    return np.concatenate([inference_features(model,x[i:i+512],seed+i*100,.4,8,dynamics) for i in range(0,len(x),512)])


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    source=json.loads((root/'artifacts/p2-paired-score-diagnostic-latest.json').read_text(encoding='utf-8'))
    rows=json.loads((root/source['run']/'trials.json').read_text(encoding='utf-8'))
    source_rows=[r for r in rows if r['condition'] in ('offline_joint_full','offline_joint_frozen')]
    run=new_run(root,'p2-timescale-view-audit-v1',{'protocol':'experiments/p2-timescale-view-audit-v1.md',
        'display_title':'P2 · ALIF time-constant views · giữ nguyên weights',
        'source_run':source['run'],'views':TIMESCALE_VIEWS,'hidden':48,'synapse_topology_unchanged':True,
        'model_scope':'Proposed dynamics diagnostic only; no learner/new weights, no deployed capability or local-rule change',
        'new_training':False,'posthoc_exploratory':True})
    records=[]
    with ResourceMonitor(interval=.1) as monitor:
        for row in source_rows:
            start=time.perf_counter();seed=row['seed'];model=PairedScoreCore.loads((root/row['checkpoint']).read_bytes());state=model.dumps()
            x,y,_=mixed_stream(seed);tests=[context_task(seed+7500000+131*j,128,j,20) for j in range(20)]
            tx=np.concatenate([t[0] for t in tests]);ty=np.concatenate([t[1] for t in tests])
            base=timescale_view(model.config,'baseline',seed)
            np.testing.assert_array_equal(inference_features(model,x[:4],451,.4,8,base),model.features_many(x[:4],451,samples=8))
            for view in TIMESCALE_VIEWS:
                dynamics=timescale_view(model.config,view,seed)
                a=features(model,x,seed+8700000,dynamics);b=features(model,tx,seed+8800000,dynamics)
                probe,pred=ridge_multiclass(a,y,b,ty);probabilities=model.probabilities(b);actual=probabilities.argmax(1)
                record={'seed':seed,'condition':row['condition'],'view':view,'refit_probe':probe,
                    'direct_original_head':float(np.mean(actual==ty)),'neuron_metadata_bytes':sum(v.nbytes for v in dynamics.values()),
                    'core_state_unchanged':True,'seconds_model':time.perf_counter()-start}
                records.append(record);name=f'{seed}-{row["condition"]}-{view}'
                write_json(run/(name+'.json'),{'summary':record,'decays':{k:v.tolist() for k,v in dynamics.items()},
                    'probe_predictions':pred.tolist(),'original_head_predictions':actual.tolist(),'targets':ty.tolist()})
                append(run,'answer',{'scope':'ALIF timescale diagnostic view · original head, unchanged learned weights',
                    'case_id':name,'prompt':json.dumps(tx[0].tolist()),'generated':str(int(actual[0])),'target':str(int(ty[0])),
                    'input_scope':f'Actual float32[12,24], view {view}, MC8/temp.4. Neuron decay constants changed; no training or runtime capability claim.',
                    'qwen':None,'active':48,'allocated':48,'neural_ticks':96,'checkpoint':row['checkpoint']})
                progress(run,f'{row["condition"]} · seed{seed} · {view}',len(records),60,
                    metrics={'refit_probe':probe,'direct_original_head':record['direct_original_head'],'metadata_bytes':record['neuron_metadata_bytes']})
                monitor.check()
                if time.perf_counter()-start>120 or time.perf_counter()-began>600:raise TimeoutError('Registered timescale-view deadline')
            assert model.dumps()==state
        summary={};screens={};contrasts={}
        for condition in ('offline_joint_full','offline_joint_frozen'):
            summary[condition]={};screens[condition]={};contrasts[condition]={}
            for view in TIMESCALE_VIEWS:
                rs=[r for r in records if r['condition']==condition and r['view']==view]
                gain=[r['refit_probe']-next(b['refit_probe'] for b in records if b['condition']==condition and b['seed']==r['seed'] and b['view']=='baseline') for r in rs]
                mean=float(np.mean([r['refit_probe'] for r in rs]));effect=float(np.mean(gain))
                summary[condition][view]={'direct':float(np.mean([r['direct_original_head'] for r in rs])),'probe':mean,'gain':effect}
                screens[condition][view]=bool(mean>=.65 and effect>=.05 and sum(v>0 for v in gain)>=2)
                contrasts[condition][view]={'gain':interval(gain),'exploratory_source_data_reuse':True}
            for short in ('leak','adaptation','feature','all'):
                differences=[next(r['refit_probe'] for r in records if r['condition']==condition and r['seed']==s and r['view']==short+'_spread')-
                    next(r['refit_probe'] for r in records if r['condition']==condition and r['seed']==s and r['view']==short+'_mean') for s in sorted({r['seed'] for r in records})]
                contrasts[condition][short+'_spread_minus_mean']={'effect':interval(differences),'exploratory_source_data_reuse':True}
        informative=any(v for c in screens.values() for v in c.values())
        result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':'TIMESCALE_VIEW_INFORMATIVE_ONLY' if informative else 'STOP_TIMESCALE_VIEW_SCREEN_V1',
            'summary':summary,'informative_screen':screens,'contrasts':contrasts,'all_state_unchanged':True,'reference_parity':True,
            'new_training':False,'new_local_rule':False,'deployed':False,'normal_chat_or_vision_proven':False,'retention_confirmed':False,
            'resources':monitor.report(),'cap':cap,'limits':['Weights learned under different dynamics; no inference-view result is a trained architecture or new efficacy gate.',
                'No borrowed paper performance/energy claim; source uses other training algorithms.','No test-based production winner, hidden-rule retuning or larger model.']}
        (run/'report.md').write_text('# ALIF fixed timescale views\n\nExploratory immutable checkpoint dynamics, not a new learner or deployed ability.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/p2-timescale-view-latest.json',result)
        print(json.dumps({'run':str(run),'decision':result['decision'],'summary':summary,'informative_screen':screens}))


if __name__=='__main__':main()
