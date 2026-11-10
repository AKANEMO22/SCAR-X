"""Observe existing neuronstate channels, train diagnostic localheads only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.paired_score import PairedScoreCore
from scarx.evaluation.neuron_state_views import state_features,local_head,VIEWS
from scarx.evaluation.paired_retention import mixed_stream,ridge_multiclass
from scarx.evaluation.tasks21 import context_task
from scarx.evaluation.phase21 import interval
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def features(model,x,seed):
    out={k:[] for k in VIEWS}
    for index in range(0,len(x),512):
        block=state_features(model,x[index:index+512],seed+index*100,8)
        for key,value in block.items():out[key].append(value)
    return {key:np.concatenate(value) for key,value in out.items()}


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    source=json.loads((root/'artifacts/p2-paired-score-diagnostic-latest.json').read_text(encoding='utf-8'))
    rows=json.loads((root/source['run']/'trials.json').read_text(encoding='utf-8'));models=[r for r in rows if r['condition'].startswith('offline_joint')]
    run=new_run(root,'p2-neuron-state-observability-v1',{'protocol':'experiments/p2-neuron-state-observability-v1.md',
        'display_title':'P2 · thông tin trong voltage/adaptation hay spikes?',
        'model_scope':'ExistingALIFstate channels, no dynamics/coreweight changes; analogreadout assumption disclosed',
        'source_run':source['run'],'views':VIEWS,'new_core_training':False,'new_candidate':False})
    private=root/'local/p2-neuron-states'/run.name;private.mkdir(parents=True);measured=[]
    with ResourceMonitor(interval=.1) as monitor:
        for row in models:
            started=time.perf_counter();seed=row['seed'];m=PairedScoreCore.loads((root/row['checkpoint']).read_bytes());state=m.dumps()
            x,y,_=mixed_stream(seed);tests=[context_task(seed+7500000+131*j,128,j,20) for j in range(20)];tx=np.concatenate([t[0] for t in tests]);ty=np.concatenate([t[1] for t in tests])
            train=features(m,x,seed+9100000);test=features(m,tx,seed+9200000)
            np.testing.assert_array_equal(state_features(m,x[:4],1091,8)['filtered_spikes'],m.features_many(x[:4],1091,samples=8))
            for view in VIEWS:
                ridge,pred=ridge_multiclass(train[view],y,test[view],ty);local=local_head(train[view],y,test[view],ty)
                record={'seed':seed,'condition':row['condition'],'view':view,'ridge':ridge,'local_output_head':local['accuracy'],
                    'head_updates':local['updates'],'head_loss':local['mean_loss'],'source_state_unchanged':True,
                    'feature_dimensions':48,'analog_state_channel':view not in ('filtered_spikes','last_spikes')}
                measured.append(record);name=f'{seed}-{row["condition"]}-{view}'
                np.savez(private/(name+'-features.npz'),train=train[view],test=test[view])
                write_json(run/(name+'.json'),{'summary':record,'local_predictions':local['predictions'],'ridge_predictions':pred.tolist(),'targets':ty.tolist()})
                append(run,'answer',{'scope':'P2 diagnostic localhead · '+view+' · not serving/neuralcoretraining',
                    'case_id':name,'prompt':json.dumps(tx[0].tolist()),'generated':str(local['predictions'][0]),'target':str(int(ty[0])),
                    'input_scope':'Actual float32[12,24], MC8originalALIF. Outputlocaldiagnostichead on48existingneuronstatefeatures; analogreadout ifdeclared, no inputbypass.',
                    'qwen':None,'active':48,'allocated':48,'neural_ticks':96,'checkpoint':row['checkpoint']})
                progress(run,name,len(measured),30,metrics={k:record[k] for k in ('ridge','local_output_head','head_loss','analog_state_channel')})
                monitor.check()
                if time.perf_counter()-started>90 or time.perf_counter()-began>600:raise TimeoutError('State-observability audit deadline')
            assert m.dumps()==state
        summary={};screen={};effects={}
        for condition in ('offline_joint_full','offline_joint_frozen'):
            summary[condition]={};screen[condition]={};effects[condition]={}
            for view in VIEWS:
                rs=[r for r in measured if r['condition']==condition and r['view']==view]
                gain=[r['local_output_head']-next(b['local_output_head'] for b in measured if b['condition']==condition and b['seed']==r['seed'] and b['view']=='filtered_spikes') for r in rs]
                summary[condition][view]={'ridge':float(np.mean([r['ridge'] for r in rs])),'local':float(np.mean([r['local_output_head'] for r in rs])),'gain':float(np.mean(gain))}
                screen[condition][view]=bool(summary[condition][view]['local']>=.65 and np.mean(gain)>=.05 and sum(v>0 for v in gain)>=2)
                effects[condition][view]={'gain':interval(gain),'exploratory_only':True}
        informative=any(v for c in screen.values() for v in c.values())
        result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':'STATE_CHANNEL_INFORMATIVE_RESEARCH_ONLY' if informative else 'STOP_STATE_OBSERVABILITY_SCREEN_V1',
            'summary':summary,'informative_screen':screen,'paired_effects':effects,'measurements':measured,
            'all_core_state_unchanged':True,'reference_forward_parity':True,'new_core_training':False,'deployed':False,
            'normal_chat_or_vision_proven':False,'retention_confirmed':False,'resources':monitor.report(),'cap':cap,
            'limits':['Analogpresynapticstatechannel is an additionalassumption, not pureaxonspikecommunication or biologicalproof.',
                'Offlinejointsource hadfuturelabels, probesare diagnostic not deployable lifelongmodels.','Three developmentseeds, no fullP2gate/confirmation or worddecoderfamilyreopening.']}
        (run/'report.md').write_text('# Existing ALIF neuronstate readout observability\n\nNo sourceweight/dynamicschanges; analogchannel assumption explicit, localdiagnosticheads only.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/p2-neuron-state-observability-latest.json',result)
        print(json.dumps({'run':str(run),'decision':result['decision'],'summary':summary,'screen':screen}))


if __name__=='__main__':main()
