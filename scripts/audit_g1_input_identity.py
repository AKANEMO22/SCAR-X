"""Inputinformation diagnostic; no neural model or template-answer selection."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.evaluation.input_identity import fit,transform,dual_ridge,VIEWS
from scarx.evaluation.dialogue_probe import local_probe
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from generate_g1_state_paraphrases import INTENTS


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    info=json.loads((root/'artifacts/g1-curriculum-reviewed.json').read_text(encoding='utf-8'));rows=json.loads((root/info['data']).read_text(encoding='utf-8'))['items']
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'));train=[]
    for i,f in enumerate(corpus['families']):train += [{'id':f'known-{i}-{j}','prompt':p,'intent':INTENTS[i]} for j,p in enumerate(f['train'])]
    train += [r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='development']
    prior_info=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'));prior=json.loads((root/prior_info['data']).read_text(encoding='utf-8'))['items']
    metadata=fit(r['prompt'] for r in train);y=np.array([INTENTS.index(r['intent']) for r in train])
    run=new_run(root,'g1-input-identity-control-v1',{'protocol':'experiments/g1-input-identity-control-v1.md',
        'display_title':'Inputcontrol · hash hay nhận dạng từ làm mất thông tin?',
        'model_scope':'NONNEURALinput/probecontrols only, no answeralgorithm, coretraining or largerneuralmodel',
        'views':VIEWS,'train_prompts':len(train),'new_core_training':False,'selected_vocabulary_uses_labels':False})
    records=[]
    with ResourceMonitor(interval=.1) as monitor:
        for index,view in enumerate(VIEWS):
            a,train_coverage=transform(metadata,[r['prompt'] for r in train],view);out={}
            for name,cases in (('development',dev),('prior',prior)):
                b,coverage=transform(metadata,[r['prompt'] for r in cases],view);ty=np.array([INTENTS.index(r['intent']) for r in cases])
                ridge=dual_ridge(a,y,b,ty);local=[local_probe(a,y,b,ty,s) for s in (1901,1907,1913)]
                summary={'ridge':ridge['accuracy'],'local_mean':float(np.mean([r['accuracy'] for r in local])),
                    'local_orders':[r['accuracy'] for r in local],'coverage':coverage}
                out[name]=summary;write_json(run/f'{view}-{name}.json',{'summary':summary,'ridge_predictions':ridge['predictions'],'local_predictions':[r['predictions'] for r in local],'targets':ty.tolist(),'case_ids':[r['id'] for r in cases]})
            record={'view':view,'dimensions':a.shape[1],'splits':out,'train_coverage':train_coverage};records.append(record)
            progress(run,view,index+1,4,metrics={'dev_local':out['development']['local_mean'],'dev_ridge':out['development']['ridge'],'dimensions':a.shape[1]})
            monitor.check()
            if time.perf_counter()-began>120:raise TimeoutError('Inputcontrol120s deadline')
        summary={r['view']:{name:values for name,values in r['splits'].items()} for r in records}
        informative=summary['dictionary128']['development']['local_mean']>=.65 and summary['dictionary128']['development']['local_mean']-summary['legacy128']['development']['local_mean']>=.1 and summary['dictionary128']['prior']['local_mean']>=.65
        result={'run':run.relative_to(root).as_posix(),'decision':'INPUT_IDENTITY_INFORMATIVE_RESEARCH_ONLY' if informative else 'STOP_INPUT_IDENTITY_CONTROL_V1',
            'summary':summary,'dimensions':{r['view']:r['dimensions'] for r in records},'metadata':metadata,'informative_screen':informative,
            'new_core_training':False,'deployed':False,'normal_chat_or_vision_proven':False,'retention_confirmed':False,'resources':monitor.report(),'cap':cap,
            'limits':['No neuralcore trained, probe intentlabels never used as actual responsealgorithm.','Three probeorders not independentmodels; data alreadyinspected, no formal gate.',
                'Fullworddictionary has morefeatures, an informationreference only; no scale permission.']}
        (run/'report.md').write_text('# Fixedbudget input identity control\n\nNonneuralinformation diagnostic, no answerlookup/coretraining/deployment.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-input-identity-latest.json',result)
        print(json.dumps({'run':str(run),'decision':result['decision'],'summary':summary,'dimensions':result['dimensions']}))


if __name__=='__main__':main()
