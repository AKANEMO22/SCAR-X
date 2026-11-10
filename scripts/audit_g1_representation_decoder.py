"""Split input/hidden/decoder diagnostics, no serving mutation or new core training."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.dialogue_probe import inspect_example,ridge,local_probe
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from generate_g1_state_paraphrases import INTENTS


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    source=json.loads((root/'artifacts/g1-curated-curriculum-latest.json').read_text(encoding='utf-8'))
    models=source['conditions'];assert len(models)==3 and all(r['condition']=='surface_expanded' for r in models)
    info=json.loads((root/'artifacts/g1-curriculum-reviewed.json').read_text(encoding='utf-8'));data=json.loads((root/info['data']).read_text(encoding='utf-8'))['items']
    old=json.loads((root/'artifacts/g1-basic-rehearsal-latest.json').read_text(encoding='utf-8'))
    private_source=(root/old['conditions'][0]['candidate_private_path']).parent
    skipped=set(json.loads((private_source/'source-outputs.json').read_text(encoding='utf-8'))['skipped'])
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'));train=[]
    for intent,family in enumerate(corpus['families']):
        train += [{'id':f'known-{intent}-{j}','prompt':p,'target':family['answer'],'class':intent,'split':'train'} for j,p in enumerate(family['train']) if f'known-{intent}-{j}' not in skipped]
    train += [{**r,'class':INTENTS.index(r['intent'])} for r in data if r['split']=='train']
    dev=[{**r,'class':INTENTS.index(r['intent'])} for r in data if r['split']=='development']
    prior=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'))
    agy=[{**r,'class':INTENTS.index(r['intent']),'split':'prior_diagnostic'} for r in json.loads((root/prior['data']).read_text(encoding='utf-8'))['items']]
    run=new_run(root,'g1-representation-decoder-audit-v1',{'protocol':'experiments/g1-representation-decoder-audit-v1.md',
        'display_title':'G1 audit · input/hidden có nhận ý hay decoder làm hỏng câu?',
        'model_scope':'Immutable3privateclones, diagnostic heads only; goldteacher view explicitly leaky, never an answer algorithm',
        'source_run':source['run'],'new_core_training':False,'new_candidate':False,'splits':{'train':len(train),'development':len(dev),'prior':len(agy)}})
    private=root/'local/g1-representation-audit'/run.name;private.mkdir(parents=True);results=[]
    with ResourceMonitor(interval=.1) as monitor:
        for model_index,model_info in enumerate(models):
            started=time.perf_counter();payload=(root/model_info['candidate_private_path']).read_bytes();model=UnifiedCore.loads(payload);state=model.dumps()
            split_data={};extract_count=0
            for name,rows in (('train',train),('development',dev),('prior',agy)):
                output=[];features={k:[] for k in ('context','prompt_hidden','gold_teacher_hidden_leaky','idle')}
                for case in rows:
                    measured,vectors=inspect_example(payload,case['prompt'],case['target']);output.append({**measured,'id':case['id'],'class':case['class']})
                    for key,v in vectors.items():features[key].append(v)
                    extract_count+=1
                arrays={k:np.stack(v) for k,v in features.items()};np.savez(private/f"order{model_info['curriculum_order_seed']}-{name}-features.npz",**arrays)
                write_json(private/f"order{model_info['curriculum_order_seed']}-{name}-outputs.json",output)
                split_data[name]={'outputs':output,'features':arrays,'y':np.array([r['class'] for r in rows],np.int32)}
                progress(run,f'Đã đo features/tokenloss: order{model_info["curriculum_order_seed"]} · {name}',model_index*3+len(split_data),9,
                    metrics={'examples':len(rows),'strict_style_exact':float(np.mean([r['strict_style_exact'] for r in output])),
                        'first_token_accuracy':float(np.mean([r['first_token_correct'] for r in output]))})
                monitor.check()
            summaries={};probes={}
            for name,data_split in split_data.items():
                outputs=data_split['outputs'];summaries[name]={key:float(np.mean([r[key] for r in outputs])) for key in
                    ('strict_style_exact','first_token_correct','teacher_forced_token_accuracy','teacher_forced_nll','prefix_fraction','repeated_bigram')}
                probes[name]={}
                for feature in ('context','prompt_hidden','gold_teacher_hidden_leaky','idle'):
                    probes[name][feature]={'ridge':ridge(split_data['train']['features'][feature],split_data['train']['y'],data_split['features'][feature],data_split['y'])}
                    if feature in ('context','prompt_hidden'):
                        probes[name][feature]['local_softmax']=[local_probe(split_data['train']['features'][feature],split_data['train']['y'],data_split['features'][feature],data_split['y'],s) for s in (1901,1907,1913)]
                if name!='train':
                    for r in outputs[:8]:append(run,'answer',{'scope':f'Immutable G1 clone · order{model_info["curriculum_order_seed"]} · {name} · no serving change',
                        'case_id':r['id'],'prompt':r['prompt'],'target':r['target'],'generated':r['generated'],
                        'input_scope':'Actual autoregressive freegeneration. Goldtarget used ONLY for separate diagnostic teacherforcing/probe labels, not this answer.',
                        'qwen':None,'active':model.active_count,'allocated':model.core.config.hidden,'checkpoint':'private immutable checkpoint inlocal/'})
            record={'curriculum_order_seed':model_info['curriculum_order_seed'],'summary':summaries,'intent_probes':probes,
                'informative_prompt_probe':probes['development']['prompt_hidden']['ridge']['accuracy']>=.65 and
                    probes['development']['prompt_hidden']['ridge']['accuracy']-probes['development']['idle']['ridge']['accuracy']>=.2,
                'core_state_unchanged':model.dumps()==state,'seconds':time.perf_counter()-started}
            assert record['core_state_unchanged'];results.append(record);write_json(run/f"order{model_info['curriculum_order_seed']}.json",record)
            if time.perf_counter()-started>120 or time.perf_counter()-began>360:raise TimeoutError('Representation/decoder audit deadline')
        summary={split:{'strict_style_exact':float(np.mean([r['summary'][split]['strict_style_exact'] for r in results])),
            'first_token_accuracy':float(np.mean([r['summary'][split]['first_token_correct'] for r in results])),
            'teacher_forced_token_accuracy':float(np.mean([r['summary'][split]['teacher_forced_token_accuracy'] for r in results])),
            'teacher_forced_nll':float(np.mean([r['summary'][split]['teacher_forced_nll'] for r in results])),
            'intent_ridge':{feature:float(np.mean([r['intent_probes'][split][feature]['ridge']['accuracy'] for r in results])) for feature in ('context','prompt_hidden','gold_teacher_hidden_leaky','idle')}} for split in ('train','development','prior')}
        result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':'REPRESENTATION_DECODER_DIAGNOSTIC_ONLY',
            'summary':summary,'models':results,'new_core_training':False,'all_core_state_unchanged':True,'deployed':False,
            'normal_chat_or_vision_proven':False,'retention_confirmed':False,'resources':monitor.report(),'cap':cap,
            'limits':['Threeclones sharehiddenweights; not independentmodelinitializations.','Teacherforcedwhole-responsefeature is intentionally labelleaky and cannot support inferencequalityclaim.',
                'Ridge/local8classheads diagnostic only, never a template answer selector or language model.','Source/families/priorcases already inspected; no independentnewgate or performance-baseddeployment.']}
        (run/'report.md').write_text('# G1 representation/decoder split\n\nDiagnostic only, immutablecore; goldteacherfeature explicitlyleaky. No runtime/template answer selector or newqualitygate.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-representation-decoder-latest.json',result)
        print(json.dumps({'run':str(run),'summary':summary,'core_state_unchanged':True}))


if __name__=='__main__':main()
