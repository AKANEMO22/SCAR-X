"""Fixed-budget generic neural-token search diagnostic, no model training/import."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,itertools
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.unified import UnifiedCore
from scarx.neural.language import tokenize
from scarx.learning.dialogue import normalized
from scarx.evaluation.dialogue_probe import canonical
from scarx.evaluation.neural_beam import decode
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

CONFIGS=((1,0.),(1,.6),(4,0.),(4,.6),(8,0.),(8,.6))


def evaluate(payload,cases,width,alpha,run,private,name):
    outputs=[];times=[];last=None
    for case in cases:
        m=UnifiedCore.loads(payload);canonical(m,case['prompt']);fingerprint=m.core.fingerprint()
        result=decode(m,case['prompt'],width,alpha)
        assert m.core.fingerprint()==fingerprint
        if result['search']['seconds']>5:raise TimeoutError('Registered5s beam case limit')
        row={'id':case['id'],'prompt':case['prompt'],'target':case['target'],'generated':result['text'],
            'strict_style_exact':tokenize(result['text'])==tokenize(case['target']),
            'normalized_style_exact':normalized(result['text'])==normalized(case['target']),
            'first_token_correct':bool(result['tokens']) and result['tokens'][0]==tokenize(case['target'])[0],
            'stopped':result['stopped'],'stop_reason':result['stop_reason'],'search':result['search']}
        outputs.append(row);times.append(result['search']['seconds']);last=m
    summary={key:float(np.mean([r[key] for r in outputs])) for key in ('strict_style_exact','normalized_style_exact','first_token_correct','stopped')}
    summary.update(mean_seconds=float(np.mean(times)),p95_seconds=float(np.quantile(times,.95)),max_seconds=max(times),
        mean_computation_ticks=float(np.mean([r['search']['computation_ticks'] for r in outputs])),
        peak_search_state_bytes=max(r['search']['peak_unique_state_array_bytes'] for r in outputs),
        budget_exhausted=sum(r['search']['budget_exhausted'] for r in outputs))
    write_json(private/(name+'-outputs.json'),outputs)
    neural={'computed':last.active_count,'allocated':last.core.config.hidden,'tick':last.ticks,
        'spikes':last.core.spikes.astype(int).tolist(),'voltage':np.round(last.core.voltage,4).tolist(),
        'sample_kind':'selected branch terminal virtual state, not global/runtime monotonictrace'}
    for row in outputs[:8]:append(run,'answer',{'scope':name+' · immutable G1 diagnostic search, not serving',
        'case_id':row['id'],'prompt':row['prompt'],'target':row['target'],'generated':row['generated'],
        'input_scope':'Generic token beam on same sharedweights; no target/template in search. Virtualpath and workticks distinguished; no runtimeclock compatibility claim.',
        'qwen':None,'active':last.active_count,'allocated':last.core.config.hidden,'neural_ticks':row['search']['computation_ticks'],
        'seconds':row['search']['seconds'],'checkpoint':'private existing checkpoint inlocal/'})
    return summary,outputs,neural


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    source=json.loads((root/'artifacts/g1-curated-curriculum-latest.json').read_text(encoding='utf-8'));models=source['conditions']
    info=json.loads((root/'artifacts/g1-curriculum-reviewed.json').read_text(encoding='utf-8'))
    dev=[r for r in json.loads((root/info['data']).read_text(encoding='utf-8'))['items'] if r['split']=='development']
    prior_info=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'))
    prior=json.loads((root/prior_info['data']).read_text(encoding='utf-8'))['items']
    run=new_run(root,'g1-bounded-beam-diagnostic-v1',{'protocol':'experiments/g1-bounded-beam-diagnostic-v1.md',
        'display_title':'G1 · generic beam decoder · không template/không học mới',
        'source_run':source['run'],'model_scope':'One sharedweightclone percase, boundedhypothesisstates; virtualclock diagnostic, not runtime',
        'widths':[1,4,8],'alphas':[0.,.6],'max_word_steps':256,'new_training':False})
    private=root/'local/g1-beam-diagnostic'/run.name;private.mkdir(parents=True);records=[]
    with ResourceMonitor(interval=.1) as monitor:
        for row in models:
            payload=(root/row['candidate_private_path']).read_bytes();order=row['curriculum_order_seed'];baseline=None
            # Width1 parity checks exact arrays/state before opening decoder results.
            a=UnifiedCore.loads(payload);b=UnifiedCore.loads(payload);canonical(a,dev[0]['prompt']);canonical(b,dev[0]['prompt'])
            oracle=a.generate(dev[0]['prompt'],24);result=decode(b,dev[0]['prompt'],1,0.)
            assert all(oracle[k]==result[k] for k in ('text','tokens','stopped','stop_reason','mean_probability')) and a.dumps()==b.dumps()
            for width,alpha in CONFIGS:
                name=f'order{order}-dev-B{width}-A{alpha}';summary,outputs,neural=evaluate(payload,dev,width,alpha,run,private,name)
                if (width,alpha)==(1,0.):baseline=outputs
                elif width==1:assert [r['generated'] for r in outputs]==[r['generated'] for r in baseline]
                record={'order':order,'width':width,'alpha':alpha,'summary':summary};records.append(record)
                write_json(run/(name+'.json'),record)
                progress(run,name,len(records),18,metrics=summary,neural=neural)
                monitor.check()
                if time.perf_counter()-began>300:raise TimeoutError('Registered300s beamround limit')
        scores={str(c):float(np.mean([r['summary']['normalized_style_exact'] for r in records if (r['width'],r['alpha'])==c])) for c in CONFIGS}
        selected=max(CONFIGS,key=lambda c:(scores[str(c)],-c[0],-c[1]));baseline_score=scores[str((1,0.))]
        selection={'width':selected[0],'alpha':selected[1],'development_scores':scores,'prior_opened':False}
        write_json(run/'selection-frozen.json',selection);post=[];per_order_gain=[];prior_gains=[]
        for index,row in enumerate(models):
            payload=(root/row['candidate_private_path']).read_bytes();order=row['curriculum_order_seed'];measurements={};outputs_by={}
            for label,(width,alpha) in (('greedy',(1,0.)),('selected',selected)):
                name=f'order{order}-prior-{label}';summary,outputs,neural=evaluate(payload,prior,width,alpha,run,private,name)
                measurements[label]=summary;outputs_by[label]=outputs;progress(run,name,index*2+(label=='selected')+1,6,metrics=summary,neural=neural)
            prior_gain=measurements['selected']['normalized_style_exact']-measurements['greedy']['normalized_style_exact'];prior_gains.append(prior_gain)
            dev_gain=next(r['summary']['normalized_style_exact'] for r in records if r['order']==order and (r['width'],r['alpha'])==selected)-next(r['summary']['normalized_style_exact'] for r in records if r['order']==order and (r['width'],r['alpha'])==(1,0.));per_order_gain.append(dev_gain)
            lost=sum(a['normalized_style_exact'] and not b['normalized_style_exact'] for a,b in zip(outputs_by['greedy'],outputs_by['selected']))
            post.append({'order':order,'summary':measurements,'prior_normalized_gain':prior_gain,'dev_normalized_gain':dev_gain,'lost_old_correct_prior':lost})
        informative=scores[str(selected)]>=.25 and scores[str(selected)]-baseline_score>=.1 and all(v>0 for v in per_order_gain) and float(np.mean(prior_gains))>=.1 and all(r['summary']['p95_seconds']<5 for r in records)
        result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':'DECODER_SEARCH_PROMISING_RESEARCH_ONLY' if informative else 'STOP_G1_WORD_DECODER_SEARCH',
            'selection':selection,'development':records,'prior_comparisons':post,'informative_screen':informative,
            'summary':{'dev_greedy_normalized':baseline_score,'dev_selected_normalized':scores[str(selected)],'mean_prior_normalized_gain':float(np.mean(prior_gains))},
            'greedy_state_parity':True,'weights_unchanged':True,'new_training':False,'deployed':False,'normal_chat_or_vision_proven':False,'retention_confirmed':False,
            'resources':monitor.report(),'cap':cap,'limits':['Diagnosticvirtualclock not compatiblewith live monotonicclock yet.','Search cannot create missing language/visualknowledge or demonstrate new synapselearning.',
                'Sharedsourceinitialization, known8intent styles, alreadyinspectedcases; no independentnormalchat or researchgatepass.']}
        (run/'report.md').write_text('# Bounded generic neural-token search\n\nWeights shared/unchanged, no target/template access, no deployment or learninggain claim.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-bounded-beam-latest.json',result)
        print(json.dumps({'run':str(run),'decision':result['decision'],'selection':selection,'summary':result['summary']}))


if __name__=='__main__':main()
