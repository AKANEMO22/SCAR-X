"""Registered38readonly audit: no neuraltraining, optimizer, Torch or policyfit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from direct_words_common import source_data
from scarx.evaluation.word_training_audit import load_reference,protected,free_forward,prefix_forward,score_prefix,potential_direction,credit_summary,corpus_structure
from scarx.evaluation.direct_word_data import FAMILIES,TARGETS
from scarx.evaluation.phase21 import interval
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity,append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

SOURCE='runs/20261110T062226Z-direct-word-bridge-v1-a6a908'


def aggregate(rows):
    fields=('free_exact','tf_first','tf_later','wrong_first','wrong_later','zero_first','zero_later','tf_later_content','tf_later_punctuation',
        'tf_same_free_later','free_prefix_same_later','tf_first_nll','tf_later_nll')
    def macro(key):
        values=[]
        for family in FAMILIES:
            group=[r[key] for r in rows if r['family']==family and r.get(key) is not None]
            if group:values.append(float(np.mean(group)))
        return float(np.mean(values)) if values else None
    result={k:macro(k) for k in fields}
    result.update(cases=len(rows),gold_later_positions=sum(r['gold_later_positions'] for r in rows),free_later_positions_reached=sum(r['free_later_positions_reached'] for r in rows),
        free_gold_later_positions_censored=sum(r['free_gold_later_positions_censored'] for r in rows))
    result['wrong_first_drop']=result['tf_first']-result['wrong_first'];result['wrong_later_drop']=result['tf_later']-result['wrong_later']
    result['zero_first_drop']=result['tf_first']-result['zero_first'];result['zero_later_drop']=result['tf_later']-result['zero_later']
    result['matched_later_exposure_gap']=result['tf_same_free_later']-result['free_prefix_same_later']
    credits=[r['credit'] for r in rows if r.get('credit')]
    if credits:
        valid=lambda k:[r[k] for r in credits if r[k] is not None]
        result['potential_credit']={k:float(np.median(valid(k))) if valid(k) else None for k in ('first_norm','later_sum_norm','first_later_cosine','first_all_cosine')}
        result['potential_credit']['groups']={group:{k:float(np.median([c['groups'][group][k] for c in credits])) for k in ('first_norm','later_sum_norm')} for group in credits[0]['groups']}
    else:result['potential_credit']=None
    result['flags']={'POTENTIAL_PREFIX_RELIANCE':result['tf_later']>=.9 and result['wrong_later_drop']<=.1 and result['wrong_first_drop']>=.2,
        'EXPOSURE_SENSITIVITY':result['matched_later_exposure_gap']>=.1,
        'DIRECTION_CONFLICT':(result['potential_credit']['first_later_cosine']<0 if result['potential_credit'] and result['potential_credit']['first_later_cosine'] is not None else None)}
    return result


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();receipt,train,dev,vocab,files=source_data();cases=train+dev;source=json.loads((root/SOURCE/'result.json').read_text(encoding='utf-8'))
    assert source['decision']=='STOP_DIRECT_WORD_BRIDGE_V1' and len(source['trials'])==15
    trials=sorted(source['trials'],key=lambda r:(r['seed'],r['method']))
    for r in trials:files.extend([r['checkpoint'],SOURCE+f"/word-{r['seed']}-{r['method']}.json"])
    run=new_run(root,'word-training-objective-audit-v1',{'protocol':'experiments/word-training-objective-audit-v1.md','display_title':'READ ONLY38 · hiểu câu hỏi hay nối mẫu đáp án?',
        'model_scope':'15existing36checkpoints, FREE vs explicitGOLDPREFIX/wrong/zeroquestion. No training/optimizer/headfit/Transformer/mainchange.',
        'source_models':15,'cases_per_source':128,'neural_training':False,'neural_update_steps':0,'source_run':SOURCE},
        data_description='PinnedreviewedactualAGY128questions/8ownedstyles, no newlabels/teacher/Qwen/privateinputs. Existing15modelsourceweights read only.',data_files=files+[SOURCE+'/result.json'])
    (run/'report.md').write_text('# Readonlywordobjectiveaudit38\n\nDiagnostics inprogress, no neuraltraining. Teacherprefix metrics mustnotbe called freeanswers.\n',encoding='utf-8')
    write_json(run/'answer-source.json',{'source_run':SOURCE,'scope':'Archived36freewordoutputs, originalfiletimestamps, NOT teacherprefixauditanswers.'})
    activity(run,'Bắt đầu38audit cáchdạyALIF, khôngtraining','3tests passed. Đo15checkpoint36×128cases, FREE vs GOLDPREFIX vàwrong/zeroquestion; potentiallocaldirections chỉtính, khôngápdụng. Khôngmởlearner/encoder/mẫu mới.',status='started',evidence={'protocol':'experiments/word-training-objective-audit-v1.md','neural_updates':0,'source_models':15})
    records=[];table=[];structure=None;forward_calls=0
    with ResourceMonitor(interval=.1) as monitor:
        for src in trials:
            seed=src['seed'];method=src['method'];path=root/src['checkpoint'];payload=path.read_bytes();assert hashlib.sha256(payload).hexdigest()==src['sha256']
            m=load_reference(payload,method=='bptt_control');before=protected(m);initial_updates=m.core.updates;ticks=m.ticks
            if structure is None:structure=corpus_structure([(m.target_tokens(t),m.vocabulary) for t in TARGETS.values()])
            archived=json.loads((root/SOURCE/f'word-{seed}-{method}.json').read_text(encoding='utf-8'))['outputs'];saved={r['id']:r for split in ('train','development') for r in archived[split]};rows=[]
            for index,case in enumerate(cases,1):
                gold=m.target_tokens(case['target']);inference_started=time.perf_counter();free=free_forward(m,case['prompt']);inference_seconds=time.perf_counter()-inference_started;forward_calls+=1;old=saved[case['id']]
                assert free['text']==old['generated'] and free['stopped']==old['stopped']
                teacher=prefix_forward(m,case['prompt'],gold[:-1],measure_credit=method=='correct');forward_calls+=1;tf=score_prefix(teacher,gold)
                assert abs(float(np.mean([r['nll'] for r in tf]))-old['teacher']['nll'])<=1e-7 and float(np.mean([r['correct'] for r in tf]))==old['teacher']['token_accuracy']
                wrong=next(r for r in cases if r['target']!=case['target']);wr=score_prefix(prefix_forward(m,wrong['prompt'],gold[:-1]),gold);forward_calls+=1
                zr=score_prefix(prefix_forward(m,case['prompt'],gold[:-1],zero_question=True),gold);forward_calls+=1
                free_exact=free['tokens']==gold[:-1] and free['stopped'];teacher_all=all(r['correct'] for r in tf);assert teacher_all==free_exact
                reached=min(len(free['rows']),len(gold));tf_matched=[tf[i]['correct'] for i in range(1,reached)];free_matched=[free['rows'][i]['predicted']==gold[i] for i in range(1,reached)]
                content=[r['correct'] for r in tf[1:] if r['target']!=0 and any(c.isalnum() for c in m.vocabulary[r['target']])]
                punctuation=[r['correct'] for r in tf[1:] if r['target']==0 or not any(c.isalnum() for c in m.vocabulary[r['target']])]
                credit=credit_summary(m,[potential_direction(m,row,target) for row,target in zip(teacher,gold)]) if method=='correct' else None
                row={'id':case['id'],'split':case['split'],'family':case['family'],'prompt':case['prompt'],'target':case['target'],'free_generated':free['text'],'free_exact':free_exact,
                    'tf_first':int(tf[0]['correct']),'tf_later':float(np.mean([r['correct'] for r in tf[1:]])),
                    'wrong_first':int(wr[0]['correct']),'wrong_later':float(np.mean([r['correct'] for r in wr[1:]])),
                    'zero_first':int(zr[0]['correct']),'zero_later':float(np.mean([r['correct'] for r in zr[1:]])),
                    'tf_first_nll':tf[0]['nll'],'tf_later_nll':float(np.mean([r['nll'] for r in tf[1:]])),
                    'tf_later_content':float(np.mean(content)) if content else None,'tf_later_punctuation':float(np.mean(punctuation)) if punctuation else None,
                    'tf_same_free_later':float(np.mean(tf_matched)) if tf_matched else None,'free_prefix_same_later':float(np.mean(free_matched)) if free_matched else None,
                    'gold_later_positions':len(gold)-1,'free_later_positions_reached':reached-1,'free_gold_later_positions_censored':len(gold)-reached,
                    'wrong_question_id':wrong['id'],'wrong_question':wrong['prompt'],'credit':credit,
                    'teacherprefix_scope':'EXPLICITgoldprefixdiagnostic, NOT freeanswer/no learning','free_scope':'Targetfree actualneuralprefix, no proxy/lookup.'}
                assert protected(m)==before and m.core.updates==initial_updates;rows.append(row)
                if index in (1,65):append(run,'answer',{'scope':f'READ ONLY38 FREE · seed{seed}/{method}/{case["split"]} · NOT MAIN','control_only':method=='bptt_control',
                    'case_id':case['id'],'prompt':case['prompt'],'generated':free['text'],'target':case['target'],'neural_ticks':free['actual_ticks'],'seconds':inference_seconds,'active':48,'allocated':48,
                    'checkpoint':src['checkpoint'],'input_scope':'Actual FREEauditforward with previousNEURALwords only; teacherprefix/wrongquestionmetrics separate. Fixed36checkpoint, no newtraining.'})
                if index%32==0:
                    progress(run,f'READ ONLY38 {seed}/{method} {index}/128',len(records),15,metrics={'seed':seed,'method':method,'cases':index,'neural_updates':0,'seconds':time.perf_counter()-began},
                        neural={'computed':48,'allocated':48,'tick':free['rows'][-1]['tick'],'spikes':free['rows'][-1]['spikes'].astype(int).tolist(),'voltage':np.round(free['rows'][-1]['voltage'],4).tolist(),'sample_kind':'Actualreadonly38 FREEfinalwordtrajectory, not GOLDPREFIX or newlearning'});monitor.check()
                    if time.perf_counter()-began>300:raise TimeoutError('Registered38audit300s')
            summary={split:aggregate([r for r in rows if r['split']==split]) for split in ('train','development')}
            record={'seed':seed,'method':method,'summary':summary,'source_checkpoint':src['checkpoint'],'source_sha256':src['sha256'],'protected_weights_head_Adam_counters_rng_unchanged':True,
                'all128_freeoutputs_and_teachermetrics_reproduce_source':True,'teacher_all_argmax_equals_free_exact':True,'audit_forward_ticks':m.ticks-ticks,'neural_updates':0}
            write_json(run/f'{seed}-{method}.audit.json',{'record':record,'cases':rows});records.append(record)
            for split in ('train','development'):
                v=summary[split];table.append({'seed':seed,'method':method,'split':split,**{k:v[k] for k in ('free_exact','tf_first','tf_later','wrong_first','wrong_later','wrong_first_drop','wrong_later_drop','zero_first','zero_later','matched_later_exposure_gap','free_later_positions_reached','free_gold_later_positions_censored','flags')},
                    'potential_first_later_cosine':v['potential_credit']['first_later_cosine'] if v['potential_credit'] else None})
            progress(run,f'READ ONLY38 {seed}/{method} done, no updates',len(records),15,metrics={'neural_updates':0,'train_later':summary['train']['tf_later'],'dev_free_exact':summary['development']['free_exact']})
        assert forward_calls==15*128*4 and 'torch' not in sys.modules
        for src in trials:assert hashlib.sha256((root/src['checkpoint']).read_bytes()).hexdigest()==src['sha256']
        effects={}
        for method in ('correct','shuffled','zero','frozen','bptt_control'):
            effects[method]={}
            for split in ('train','development'):
                selected=[r['summary'][split] for r in records if r['method']==method]
                effects[method][split]={k:interval([r[k] for r in selected]) for k in ('wrong_first_drop','wrong_later_drop','matched_later_exposure_gap')}
        result={'run':run.relative_to(root).as_posix(),'source_run':SOURCE,'decision':'WORD_TRAINING_OBJECTIVE_DIAGNOSTIC_ONLY','corpus_structure':structure,
            'word_objective_rows':table,'paired_effects':effects,'records':records,'model_sources':15,'public_cases_per_source':128,'actual_forward_segments':forward_calls,
            'all_source_checkpoints_and_protected_fields_unchanged':True,'all_source_freeoutputs_and_teachermetrics_reproduced':True,'all_teacher_all_argmax_equals_free_exact':True,
            'neural_training':False,'neural_update_steps':0,'torch_imported':False,'deployed':False,'main_weights_changed':False,'normal_chat_or_vision_proven':False,
            'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-began,
            'limits':['Closed8styles/historicalcorpus,3pairedmodelseeds percondition, no generalchat/vision/lifelong or architecture-optimality claim.',
                'GOLDPREFIXexplicitdiagnostic, not freeanswer; wrong/zeroquestion are conflicts/OOD, flagsdescriptive not qualitygates.',
                'Potentialcredit FINALfixedcheckpoint only, NOT actualhistoricaltrainingupdates or prooflarge-normsynapses useful.',
                'Exposuregap onlysamepositions FREEreached; censoredlaterpositions explicitlyreported, not inventedpredictions.',
                'No newlearner/objective/epoch/LR/grid from thisaudit; any selectivecreditproposal needs separateprotocol. ExistingSTOPs/P1/P2 unchanged.']}
        (run/'report.md').write_text('# Wordtraining objective audit38\n\nNo neuraltraining; FREE/GOLDPREFIX distinct. Corpusstructure/potentialdirections not modelanswers or qualitysuccess.\n\n'+json.dumps({k:v for k,v in result.items() if k!='records'},indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/word-training-objective-audit.json',result)
        activity(run,'Hoàn tất38audit mụctiêuhọc, khôngtraining','15sources/128cases, FREE vs GOLDPREFIX/wrong-zeroquestion vàpotentialcredit đãđo. Source/RNG/counters/weights khôngđổi, no newlearner. Khôngtrộn tokenaccuracy với wholeanswer hoặc gọi templateproperty là neuralunderstanding.',evidence={'decision':result['decision'],'neural_updates':0,'forward_segments':forward_calls,'all_source_reproduced':True,'corpus_structure':structure})
        print(json.dumps({'run':result['run'],'corpus_structure':structure,'correct_summary':[r for r in records if r['method']=='correct'],'resources':result['resources']},ensure_ascii=False))


if __name__=='__main__':main()
