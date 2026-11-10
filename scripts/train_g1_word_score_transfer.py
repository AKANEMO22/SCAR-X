"""One locked local rule, threefreshmodels, next-word teacherprefix pilot only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,itertools
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from scarx.evaluation.score_word import ScoreWord
from scarx.neural.language import tokenize
from scarx.learning.dialogue import normalized
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from generate_g1_state_paraphrases import INTENTS

SEEDS=(27103,27109,27127);CONDITIONS=('frozen','full','shuffled')


def evaluate(payload,cases,seed,run,name):
    model=ScoreWord.loads(payload);outputs=[]
    for index,case in enumerate(cases):
        began=time.perf_counter();ticks=model.evaluation_ticks;reply=model.generate(case['prompt'],seed+10300000+index)
        measured_ticks=model.evaluation_ticks-ticks;seconds=time.perf_counter()-began
        teacher=model.teacher_metrics(case['prompt'],case['target'],seed+10300000+index)
        row={'id':case['id'],'prompt':case['prompt'],'target':case['target'],'generated':reply['text'],
            'strict_style_exact':tokenize(reply['text'])==tokenize(case['target']),
            'normalized_style_exact':normalized(reply['text'])==normalized(case['target']),
            'stopped':reply['stopped'],'teacher':teacher,'free_generation_ticks':measured_ticks,'seconds':seconds}
        outputs.append(row)
    for row in outputs[:8]:append(run,'answer',{'scope':name+' · experimental48ALIF, not serving',
        'case_id':row['id'],'prompt':row['prompt'],'target':row['target'],'generated':row['generated'],
        'input_scope':'MC8prefix replay from neural-generatedtokens, no target/template in freegeneration. SourceVocab closed8stylefamilies; no normalchat/visionclaim.',
        'qwen':None,'active':48,'allocated':48,'neural_ticks':row['free_generation_ticks'],'seconds':row['seconds'],
        'checkpoint':'public experimental artifact, never imported into main'})
    summary={k:float(np.mean([r[k] for r in outputs])) for k in ('strict_style_exact','normalized_style_exact','stopped')}
    summary.update(teacher_token_accuracy=float(np.mean([r['teacher']['token_accuracy'] for r in outputs])),
        teacher_nll=float(np.mean([r['teacher']['nll'] for r in outputs])),first_token_accuracy=float(np.mean([r['teacher']['first_token_correct'] for r in outputs])))
    return summary,outputs


def trial(job):
    seed,condition,runpath,artpath,train,dev,prior,vocabulary=job[:8];run=Path(runpath);artifact=Path(artpath)
    encoding=job[8] if len(job)>8 else None;tag=job[9] if len(job)>9 else ''
    cap=apply_memory_limit()
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.perf_counter();model=ScoreWord(vocabulary,seed,condition,encoding);model.calibrate([r['prompt'] for r in train])
    original={n:getattr(model.core,n).copy() for n in ('w_in','w_rec')};before={n:getattr(model.core,n).copy() for n in ('w_in','w_rec','w_out')}
    calibration=model.core.head_mean.copy();rng=np.random.default_rng(seed+10200000);groups=[[r for r in train if r['intent']==intent] for intent in INTENTS]
    losses=[];token_updates=0
    with ResourceMonitor(interval=.2) as monitor:
        for index in range(1024):
            pool=groups[int(rng.integers(8))];row=pool[int(rng.integers(len(pool)))];result=model.learn(row['prompt'],row['target'])
            losses.append(result['loss']);token_updates+=result['tokens']
            if (index+1)%256==0:
                delta={n:float(np.linalg.norm(getattr(model.core,n)-v)) for n,v in before.items()}
                append(run,'learning',{'scope':f'Wordscoretransfer · {tag+" / " if tag else ""}seed{seed}/{condition} · not serving',
                    'prompt':f'Lockedlocalrule · responses{index-254}–{index+1}','passes':1,'sample_count':256,
                    'loss_mean':float(np.mean(losses[-256:])),'losses':[float(np.mean(losses[-256:]))],
                    'update_steps':model.core.updates,'update_unit':'word target cumulative','hidden_update_steps':model.core.hidden_updates,
                    'weight_delta':delta,'attempted_weight_delta':delta,'changed_synapses':{n:int(np.count_nonzero(getattr(model.core,n)!=v)) for n,v in before.items()},
                    'accepted':True,'rolled_back':False,'guard_checked':False,'reason':'Publicexperimentalweights only; lockedscoreeligibility×scalarreturn, no networkbackward. Qualitypending.'})
                before={n:getattr(model.core,n).copy() for n in before};monitor.check()
                if any(not np.isfinite(a).all() for a in model.core.arrays().values()):raise ValueError('Nonfinite wordscorecore')
                if time.perf_counter()-began>240:raise TimeoutError('Registered240s wordtransfer trial limit')
        train_seconds=time.perf_counter()-began;path=artifact/f'{seed}-{condition}.zip';path.write_bytes(model.dumps())
        clone=ScoreWord.loads(path.read_bytes());assert clone.dumps()==model.dumps()
        # Exact future learning only on discardedclones, not selected artifact.
        second=ScoreWord.loads(model.dumps());a=clone.learn(train[0]['prompt'],train[0]['target']);b=second.learn(train[0]['prompt'],train[0]['target']);assert a==b and clone.dumps()==second.dumps()
        np.testing.assert_array_equal(calibration,model.core.head_mean)
        if condition=='frozen':
            for name,value in original.items():np.testing.assert_array_equal(value,getattr(model.core,name))
        diagnostics=[]
        for intent in INTENTS:diagnostics += [r for r in train if r['intent']==intent][:2]
        summaries={};outputs={}
        for name,cases in (('train16',diagnostics),('development',dev),('prior',prior)):
            summaries[name],outputs[name]=evaluate(model.dumps(),cases,seed,run,f'{tag+"/" if tag else ""}seed{seed}/{condition}/{name}')
        record={'seed':seed,'condition':condition,'summary':summaries,'responses':1024,'word_target_updates':model.core.updates,
            'hidden_updates':model.core.hidden_updates,'training_ticks':model.core.ticks,'array_bytes':model.core.memory_bytes(),
            'hidden_delta_in':float(np.linalg.norm(model.core.w_in-original['w_in'])),'hidden_delta_rec':float(np.linalg.norm(model.core.w_rec-original['w_rec'])),
            'mean_training_loss':float(np.mean(losses)),'train_seconds':train_seconds,'seconds':time.perf_counter()-began,
            'checkpoint':path.relative_to(root).as_posix(),'checkpoint_exact':True,'continuation_exact':True,'resources':monitor.report(),'cap':cap}
        neural={'computed':48,'allocated':48,'tick':model.core.ticks,'spikes':model.core.spikes.astype(int).tolist(),
            'voltage':np.round(model.core.voltage,4).tolist(),'sample_kind':'lastactualtrainingtrajectory; wordadapter localupdates'}
        if tag:record['input_encoding']=tag
        write_json(run/f'{tag+"-" if tag else ""}{seed}-{condition}.json',{'summary':record,'outputs':outputs,'neural_sample':neural});return record,neural


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    info=json.loads((root/'artifacts/g1-curriculum-reviewed.json').read_text(encoding='utf-8'));data=json.loads((root/info['data']).read_text(encoding='utf-8'))['items']
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'));train=[]
    for intent,family in zip(INTENTS,corpus['families']):train += [{'id':f'known-{intent}-{j}','intent':intent,'prompt':p,'target':family['answer']} for j,p in enumerate(family['train'])]
    train += [r for r in data if r['split']=='train'];dev=[r for r in data if r['split']=='development']
    prior_info=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'));prior=json.loads((root/prior_info['data']).read_text(encoding='utf-8'))['items']
    vocab=['<eos>','<bos>']+sorted({t for r in train for t in tokenize(r['target'])})
    assert all(all(t in vocab for t in tokenize(r['target'])) for r in dev+prior)
    run=new_run(root,'g1-word-score-transfer-v1',{'protocol':'experiments/g1-word-score-transfer-v1.md',
        'display_title':'Học thật · lockedscore → dự đoán từ · fresh48ALIF',
        'model_scope':'Publicexperimentalcore only, frozen/full/shuffled matched; no mainmigration or visionclaim',
        'model_seeds':SEEDS,'conditions':CONDITIONS,'responses':1024,'hidden':48,'vocabulary':len(vocab),'new_hidden_rule_tuning':False,'workers':2},
        data_description='Public117AGYtrain+64authored,31dev+48prior; no privateweights/feedback, wordtargets never in same-prefixinput',data_files=[info['data'],'data/dialogue/generation1.json',prior_info['data']])
    artifact=root/'artifacts/word-score-transfer'/run.name;artifact.mkdir(parents=True);records=[];pending={}
    jobs=iter([(s,c,str(run),str(artifact),train,dev,prior,vocab) for s,c in itertools.product(SEEDS,CONDITIONS)])
    with ResourceMonitor(interval=.1) as monitor:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for _ in range(2):job=next(jobs);pending[pool.submit(trial,job)]=job
            while pending:
                ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED);monitor.check()
                if time.perf_counter()-began>1500:raise TimeoutError('Registered1500s transferround limit')
                for future in ready:
                    pending.pop(future);row,neural=future.result();records.append(row)
                    progress(run,f'seed{row["seed"]} · {row["condition"]}',len(records),9,metrics={'dev_normalized':row['summary']['development']['normalized_style_exact'],
                        'prior_normalized':row['summary']['prior']['normalized_style_exact'],'hidden_delta_in':row['hidden_delta_in'],'train_seconds':row['train_seconds']},neural=neural)
                    write_json(run/'trials.json',records);print(json.dumps({k:row[k] for k in ('seed','condition','summary','train_seconds')}),flush=True)
                    try:job=next(jobs);pending[pool.submit(trial,job)]=job
                    except StopIteration:pass
        summary={c:{split:{metric:float(np.mean([r['summary'][split][metric] for r in records if r['condition']==c])) for metric in ('strict_style_exact','normalized_style_exact','teacher_token_accuracy','teacher_nll','first_token_accuracy')} for split in ('train16','development','prior')} for c in CONDITIONS}
        gain=[next(r['summary']['development']['normalized_style_exact'] for r in records if r['seed']==s and r['condition']=='full')-next(r['summary']['development']['normalized_style_exact'] for r in records if r['seed']==s and r['condition']=='frozen') for s in SEEDS]
        prior_gain=[next(r['summary']['prior']['normalized_style_exact'] for r in records if r['seed']==s and r['condition']=='full')-next(r['summary']['prior']['normalized_style_exact'] for r in records if r['seed']==s and r['condition']=='frozen') for s in SEEDS]
        informative=summary['full']['development']['normalized_style_exact']>=.4 and np.mean(gain)>=.1 and all(v>0 for v in gain) and np.mean(prior_gain)>=.1
        result={'run':run.relative_to(root).as_posix(),'decision':'WORD_SCORE_FRESH_VALIDATION_REQUIRED' if informative else 'STOP_WORD_SCORE_TRANSFER_V1',
            'summary':summary,'paired_dev_gain':interval(gain),'paired_prior_gain':interval(prior_gain),'informative_screen':informative,
            'trials':len(records),'resources':monitor.report(),'cap':cap,'deployed':False,'normal_chat_or_vision_proven':False,'retention_confirmed':False,
            'new_hidden_rule_tuning':False,'limits':['Only3freshmodels and8knownstylefamilies; pilot not formalP1/P2or normalchat.',
                'Canonicalstochasticprefixreplay differs from runtimecontinuousstate; no migration/visiontraining.','Lockedruletransfer, no hyperparameter/gridextension or pretrainedweights.']}
        (run/'report.md').write_text('# Lockedscore rule transfer to wordprediction\n\nActual localneuraltraining, freshmodels, matchedfrozen/shuffledcontrols; no servingmigration or generalcapabilityclaim.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-word-score-transfer-latest.json',result)
        print(json.dumps({'run':str(run),'decision':result['decision'],'summary':summary,'paired_dev_gain':result['paired_dev_gain']}))


if __name__=='__main__':main()
