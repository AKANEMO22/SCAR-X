"""Preregistered actual label-conditioned Hebbian observations, no gradients."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,itertools,hashlib
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from train_g1_word_score_inputs import source_data
from generate_g1_state_paraphrases import INTENTS
from scarx.evaluation.observed_association import ObservedAssociation,CONDITIONS
from scarx.evaluation.fixed_word_codes import fit
from scarx.evaluation.dialogue_probe import ridge
from scarx.evaluation.phase21 import interval
from scarx.neural.language import tokenize
from scarx.learning.dialogue import normalized
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append,activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

SEEDS=(31013,31019,31033)
METRICS=('strict_style_exact','normalized_style_exact','teacher_token_accuracy','teacher_nll','first_token_accuracy','stopped')


def evaluate(payload,cases,run,seed,condition,split):
    model=ObservedAssociation.loads(payload);outputs=[]
    for row in cases:
        start=time.perf_counter();ticks=model.ticks;reply=model.generate(row['prompt']);actual_ticks=model.ticks-ticks;seconds=time.perf_counter()-start
        teacher=model.teacher_metrics(row['prompt'],row['target'])
        outputs.append({'id':row['id'],'prompt':row['prompt'],'target':row['target'],'generated':reply['text'],
            'strict_style_exact':tokenize(reply['text'])==tokenize(row['target']),'normalized_style_exact':normalized(reply['text'])==normalized(row['target']),
            'teacher':teacher,'stopped':reply['stopped'],'neural_ticks':actual_ticks,'seconds':seconds})
    for r in outputs[:8]:append(run,'answer',{'scope':f'OBSERVED-ASSOCIATION EXPERIMENT · seed{seed}/{condition}/{split} · NOT MAIN',
        'case_id':r['id'],'prompt':r['prompt'],'target':r['target'],'generated':r['generated'],'neural_ticks':r['neural_ticks'],
        'active':48,'allocated':48,'seconds':r['seconds'],'checkpoint':'Public experimentalassociation checkpoint, not serving',
        'input_scope':'Actual deterministicALIFgeneration: prompt + neuralpreviouswords only. NO observationphase, goldanswer, template or retrieval at inference. Training used explicit observedanswers, labelconditioned.'})
    summary={k:float(np.mean([r[k] for r in outputs])) for k in ('strict_style_exact','normalized_style_exact','stopped')}
    summary.update(teacher_token_accuracy=float(np.mean([r['teacher']['token_accuracy'] for r in outputs])),
        teacher_nll=float(np.mean([r['teacher']['nll'] for r in outputs])),first_token_accuracy=float(np.mean([r['teacher']['first_token_correct'] for r in outputs])))
    return summary,outputs


def trial(job):
    seed,condition,runpath,artpath,train,dev,prior,vocab,encoding=job;run=Path(runpath);artifact=Path(artpath)
    cap=apply_memory_limit()
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.perf_counter();model=ObservedAssociation(vocab,seed,condition,encoding)
    initial=ObservedAssociation.loads(model.dumps());original={n:getattr(model.core,n).copy() for n in ('w_in','w_rec','w_out')};previous={n:v.copy() for n,v in original.items()}
    groups=[[r for r in train if r['intent']==i] for i in INTENTS];answers=[pool[0]['target'] for pool in groups];losses=[];groups_seen=np.zeros(8,np.int64)
    with ResourceMonitor(interval=.1) as monitor:
        for index in range(1024):
            group=int(model.sampler_rng.integers(8));pool=groups[group];row=pool[int(model.sampler_rng.integers(len(pool)))];groups_seen[group]+=1
            observed=answers[(group+1)%8] if condition=='cross_family' else row['target']
            result=model.learn(row['prompt'],row['target'],observed);losses.append(result['loss'])
            if (index+1)%256==0:
                delta={n:float(np.linalg.norm(getattr(model.core,n)-v)) for n,v in previous.items()}
                append(run,'learning',{'scope':f'Observedassociation · seed{seed}/{condition} · labelconditioned · NOT MAIN',
                    'prompt':f'Observation+prediction experiences{index-254}–{index+1}','passes':1,'sample_count':256,
                    'loss_mean':float(np.mean(losses[-256:])),'losses':[float(np.mean(losses[-256:]))],
                    'update_steps':model.core.updates,'update_unit':'output-local word target cumulative','hidden_update_steps':model.hidden_transactions,
                    'weight_delta':delta,'attempted_weight_delta':delta,'changed_synapses':{n:int(np.count_nonzero(getattr(model.core,n)!=v)) for n,v in previous.items()},
                    'accepted':True,'rolled_back':False,'guard_checked':False,
                    'reason':'EXPERIMENT: observedanswer onlytrainingphase; contextsynapses use pretrace×postactivityminusmean. NO gradients/rewardscore, no liveguard/weights change.'})
                previous={n:getattr(model.core,n).copy() for n in previous};monitor.check()
                if time.perf_counter()-began>120:raise TimeoutError('Registered120s observedassociation trialcap')
        training_seconds=time.perf_counter()-began;path=artifact/f'{seed}-{condition}.association.zip';payload=model.dumps();path.write_bytes(payload)
        if not all(np.isfinite(a).all() for a in model.core.arrays().values()):raise ValueError('Nonfinite associationcore')
        np.testing.assert_array_equal(original['w_rec'],model.core.w_rec)
        noncontext=np.r_[np.arange(model.context_start),np.arange(model.context_end,model.core.config.inputs)]
        np.testing.assert_array_equal(original['w_in'][:,noncontext],model.core.w_in[:,noncontext])
        if condition in ('frozen','erase_trace'):np.testing.assert_array_equal(original['w_in'],model.core.w_in)
        clone=ObservedAssociation.loads(payload);second=ObservedAssociation.loads(payload)
        assert clone.dumps()==payload
        first=train[0];observed=answers[1] if condition=='cross_family' else first['target']
        assert clone.learn(first['prompt'],first['target'],observed)==second.learn(first['prompt'],first['target'],observed) and clone.dumps()==second.dumps()
        np.testing.assert_array_equal(model.core.w_in,ObservedAssociation.loads(payload).core.w_in)
        diagnostics=[]
        for intent in INTENTS:diagnostics += [r for r in train if r['intent']==intent][:2]
        summaries={};outputs={}
        for split,cases in [('train16',diagnostics),('development',dev),('prior',prior)]:summaries[split],outputs[split]=evaluate(payload,cases,run,seed,condition,split)
        # Refit diagnostic labelhead, never used by the generative model.
        trained=ObservedAssociation.loads(payload)
        y=np.array([INTENTS.index(r['intent']) for r in train]);dy=np.array([INTENTS.index(r['intent']) for r in dev])
        tf=np.stack([trained.prompt_features(r['prompt']) for r in train]);df=np.stack([trained.prompt_features(r['prompt']) for r in dev])
        initial_fit=np.stack([initial.prompt_features(r['prompt']) for r in train]);initial_dev=np.stack([initial.prompt_features(r['prompt']) for r in dev])
        probe=ridge(tf,y,df,dy);initial_probe=ridge(initial_fit,y,initial_dev,dy)
        np.savez_compressed(artifact/f'{seed}-{condition}.prompt-features.npz',train=tf,development=df,initial_train=initial_fit,initial_development=initial_dev,train_labels=y,development_labels=dy)
        neural={'computed':48,'allocated':48,'tick':model.ticks,'spikes':model.core.spikes.astype(int).tolist(),
            'voltage':np.round(model.core.voltage,4).tolist(),'sample_kind':'actual finaltrainingpredictionphase; covarianceupdates happen only separate observationphase'}
        record={'seed':seed,'condition':condition,'summary':summaries,'experiences':model.experiences,'observations':model.observations,
            'word_updates':model.core.updates,'hidden_transactions':model.hidden_transactions,'observation_opportunities':model.slot_clock,
            'actual_ticks':model.ticks,'array_bytes':model.memory_bytes(),'hidden_input_delta':float(np.linalg.norm(model.core.w_in-original['w_in'])),
            'recurrent_delta':0.,'input_block_locality':True,'checkpoint_exact':True,'continuation_exact':True,
            'probe':probe['accuracy'],'initial_probe':initial_probe['accuracy'],'probe_gain':probe['accuracy']-initial_probe['accuracy'],
            'group_counts':groups_seen.tolist(),'train_seconds':training_seconds,'seconds':time.perf_counter()-began,
            'mean_training_loss':float(np.mean(losses)),'checkpoint':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(payload).hexdigest(),
            'resources':monitor.report(),'cap':cap}
        monitor.check()
        if time.perf_counter()-began>120:raise TimeoutError('Registered120s associationtrial including eval')
        write_json(run/f'{seed}-{condition}.json',{'record':record,'outputs':outputs,'probe_predictions':probe['predictions'],'neural_sample':neural})
        return record,neural


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();train,dev,prior,vocab,sources=source_data()
    encoding=fit([r['prompt'] for r in train],'known-digest128-v1')
    assert max(len(tokenize(r['target']))+1 for r in train+dev+prior)<=7
    run=new_run(root,'g1-observed-association-v1',{'protocol':'experiments/g1-observed-association-v1.md',
        'display_title':'ALIF học từ trải nghiệm đãdạy · covarianceassociation',
        'model_scope':'Experimental labelconditioned observedanswer phase; inference GOLD-FREE. No backward/Transformer/mainmigration.',
        'model_seeds':SEEDS,'conditions':CONDITIONS,'experiences':1024,'hidden':48,'context_channels':128,'workers':2,
        'new_rule_family':'observed pre/post covariance, NOT reward-score or fixedtargetrate'},
        data_description='Actual181train(117AGY+64authored),31dev/prior48 inspected; observedlabels explicittraining only. No private model/data.',data_files=sources)
    artifact=root/'artifacts/observed-association'/run.name;artifact.mkdir(parents=True)
    write_json(run/'encoding.json',encoding);(run/'report.md').write_text('# Observed-response association\n\nActualtraining in progress; separate observation/prediction phases, no servingchange. Outcome pending.\n',encoding='utf-8')
    activity(run,'Bắt đầu12lượt học liên kết quan sát đã đăng ký','Ba freshseeds×paired/frozen/crossfamily/erase-trace; cùngALIF48,7observation-slots và1024experiences. Chỉinputcontextsynapses học; currenttargetkhông vào inference/probe.',status='started',evidence={'trials':12,'new_hidden_global_gradient':False,'main_core_changed':False})
    records=[];pending={};jobs=iter([(s,c,str(run),str(artifact),train,dev,prior,vocab,encoding) for s,c in itertools.product(SEEDS,CONDITIONS)])
    with ResourceMonitor(interval=.1) as monitor:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for _ in range(2):job=next(jobs);pending[pool.submit(trial,job)]=job
            while pending:
                ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED);monitor.check()
                if time.perf_counter()-began>1800:raise TimeoutError('Registered1800s associationround cap')
                for future in ready:
                    pending.pop(future);row,neural=future.result();records.append(row)
                    progress(run,f'seed{row["seed"]} · {row["condition"]}',len(records),12,metrics={'dev_normalized':row['summary']['development']['normalized_style_exact'],'prior_normalized':row['summary']['prior']['normalized_style_exact'],'probe_gain':row['probe_gain'],'hidden_delta':row['hidden_input_delta'],'train_seconds':row['train_seconds']},neural=neural)
                    write_json(run/'trials.json',records);print(json.dumps({k:row[k] for k in ('seed','condition','summary','probe','probe_gain','train_seconds')}),flush=True)
                    try:job=next(jobs);pending[pool.submit(trial,job)]=job
                    except StopIteration:pass
        def row(s,c):return next(r for r in records if r['seed']==s and r['condition']==c)
        erased_exact=[]
        for s in SEEDS:
            a=ObservedAssociation.loads((root/row(s,'frozen')['checkpoint']).read_bytes());b=ObservedAssociation.loads((root/row(s,'erase_trace')['checkpoint']).read_bytes())
            for name,value in a.core.arrays().items():np.testing.assert_array_equal(value,b.core.arrays()[name])
            assert row(s,'frozen')['summary']==row(s,'erase_trace')['summary'];erased_exact.append(True)
        summary={c:{split:{metric:float(np.mean([r['summary'][split][metric] for r in records if r['condition']==c])) for metric in METRICS} for split in ('train16','development','prior')} for c in CONDITIONS}
        dev_gain=[row(s,'paired')['summary']['development']['normalized_style_exact']-row(s,'frozen')['summary']['development']['normalized_style_exact'] for s in SEEDS]
        prior_gain=[row(s,'paired')['summary']['prior']['normalized_style_exact']-row(s,'frozen')['summary']['prior']['normalized_style_exact'] for s in SEEDS]
        unpaired_gain=[row(s,'paired')['summary']['development']['normalized_style_exact']-row(s,'cross_family')['summary']['development']['normalized_style_exact'] for s in SEEDS]
        probe_gain=[row(s,'paired')['probe']-row(s,'frozen')['probe'] for s in SEEDS]
        gates={'paired_dev40':summary['paired']['development']['normalized_style_exact']>=.4,
            'paired_vs_frozen_dev10_all_positive':bool(np.mean(dev_gain)>=.1 and all(v>0 for v in dev_gain)),
            'paired_vs_frozen_prior10':bool(np.mean(prior_gain)>=.1),'paired_vs_crossfamily_dev5_all_positive':bool(np.mean(unpaired_gain)>=.05 and all(v>0 for v in unpaired_gain)),
            'prompt_only_probe_gain5_all_positive':bool(np.mean(probe_gain)>=.05 and all(v>0 for v in probe_gain)),
            'erased_frozen_exact_allseeds':all(erased_exact),'all_checkpoint_continuations_exact':all(r['checkpoint_exact'] and r['continuation_exact'] for r in records)}
        informative=all(gates.values());result={'run':run.relative_to(root).as_posix(),'decision':'OBSERVED_ASSOCIATION_FRESH_VALIDATION_REQUIRED' if informative else 'STOP_OBSERVED_ASSOCIATION_V1',
            'summary':summary,'gates':gates,'paired_dev_gain':interval(dev_gain),'paired_prior_gain':interval(prior_gain),
            'paired_crossfamily_dev_gain':interval(unpaired_gain),'paired_prompt_probe_gain':interval(probe_gain),
            'probe_by_condition':{c:float(np.mean([r['probe'] for r in records if r['condition']==c])) for c in CONDITIONS},
            'trials':len(records),'resources':monitor.report(),'cap':cap,'deployed':False,'normal_chat_or_vision_proven':False,'retention_confirmed':False,
            'new_hidden_backward':False,'label_conditioned_learning':True,'inference_gold_input':False,'historical_stopped_families_preserved':True,
            'limits':['Only3freshseeds,8knownstyleanswers and inspecteddev/prior; not originalP1confirmation or normalchat/vision.',
                'Teacheranswer is privilegedtraininginput in separateobservationphase, not unsupervised or verifiedtruth.',
                'Onlyinputcontextsynapseslearn. DeterministicG1forward/rawhead differs from stochasticScoreWord; compare own matchedcontrols only.',
                'One registeredscreen, no autoLR/encoder/epoch/round2/modelscale tune or mainimport.']}
        (run/'report.md').write_text('# Observed-response covariance association\n\nActual label-conditioned localpre/post learning on experimentalALIF48. Nocurrenttarget at inference/probe. Fixedcontrols/gates; no servingimport or broadqualityclaim.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-observed-association-latest.json',result)
        activity(run,'Hoàn tất12lượt observedassociation',result['decision']+'; giữ mọi kết quả và ngưỡng. Không importcandidate hoặc gọi labelconditionedlearning là unsupervised.',evidence={'gates':gates,'paired_prompt_probe_gain':result['paired_prompt_probe_gain'],'new_main_weights':False})
        print(json.dumps({'run':result['run'],'decision':result['decision'],'gates':gates,'summary':summary},ensure_ascii=False))


if __name__=='__main__':main()
