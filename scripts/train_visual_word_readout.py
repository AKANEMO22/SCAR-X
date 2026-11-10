"""Actual jointquestion/RGB neuralword learning, hiddenFROZEN, no backward."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,itertools,argparse,sqlite3
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from scarx.evaluation.visual_words import VisualWords
from scarx.evaluation.vision_fixtures import expected,image_uri
from scarx.neural.language import tokenize
from scarx.learning.dialogue import normalized
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval
from scarx.runtime.experiment_status import progress,RUN_ID,_process
from scarx.runtime.work_trace import append,activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

SEEDS=(33113,33119,33133);CONDITIONS=('full','no_image','no_question')
KINDS=('single_color','single_shape','object_count','left_color');LANGUAGES=('vi','en')


def macro(rows,key='exact'):
    groups=[[r[key] for r in rows if r['kind']==kind and r['language']==language] for kind,language in itertools.product(KINDS,LANGUAGES)]
    return float(np.mean([np.mean(g) for g in groups if g])) if any(groups) else None


def job_parts(job):
    if len(job) not in (9,10):raise ValueError('Expectedninebasearguments and optionalrecovery')
    return tuple(job[:9]),job[9] if len(job)==10 else None


def evaluate(payload,cases,images,scene_rows,run,seed,condition,split,ablation=None,teacher=True):
    model=VisualWords.loads(payload);rows=[]
    for case in cases:
        used=case['scene_id']
        if ablation=='swapped_image':used=next(s['id'] for s in scene_rows if (case['kind']=='object_count' or (len(s['objects'])==1 if case['kind'].startswith('single_') else len(s['objects'])>1)) and expected(s,case['kind'],case['language'])!=case['target'])
        started=time.perf_counter();ticks=model.ticks
        reply=model.generate(case['prompt'],images[used],ablation=ablation if ablation in ('blank_image','blank_question') else None)
        actual_ticks=model.ticks-ticks;elapsed=time.perf_counter()-started
        metrics=model.teacher_metrics(case['prompt'],images[case['scene_id']],case['target']) if teacher else None
        rows.append({'id':case['id'],'scene_id':case['scene_id'],'image_used':used,'prompt':case['prompt'],'target':case['target'],'generated':reply['text'],
            'kind':case['kind'],'language':case['language'],'novel_composition':case['novel_composition'],
            'exact':normalized(reply['text'])==normalized(case['target']),'strict_exact':tokenize(reply['text'])==tokenize(case['target']),
            'stopped':reply['stopped'],'teacher':metrics,'actual_ticks':actual_ticks,'seconds':elapsed})
    if ablation is None:
        for r in rows[:8]:append(run,'answer',{'scope':f'JOINT VISUAL WORD EXPERIMENT · seed{seed}/{condition}/{split} · HIDDEN FROZEN, NOT MAIN',
            'case_id':r['id'],'prompt':r['prompt'],'target':r['target'],'generated':r['generated'],
            'image_data_uri':image_uri(images[r['scene_id']]),'image_source':'Owned rendered geometricfixture '+r['scene_id'],
            'input_scope':'Actual SAME ALIF receives fixedquestionfeature andRGBpatches. Wordreadoutautoregressive; no kind/language/classmetadata/goldtarget at inference. Hiddenfrozen; workingmemory48neuralfeatures, not lookup.',
            'qwen':None,'qwen_missing_reason':'ExistingQwen4B text-only, no realvisionreference. No image-as-text substitute.',
            'neural_ticks':r['actual_ticks'],'seconds':r['seconds'],'active':48,'allocated':48,'checkpoint':'Experimental visualword artifact; never imported into main'})
    summary={'macro_exact':macro(rows),'micro_exact':float(np.mean([r['exact'] for r in rows])),
        'strict_macro_exact':macro(rows,'strict_exact'),'novel_composition_macro':macro([r for r in rows if r['novel_composition']]),
        'seen_composition_macro':macro([r for r in rows if not r['novel_composition']]),
        'by_kind':{k:macro([r for r in rows if r['kind']==k]) for k in KINDS},'by_language':{l:macro([r for r in rows if r['language']==l]) for l in LANGUAGES},
        'stopped_fraction':float(np.mean([r['stopped'] for r in rows])),'actual_ticks':sum(r['actual_ticks'] for r in rows)}
    if teacher:summary.update(teacher_token_accuracy=float(np.mean([r['teacher']['token_accuracy'] for r in rows])),teacher_nll=float(np.mean([r['teacher']['nll'] for r in rows])))
    return summary,rows


def trial(job):
    base,recovery=job_parts(job)
    seed,condition,runpath,artpath,train,dev,images,scene_rows,vocab=base;run=Path(runpath);artifact=Path(artpath)
    cap=apply_memory_limit()
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.perf_counter();model=VisualWords(vocab,seed,condition)
    initial_in=model.core.w_in.copy();initial_rec=model.core.w_rec.copy();previous=model.core.w_out.copy()
    calibration=[{'prompt':r['prompt'],'image':images[r['scene_id']]} for r in train];model.calibrate(calibration);initial_payload=model.dumps()
    pools={}
    for kind,language in itertools.product(KINDS,LANGUAGES):
        values=[r for r in train if r['kind']==kind and r['language']==language]
        pools[(kind,language)]={target:[r for r in values if r['target']==target] for target in sorted({r['target'] for r in values})}
    losses=[];counts={};ticks_before=model.ticks
    with ResourceMonitor(interval=.1) as monitor:
        recovered=recovery is not None;prior_active=recovery['prior_active_seconds'] if recovered else 0.
        if recovered:
            old_path=root/recovery['checkpoint'];old_payload=old_path.read_bytes();assert hashlib.sha256(old_payload).hexdigest()==recovery['sha256']
            old=VisualWords.loads(old_payload);assert (old.core.config.seed,old.condition,old.experiences)==(seed,condition,1536)
            for name in ('w_in','w_rec','indices'):np.testing.assert_array_equal(getattr(old.core,name),getattr(model.core,name))
            np.testing.assert_array_equal(old.head.mean,model.head.mean);np.testing.assert_array_equal(old.head.m2,model.head.m2)
            # Reconstruct ONLY sampler/counter bookkeeping; no parameterupdate.
            for _ in range(1536):
                kind=KINDS[int(model.sampler_rng.integers(4))];language=LANGUAGES[int(model.sampler_rng.integers(2))];group=pools[(kind,language)]
                target=sorted(group)[int(model.sampler_rng.integers(len(group)))];pool=group[target];case=pool[int(model.sampler_rng.integers(len(pool)))];key=kind+'/'+language+'/'+target;counts[key]=counts.get(key,0)+1
            assert model.sampler_rng.bit_generator.state==old.sampler_rng.bit_generator.state
            model=old
            with sqlite3.connect(root/'local/work-trace.sqlite') as db:
                prior=[json.loads(r[0]) for r in db.execute('SELECT payload FROM trace WHERE kind=?',('learning',))]
            prior=[r for r in prior if r.get('run_id')==recovery['source_run_id'] and r.get('scope')==f'JOINT VISUAL WORD · seed{seed}/{condition} · READOUT ONLY, not main']
            if len(prior)==6:losses=[r['loss_mean'] for r in prior] # equal256responseblocks, same fullmean
            activity(run,'Khôi phục checkpoint1536 đãhọc, không học lại',f'seed{seed}/{condition}: chỉre-evaluation; sourceSHA vàsampler/calibration khớp.',evidence={'source_checkpoint':recovery['checkpoint'],'source_sha256':recovery['sha256'],'training_updates_repeated':0})
        for index in range(0 if recovered else 1536):
            kind=KINDS[int(model.sampler_rng.integers(4))];language=LANGUAGES[int(model.sampler_rng.integers(2))]
            group=pools[(kind,language)];target=sorted(group)[int(model.sampler_rng.integers(len(group)))];pool=group[target];case=pool[int(model.sampler_rng.integers(len(pool)))]
            result=model.learn(case['prompt'],images[case['scene_id']],case['target']);losses.append(result['loss']);key=kind+'/'+language+'/'+target;counts[key]=counts.get(key,0)+1
            if (index+1)%256==0:
                delta=float(np.linalg.norm(model.core.w_out-previous))
                append(run,'learning',{'scope':f'JOINT VISUAL WORD · seed{seed}/{condition} · READOUT ONLY, not main',
                    'prompt':f'Image-question learning experiences{index-254}–{index+1}','passes':1,'sample_count':256,
                    'loss_mean':float(np.mean(losses[-256:])),'losses':[float(np.mean(losses[-256:]))],
                    'update_steps':model.core.updates,'update_unit':'output-local wordtargets cumulative','hidden_update_steps':0,
                    'weight_delta':{'w_in':0.,'w_rec':0.,'w_out':delta},'attempted_weight_delta':{'w_in':0.,'w_rec':0.,'w_out':delta},
                    'changed_synapses':{'w_in':0,'w_rec':0,'w_out':int(np.count_nonzero(model.core.w_out!=previous))},
                    'accepted':True,'rolled_back':False,'guard_checked':False,'reason':'Actual supervisedneuralwordheadupdates. HiddenFROZEN, no teachererrortransport/backward or mainweightschange; qualitypending.'})
                previous=model.core.w_out.copy();monitor.check()
                if time.perf_counter()-began>300:raise TimeoutError('Registered300s visualwordtrial cap')
        train_seconds=None if recovered else time.perf_counter()-began;training_ticks=model.ticks-ticks_before
        np.testing.assert_array_equal(initial_in,model.core.w_in);np.testing.assert_array_equal(initial_rec,model.core.w_rec)
        payload=model.dumps();path=artifact/f'{seed}-{condition}.visualword.zip';path.write_bytes(payload);copy=VisualWords.loads(payload);assert copy.dumps()==payload
        second=VisualWords.loads(payload);case=train[0]
        assert copy.learn(case['prompt'],images[case['scene_id']],case['target'])==second.learn(case['prompt'],images[case['scene_id']],case['target']) and copy.dumps()==second.dumps()
        summaries={};outputs={}
        for split,cases in [('train64',train[:64]),('development',dev)]:summaries[split],outputs[split]=evaluate(payload,cases,images,scene_rows,run,seed,condition,split)
        causal={}
        if condition=='full':
            summaries['initial_head'],outputs['initial_head']=evaluate(initial_payload,dev,images,scene_rows,run,seed,condition,'initial_head')
            for intervention in ('blank_image','blank_question','swapped_image'):
                causal[intervention],outputs[intervention]=evaluate(payload,dev,images,scene_rows,run,seed,condition,intervention,ablation=intervention,teacher=False)
        np.testing.assert_array_equal(model.core.w_in,initial_in);np.testing.assert_array_equal(model.core.w_rec,initial_rec)
        assert model.dumps()==payload # ALL evaluation uses discardedcopies
        row={'seed':seed,'condition':condition,'summary':summaries,'causal':causal,'experiences':1536,'word_updates':model.core.updates,
            'training_ticks':training_ticks,'calibration_count':model.head.count,'array_bytes':model.memory_bytes(),'vocabulary':len(vocab),
            'hidden_unchanged':True,'checkpoint_exact':True,'continuation_exact':True,'stratum_counts':counts,
            'checkpoint':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(payload).hexdigest(),
            'train_seconds':train_seconds,'seconds':time.perf_counter()-began,'mean_training_loss':float(np.mean(losses)) if losses else None,
            'recovered_without_retraining':recovered,'source_recovery':recovery,'effective_trial_active_seconds':time.perf_counter()-began+prior_active,
            'resources':monitor.report(),'cap':cap}
        monitor.check()
        if time.perf_counter()-began+prior_active>300:raise TimeoutError('Registered300s visualwordtrial incl originalattempt/evaluation')
        neural={'computed':48,'allocated':48,'tick':model.ticks,'spikes':model.core.spikes.astype(int).tolist(),
            'voltage':np.round(model.core.voltage,4).tolist(),'sample_kind':'Actual finaltrainingwordtrajectory; hiddenfrozen/readoutlocaldelta only'}
        write_json(run/f'{seed}-{condition}.json',{'record':row,'outputs':outputs,'neural_sample':neural});return row,neural


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--recover-aggregation-failure');args=parser.parse_args();recoveries={};failure=None;prior_active=0.
    if args.recover_aggregation_failure:
        identifier=args.recover_aggregation_failure
        if not RUN_ID.fullmatch(identifier):raise ValueError('InvalidsourcefailureID')
        old_run=root/'runs'/identifier;journal=json.loads((root/'local/experiment-status'/(identifier+'.json')).read_text(encoding='utf-8'))
        if journal['status']!='stopped_without_result' or _process(journal)[0]:raise ValueError('Original process not verifiedterminal')
        if (old_run/'trials.json').exists() or (old_run/'result.json').exists():raise ValueError('Unexpected prior completedqualityresult')
        source_manifest=json.loads((old_run/'manifest.json').read_text(encoding='utf-8'))
        for name in ('src/scarx/evaluation/visual_words.py','src/scarx/evaluation/readout_conditioning.py','src/scarx/neural/core.py','src/scarx/neural/input_events.py','experiments/visual-word-readout-v1.md'):
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=source_manifest['source_sha256'][name]:raise ValueError('Aggregationrepair changed neuralmath/protocol')
        original_active=journal['updated']-journal['started'];prior_active=original_active;assert 0<prior_active<300
        failed_wrappers=[]
        for journal_path in (root/'local/experiment-status').glob('*.json'):
            if journal_path.stat().st_size>196608:continue
            other=json.loads(journal_path.read_text(encoding='utf-8'));technical=other.get('config',{}).get('technical_recovery')
            if not isinstance(technical,dict) or technical.get('source_run')!='runs/'+identifier:continue
            other_run=root/'runs'/other['run_id']
            if other['status']!='stopped_without_result' or _process(other)[0] or (other_run/'result.json').exists() or list((root/'artifacts/visual-word-readout'/other['run_id']).glob('*.zip')):raise ValueError('Unexpected earlierrecovery progress; stop rather than retrain')
            seconds=other['updated']-other['started'];prior_active+=seconds
            failed_wrappers.append({'run':other_run.relative_to(root).as_posix(),'active_seconds':seconds,'neural_training_started':False})
        sources=root/'artifacts/visual-word-readout'/identifier
        for p in sources.glob('*.visualword.zip'):
            m=VisualWords.loads(p.read_bytes());key=(m.core.config.seed,m.condition)
            if key not in ((33113,'full'),(33113,'no_image')) or m.experiences!=1536:raise ValueError('Unexpectedpartialcheckpoint')
            recoveries[key]={'source_run_id':identifier,'checkpoint':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'prior_active_seconds':prior_active}
        if len(recoveries)!=2:raise ValueError('Expectedtwo intact1536checkpoints')
        failure={'source_run':'runs/'+identifier,'failure_kind':'empty_diagnostic_subset_macro_NaN_JSON_rejection','checkpoint_sources':list(recoveries.values()),
            'prior_active_program_seconds':prior_active,'original_active_seconds':original_active,'failed_recovery_wrappers':failed_wrappers,'unknown_prior_peak_memory':True,
            'repair_scope':'MissingmetricNone and recoveryargumentplumbing only; no algorithm/gate/data/hyperparameterchange. Recoveredmodels NOT retrained.'}
        failure_path=old_run/('failure-recovery-chain.json' if (old_run/'failure.json').exists() else 'failure.json')
        if not failure_path.exists():write_json(failure_path,failure)
    cap=apply_memory_limit();began=time.perf_counter()-prior_active;info=json.loads((root/'artifacts/visual-word-dataset-latest.json').read_text(encoding='utf-8'));path=(root/info['data']).resolve()
    if not path.is_relative_to(root/'data/candidates') or hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Dataset sourcechecksum')
    data=json.loads(path.read_text(encoding='utf-8'));train=[r for r in data['cases'] if r['split']=='train'];dev=[r for r in data['cases'] if r['split']=='development'];assert len(train)==len(dev)==864
    images={};files=[info['data']]
    for scene in data['scenes']:
        p=(path.parent/scene['file']).resolve()
        if not p.is_relative_to(path.parent) or hashlib.sha256(p.read_bytes()).hexdigest()!=scene['sha256']:raise ValueError('Ownedimage sourcechecksum')
        images[scene['id']]=p.read_bytes();files.append(p.relative_to(root).as_posix())
    vocab=['<eos>','<bos>']+sorted({t for r in train for t in tokenize(r['target'])})
    assert all(t in vocab for r in dev for t in tokenize(r['target']))
    run=new_run(root,'visual-word-readout-v1',{'protocol':'experiments/visual-word-readout-v1.md',
        'display_title':'JointRGB/câu hỏi → ALIF → từthật · readout-only',
        'model_scope':'Experimentalonecore48, hiddenFROZEN, no renderer/intentoracle at inference. No main/proposal30change.',
        'model_seeds':SEEDS,'conditions':CONDITIONS,'experiences':1536,'train_cases':864,'dev_cases':864,'hidden':48,'workers':2,
        'dataset':info['data'],'dataset_sha256':info['sha256'],'technical_recovery':failure},data_description='Owned192geometricimages/1728QA, actualAGYwording split, labels fromrenderer, no private/VLM/pretrainedweights.',data_files=files)
    artifact=root/'artifacts/visual-word-readout'/run.name;artifact.mkdir(parents=True)
    (run/'report.md').write_text('# Joint visual word localreadout\n\nActualtraining in progress; hiddenweights frozen, no renderer/goldtargets at freeinference, no normalvision/mainclaim.\n',encoding='utf-8')
    activity(run,'Bắt đầu9lượt visualword đãđăngký','3freshseeds×full/noimage/noquestion,1536actualexperiences; outputlocalwordlearning, no hiddenerrortransport/proposal30approval. Causalblank/swap vàheldoutcombinations khóa trước.',status='started',evidence={'trials':9,'hidden_learning':False,'vocabulary':len(vocab),'train_dev_cases':[864,864]})
    pending={};records=[];jobs=iter([(s,c,str(run),str(artifact),train,dev,images,[s for s in data['scenes'] if s['split']=='development'],vocab,recoveries.get((s,c))) for s,c in itertools.product(SEEDS,CONDITIONS)])
    with ResourceMonitor(interval=.1) as monitor:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for _ in range(2):job=next(jobs);pending[pool.submit(trial,job)]=job
            while pending:
                ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED);monitor.check()
                if time.perf_counter()-began>2400:raise TimeoutError('Registered2400s visualwordround cap')
                for future in ready:
                    pending.pop(future);row,neural=future.result();records.append(row)
                    progress(run,f'seed{row["seed"]} · {row["condition"]}',len(records),9,metrics={'dev_macro':row['summary']['development']['macro_exact'],'novel_combo_macro':row['summary']['development']['novel_composition_macro'],'train_seconds':row['train_seconds'],'hidden_changed':False},neural=neural)
                    write_json(run/'trials.json',records);print(json.dumps({k:row[k] for k in ('seed','condition','summary','causal','train_seconds')}),flush=True)
                    try:job=next(jobs);pending[pool.submit(trial,job)]=job
                    except StopIteration:pass
        def row(seed,condition):return next(r for r in records if r['seed']==seed and r['condition']==condition)
        summary={c:{'macro_exact':float(np.mean([r['summary']['development']['macro_exact'] for r in records if r['condition']==c])),
            'novel_composition_macro':float(np.mean([r['summary']['development']['novel_composition_macro'] for r in records if r['condition']==c])),
            'by_kind':{k:float(np.mean([r['summary']['development']['by_kind'][k] for r in records if r['condition']==c])) for k in KINDS}} for c in CONDITIONS}
        paired={c:interval([row(s,'full')['summary']['development']['macro_exact']-row(s,c)['summary']['development']['macro_exact'] for s in SEEDS]) for c in ('no_image','no_question')}
        causal={a:interval([row(s,'full')['summary']['development']['macro_exact']-row(s,'full')['causal'][a]['macro_exact'] for s in SEEDS]) for a in ('blank_image','blank_question','swapped_image')}
        gates={'full_macro65_everyseed':all(row(s,'full')['summary']['development']['macro_exact']>=.65 for s in SEEDS),
            'each_kind50_everyseed':all(v>=.5 for s in SEEDS for v in row(s,'full')['summary']['development']['by_kind'].values()),
            'novel_macro60_everyseed':all(row(s,'full')['summary']['development']['novel_composition_macro']>=.6 for s in SEEDS),
            'trained_controls20_allpositive_lowerCIpositive':all(stats['mean']>=.2 and stats['lower95']>0 and all(row(s,'full')['summary']['development']['macro_exact']>row(s,c)['summary']['development']['macro_exact'] for s in SEEDS) for c,stats in paired.items()),
            'all_causal_drops20':all(stats['mean']>=.2 for stats in causal.values()),
            'all_checkpoints_exact_hidden_frozen':all(r['checkpoint_exact'] and r['continuation_exact'] and r['hidden_unchanged'] for r in records)}
        passed=all(gates.values());result={'run':run.relative_to(root).as_posix(),'decision':'FRESH_VISUAL_WORD_VALIDATION_REQUIRED' if passed else 'STOP_VISUAL_WORD_READOUT_V1',
            'summary':summary,'paired_trained_control_gains':paired,'paired_causal_drops':causal,'gates':gates,'dataset':info['data'],'dataset_sha256':info['sha256'],
            'trials':len(records),'deployed':False,'hidden_representation_learning':False,'normal_photo_qa_proven':False,'retention_confirmed':False,
            'proposal30_approved':False,'main_weights_changed':False,'resources':monitor.report(),'cap':cap,
            'technical_recovery':failure,'total_program_active_seconds_including_failed_attempt':time.perf_counter()-began,
            'limits':['Syntheticattributes/ownedgeometricimages,3freshmodels; not naturalphoto/OCR/generalVQA/lifelong or originalP1hiddenlearning.',
                'Outputlocalreadout only, hiddenfrozen. Fullwordoutputs are neural, rendererlabels never inferenceinput.',
                'Onefixedscreen; no after-results encoder/LR/epoch/cachemixture/modelsize/gridextension or mainimport.',
                'Qwen4Btext-only not visionbaseline, no fakeimage-as-text or VLMdownload/9Bstartup.']}
        (run/'report.md').write_text('# Joint RGB/question to actual neuralwords\n\nFixedreadoutonly feasibility, heldoutcompositions/wordings and causalcontrols. Hiddenfrozen, no oracle/template/classnameformatter at inference. No normalvision/mainclaim.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/visual-word-readout-latest.json',result)
        activity(run,'Hoàn tất9lượt jointvisualwords',result['decision']+'; giữ ngưỡng/ablations, khôngmainimport hoặc hiddenlearningclaim.',evidence={'gates':gates,'paired_trained_control_gains':paired,'proposal30_approved':False})
        print(json.dumps({'run':result['run'],'decision':result['decision'],'summary':summary,'gates':gates},ensure_ascii=False))


if __name__=='__main__':main()
