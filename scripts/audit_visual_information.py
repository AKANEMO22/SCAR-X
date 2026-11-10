"""Registered read-only visual path + disposable probe audit; never train core."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib
from types import SimpleNamespace
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from visual_word_control_common import source_data,sample_stream,SEEDS,SOURCE,atomic
from scarx.evaluation.visual_information import observe,source_model,protected,digest,ridge,accuracy,collision,counterfactual_pairs,pair_summary,rank_summary
from scarx.evaluation.vision_fixtures import expected
from scarx.neural.language import tokenize
from scarx.evaluation.phase21 import interval
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.work_trace import activity
from scarx.runtime.experiment_status import progress
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

CONTROL_SOURCE='runs/20261109T234349Z-visual-word-learnability-control-v1-e51fe9'


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();receipt,data,train,dev,images,vocab,source_rows,files=source_data()
    control=json.loads((root/CONTROL_SOURCE/'result.json').read_text(encoding='utf-8'));assert control['control_only'] and control['weights_imported_into_scarx'] is False and len(control['trials'])==3
    scenes={s['id']:s for s in data['scenes']};cases=train+dev;n=len(train)
    semantic_keys=[r['kind']+'/'+r['language']+'/'+expected(scenes[r['scene_id']],r['kind'],'en') for r in cases]
    classes=sorted(set(semantic_keys[:n]));assert len(classes)==20 and set(semantic_keys[n:])<=set(classes)
    semantic=np.array([classes.index(k) for k in semantic_keys]);task_keys=sorted({r['kind']+'/'+r['language'] for r in train});tasks=np.array([task_keys.index(r['kind']+'/'+r['language']) for r in cases])
    firstword=np.array([vocab.index(tokenize(r['target'])[0]) for r in cases]);index_by_id={r['id']:i for i,r in enumerate(train)}
    sources=[]
    for family,rows in [('frozen32',[r for r in source_rows if r['condition']=='full']),('bptt_control33',control['trials'])]:
        for row in sorted(rows,key=lambda r:r['seed']):
            source=SOURCE if family=='frozen32' else CONTROL_SOURCE
            raw_path=root/source/(f"{row['seed']}-full.json" if family=='frozen32' else f"control-visual-{row['seed']}.json")
            sources.append((family,row,raw_path));files.extend([row['checkpoint'],raw_path.relative_to(root).as_posix()])
    files.extend([CONTROL_SOURCE+'/result.json',CONTROL_SOURCE+'/trials.json'])
    run=new_run(root,'visual-information-audit-v1',{'protocol':'experiments/visual-information-audit-v1.md','display_title':'READ ONLY · ảnh/câu hỏi→ALIF→readout information',
        'model_scope':'Existing32/33checkpoints only. No synapseupdates; auxiliaryridgeclassifier is NOT modelanswers/locallearning/main. Proposal30 pending.',
        'model_seeds':SEEDS,'sources':6,'forward_cases_per_source':1728,'neural_training':False,'dataset':receipt['data'],'dataset_sha256':receipt['sha256']},
        data_description='Pinned owned192PNG/AGY16train16devwordings, no neuraltraining. Exact1536exposures reused for disposablediagnosticheadfit.',data_files=files)
    (run/'report.md').write_text('# Read-only visual information audit\n\nIn progress; no neurallearning/mainchange. Auxiliaryprobe labels must not be presented as neural answers.\n',encoding='utf-8')
    activity(run,'Bắt đầu audit đường thông tin đãđăngký','3tests diagnostic qua. Đo6checkpoint32/33, không cập nhật synapse; fixedridgeprobe ngoàicore dùng đúng1536trainexposures. Không Transformer/mainchange hoặc mởproposal30.',status='started',evidence={'protocol':'experiments/visual-information-audit-v1.md','neural_training':False,'targeted_tests_passed':3})
    rows=[];stage_rows=[];raw_references={};raw_vectors=None;pairs=counterfactual_pairs(dev,semantic[n:]);stage_sources=[];observations_total=0
    with ResourceMonitor(interval=.1) as monitor:
        for family,source,outputs_path in sources:
            seed=source['seed'];path=root/source['checkpoint'];payload=path.read_bytes();assert hashlib.sha256(payload).hexdigest()==source['sha256']
            m=source_model(payload,control=family=='bptt_control33');before=protected(m);updates=m.core.updates;experiences=m.experiences;ticks=m.ticks;seen=m.images_seen
            sampler=SimpleNamespace(sampler_rng=np.random.default_rng(seed+12100000));exposure_cases=list(sample_stream(sampler,train,1536));exposures=np.array([index_by_id[r['id']] for r in exposure_cases])
            counts={}
            for r in exposure_cases:key=r['kind']+'/'+r['language']+'/'+r['target'];counts[key]=counts.get(key,0)+1
            assert counts==source['stratum_counts'] and sampler.sampler_rng.bit_generator.state==m.sampler_rng.bit_generator.state
            observed=[]
            for i,case in enumerate(cases,1):
                observed.append(observe(m,case['prompt'],images[case['scene_id']]))
                if i%288==0:progress(run,f'{family}/seed{seed} · forward{i}/1728',len(rows),6,metrics={'source_family':family,'seed':seed,'observations':i,'neural_updates':0,'elapsed_seconds':time.perf_counter()-began});monitor.check()
                if time.perf_counter()-began>300:raise TimeoutError('Registered300s visualinformation cap')
            observations_total+=len(observed)
            assert protected(m)==before and m.core.updates==updates and m.experiences==experiences and m.ticks-ticks==1728*36 and m.images_seen-seen==1728
            vectors={name:np.stack([r['vectors'][name] for r in observed]) for name in observed[0]['vectors']};actual=np.array([r['first_token'] for r in observed])
            if raw_vectors is None:raw_vectors={k:vectors[k].copy() for k in ('question','rgb_question')}
            else:
                for k,a in raw_vectors.items():np.testing.assert_array_equal(a,vectors[k])
            measured=json.loads(outputs_path.read_text(encoding='utf-8'))['outputs']['development'];assert len(measured)==864
            measured_by_id={r['id']:r for r in measured}
            for case,token in zip(dev,actual[n:]):assert tokenize(measured_by_id[case['id']]['generated'])[0]==vocab[token]
            x=vectors['head_feature'];word_probe=ridge(x[:n],x[n:],firstword[:n],m.core.config.outputs,exposures,mask=m.mask(0))
            semantic_probe=ridge(x[:n],x[n:],semantic[:n],20,exposures);shuffled=ridge(x[:n],x[n:],semantic[:n],20,exposures,shuffle_seed=seed+34000000)
            probe_summary={'firstword':{'train_observed':accuracy(word_probe['train'],firstword[:n],train,word_probe['exposure_counts']),'development':accuracy(word_probe['dev'],firstword[n:],dev)},
                'semantic':{'train_observed':accuracy(semantic_probe['train'],semantic[:n],train,semantic_probe['exposure_counts']),'development':accuracy(semantic_probe['dev'],semantic[n:],dev)},
                'shuffled_semantic':{'development':accuracy(shuffled['dev'],semantic[n:],dev)},'actual_firstword':{'development':accuracy(actual[n:],firstword[n:],dev)}}
            if seed not in raw_references:
                refs={}
                for view,label,class_count in [('question',tasks,8),('rgb_question',semantic,20)]:
                    z=raw_vectors[view];p=ridge(z[:n],z[n:],label[:n],class_count,exposures)
                    refs[view]={'train_observed':accuracy(p['train'],label[:n],train,p['exposure_counts']),'development':accuracy(p['dev'],label[n:],dev),'fit_array_bytes':p['fit_array_bytes']}
                    if view=='rgb_question':refs[view]['shuffled_development']=accuracy(ridge(z[:n],z[n:],label[:n],class_count,exposures,shuffle_seed=seed+34000000)['dev'],label[n:],dev)
                raw_references[seed]=refs
            stage_metrics={}
            for name in ('input_tape',*vectors):
                a=vectors.get(name);hashes=[r['input_tape_sha256'] for r in observed] if a is None else [digest(row) for row in a]
                split={}
                for split_name,offset,selected in [('train',0,train),('development',n,dev)]:
                    hh=hashes[offset:offset+len(selected)];split[split_name]={'firstword':collision(hh,firstword[offset:offset+len(selected)],selected),'semantic':collision(hh,semantic[offset:offset+len(selected)],selected)}
                devhash=hashes[n:];pair_metrics={key:pair_summary(devhash,pp,a[n:] if a is not None else None) for key,pp in pairs.items()}
                stats=rank_summary(a[:n]) if a is not None else {'dimensions':None,'centered_rank_rtol1e6':None,'constant_channels':None}
                stage_metrics[name]={'collisions':split,'counterfactual':pair_metrics,'rank_train_only':stats,'input_tape_numeric_not_retained':a is None}
                stage_sources.append({'family':family,'seed':seed,'stage':name,'dev_semantic_ceiling':split['development']['semantic']['oracle_macro_ceiling'],
                    'dev_firstword_ceiling':split['development']['firstword']['oracle_macro_ceiling'],'conflicting_semantic_groups':split['development']['semantic']['conflicting_groups'],
                    'image_pair_collapse':pair_metrics['image']['exact_collapse_fraction'],'question_pair_collapse':pair_metrics['question']['exact_collapse_fraction'],
                    'rank':stats['centered_rank_rtol1e6']})
            bos_hash=[digest(a) for a in vectors['bos_feature'][n:]];head_hash=[digest(a) for a in x[n:]]
            new_clip_conflicts={key:sum(bos_hash[i]!=bos_hash[j] and head_hash[i]==head_hash[j] for i,j in pp) for key,pp in pairs.items()}
            percase=[{'id':case['id'],'scene_id':case['scene_id'],'prompt':case['prompt'],'target':case['target'],'kind':case['kind'],'language':case['language'],
                'actual_neural_first_word':vocab[int(actual[n+i])],'actual_neural_firstword_correct':bool(actual[n+i]==firstword[n+i]),
                'archived_full_neural_generation':measured_by_id[case['id']]['generated'],'archived_full_generation_source':outputs_path.relative_to(root).as_posix(),
                'auxiliary_firstword_probe_label':vocab[int(word_probe['dev'][i])],
                'auxiliary_semantic_probe_label':classes[int(semantic_probe['dev'][i])],'semantic_target':classes[int(semantic[n+i])],
                'auxiliary_semantic_probe_correct':bool(semantic_probe['dev'][i]==semantic[n+i]),'scope':'ActualBOSword from existingcheckpoint + AUXILIARYCLASSIFIER, not newfullgeneration/mainlearning.'} for i,case in enumerate(dev)]
            record={'family':family,'seed':seed,'probes':probe_summary,'raw_input_references':raw_references[seed],'stages':stage_metrics,
                'new_conflicting_pairs_at_clip':new_clip_conflicts,'mean_head_clip_fraction_development':float(np.mean([r['clipped_fraction'] for r in observed[n:]])),
                'same1536_exposure_multiset':True,'unique_train_cases_exposed':len(set(exposures.tolist())),'source_protected_state_unchanged':True,'source_checkpoint_sha256':source['sha256'],
                'forward_cases':1728,'forward_ticks':1728*36,'neural_update_steps':0,'source_weights_control_only':family=='bptt_control33'}
            atomic(run/f'{seed}-{family}.path-audit.json',{'record':record,'outputs':percase});rows.append(record)
            progress(run,f'{family}/seed{seed} diagnosticscomplete',len(rows),6,metrics={'firstword_model':probe_summary['actual_firstword']['development']['macro'],
                'auxiliary_firstword_probe':probe_summary['firstword']['development']['macro'],'auxiliary_semantic_probe':probe_summary['semantic']['development']['macro'],'neural_updates':0})
            if time.perf_counter()-began>300:raise TimeoutError('Registered300s visualinformation including probes')
            monitor.check()
        firstword_gaps={};representation_gains=[]
        for family in ('frozen32','bptt_control33'):
            selected=[r for r in rows if r['family']==family]
            firstword_gaps[family]=interval([r['probes']['firstword']['development']['macro']-r['probes']['actual_firstword']['development']['macro'] for r in selected])
        for seed in SEEDS:
            def score(f):return next(r['probes']['semantic']['development']['macro'] for r in rows if r['family']==f and r['seed']==seed)
            representation_gains.append(score('bptt_control33')-score('frozen32'))
        rep_interval=interval(representation_gains)
        flags={'observed_component_information_collapse':any(r['dev_semantic_ceiling']<.65 for r in stage_sources) or any(any(r['new_conflicting_pairs_at_clip'].values()) for r in rows),
            'BOS_readout_training_gap':{f:r['mean']>=.1 and r['lower95']>0 for f,r in firstword_gaps.items()},
            'CONTROL_representation_gain':rep_interval['mean']>=.05 and rep_interval['lower95']>0}
        table=[]
        for r in rows:
            p=r['probes'];table.append({'family':r['family'],'seed':r['seed'],'actual_BOS_firstword':p['actual_firstword']['development']['macro'],
                'auxiliary_firstword_probe':p['firstword']['development']['macro'],'auxiliary_semantic_probe':p['semantic']['development']['macro'],
                'auxiliary_shuffled_semantic_probe':p['shuffled_semantic']['development']['macro'],'head_clip_fraction':r['mean_head_clip_fraction_development'],
                'actual_head_feature_rank':r['stages']['head_feature']['rank_train_only']['centered_rank_rtol1e6']})
        for family in ('frozen32','bptt_control33'):
            for stage in rows[0]['stages']:
                selected=[r for r in stage_sources if r['family']==family and r['stage']==stage]
                stage_rows.append({'family':family,'stage':stage,'dev_semantic_ceiling_mean':float(np.mean([r['dev_semantic_ceiling'] for r in selected])),
                    'dev_firstword_ceiling_mean':float(np.mean([r['dev_firstword_ceiling'] for r in selected])),
                    'image_pair_collapse_mean':float(np.mean([r['image_pair_collapse'] for r in selected])),
                    'question_pair_collapse_mean':float(np.mean([r['question_pair_collapse'] for r in selected])),
                    'expected_missingmodality_branch':stage=='question'})
        assert observations_total==10368 and 'torch' not in sys.modules
        for _,source,_ in sources:assert hashlib.sha256((root/source['checkpoint']).read_bytes()).hexdigest()==source['sha256']
        result={'run':run.relative_to(root).as_posix(),'source_run':CONTROL_SOURCE,'frozen_source_run':SOURCE,'decision':'VISUAL_INFORMATION_DIAGNOSTIC_ONLY',
            'visual_information_rows':table,'information_collapse_rows':stage_rows,'diagnostic_flags':flags,'firstword_probe_minus_realhead':firstword_gaps,
            'control_minus_frozen_semantic_probe':rep_interval,'raw_input_references':raw_references,'records':rows,'forward_observations':observations_total,'forward_ticks':observations_total*36,
            'dataset':receipt['data'],'dataset_sha256':receipt['sha256'],'same_training_exposure_multisets':True,'all_source_weights_learningstate_rng_unchanged':True,
            'neural_training':False,'neural_update_steps':0,'probe_fitting':'Disposable closed-form classifier on observedTRAINlabels; not neuralupdate, not inferenceanswers.',
            'torch_imported':False,'deployed':False,'normal_chat_or_vision_proven':False,'main_local_learning_success':False,'proposal30_approved':False,
            'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-began,
            'limits':['Historical6sources/3pairedseeds, inspecteddev and ownedshapes; no finaltest/generalvision/lifelong proof.',
                'Firstwordprobe comparison onlyFIRSTWORD; semanticprobe is classification, never verbal generation.',
                'Question-only branch lacksRGB bydesign; branchceilings do not bound fullcombinedstate/futurewords. Noexactcollisions doesnotprovelearnability.',
                'Offline ridge differentcompute/order than onlinehead, same1536labels; not a deployedlearner or matched-cost qualitygain.',
                'GlobalCONTROLrepresentation gain, ifany, cannot prove permittedlocalrepresentation. STOP32/33/earlierfamilies retained.']}
        atomic(run/'records.json',rows);atomic(run/'stage-source-metrics.json',stage_sources)
        (run/'report.md').write_text('# Read-only visual information diagnostic\n\nNo neurallearning/mainchange; auxiliaryclassifier labels are NOT modelanswers.\n\n'+json.dumps({k:v for k,v in result.items() if k!='records'},indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/visual-information-latest.json',result)
        activity(run,'Hoàn tất audit đường thông tin, không neuraltraining','10368BOSforwardobservations từ6source, fixedridgeprobe vớiđúng1536exposures/seed. Sourceweights/RNG/counters khôngđổi. Probeclassification không thay câu neural hoặc mainquality.',evidence={'decision':result['decision'],'diagnostic_flags':flags,'neural_updates':0,'forward_observations':observations_total})
        print(json.dumps({k:result[k] for k in ('run','decision','visual_information_rows','firstword_probe_minus_realhead','control_minus_frozen_semantic_probe','raw_input_references','resources')},ensure_ascii=False))


if __name__=='__main__':main()
