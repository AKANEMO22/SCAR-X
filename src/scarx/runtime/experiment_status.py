"""Bounded local experiment observability; never launch/restart a job here."""
import atexit,json,os,re,time,tempfile,threading,math
from pathlib import Path
import psutil
from .atomic_file import replace_with_retry

_owned={};_lock=threading.RLock()
RUN_ID=re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,160}$')
CATALOG=(
 ('CáchdạyALIF / readonlyFREE-vs-GOLDPREFIX','word-training-objective-audit.json'),
 ('Directvisual37 / cachedeligibility và imageQA thật','direct-visual-bridge-latest.json'),
 ('Directvisual37 / math/parity/timing','direct-visual-preflight.json'),
 ('Directwordbridge / sinh từ thật, actualAGYdata','direct-word-bridge-latest.json'),
 ('Directwordbridge / parity và dự toán','direct-word-preflight.json'),
 ('Qwen4B / actual64wordbridgeprompts','qwen-direct-word-reference.json'),
 ('AGY128wordprompts / review64train64dev','direct-word-data-reviewed.json'),
 ('AGY128wordprompts / sinh thật','antigravity-direct-word-prompts.json'),
 ('Proposal30 / directreadoutsignal + localeligibility','direct-eligibility-latest.json'),
 ('Proposal30 / math, codec và timingtrainonly','direct-eligibility-engineering.json'),
 ('Ảnh/câu hỏi→readout / readonlyinformationaudit','visual-information-latest.json'),
 ('ALIF visualwords / BPTT chỉ đối chứng','visual-word-control-latest.json'),
 ('ALIF visualcontrol / parity và timing','visual-word-control-preflight.json'),
 ('Audit visualwords / hiddenfrozen, qualitySTOP','visual-word-readout-audit.json'),
 ('Joint visualwordreadout / đối chứng thật','visual-word-readout-latest.json'),
 ('Dataset visualwords / ownedsplits','visual-word-dataset-latest.json'),
 ('Vision questionpath / inference thật, khôngtraining','vision-question-path-latest.json'),
 ('Visionquestions / editorialreview','vision-questions-reviewed.json'),
 ('AntigravityCLI / 32visionquestions thật','antigravity-vision-questions-latest.json'),
 ('Audit observedassociation / quality vẫnSTOP','g1-observed-association-audit.json'),
 ('Observedassociation / localpre-post và ablation','g1-observed-association-latest.json'),
 ('Creditnoise / readonly tín hiệu synapse','g1-credit-noise-latest.json'),
 ('Qwen4B / bù47publicprompt thật','qwen-word-control-reference-latest.json'),
 ('ALIF wordlearnability / BPTT chỉ đối chứng','g1-word-learnability-control-latest.json'),
 ('ALIF wordcontrol / parity và timing','g1-word-control-preflight.json'),
 ('Windows observer / atomicreader recovery','g1-observer-atomic-recovery.json'),
 ('Audit27checkpoint / quality vẫnSTOP','g1-word-score-inputs-audit.json'),
 ('Wordconditioning / can thiệp readonly','g1-word-conditioning-latest.json'),
 ('ALIF word-score / vòng input cuối','g1-word-score-inputs-latest.json'),
 ('Trace tự động / kiểm tra phần mềm và serving','passive-work-trace-latest.json'),
 ('G1 input identity / không đạt','g1-input-identity-latest.json'),
 ('G1 paired-score chuyển sang từ / không đạt','g1-word-score-transfer-latest.json'),
 ('P2 voltage/adaptation / không đạt','p2-neuron-state-observability-latest.json'),
 ('G1 decoder search / đã dừng','g1-bounded-beam-latest.json'),
 ('G1 representation/decoder / chẩn đoán','g1-representation-decoder-latest.json'),
 ('G1 controller / dừng lặp guard','g1-guard-stop-latest.json'),
 ('G1 explicitproject teaching / live','g1-project-supervision-latest.json'),
 ('G1 curated curriculum / không đạt','g1-curated-curriculum-latest.json'),
 ('G1 curriculum / semanticreview','g1-curriculum-reviewed.json'),
 ('G1 curriculum Antigravity / chờ review','antigravity-g1-curriculum-latest.json'),
 ('G1 dấu câu / guard acceptance','g1-surface-acceptance-latest.json'),
 ('G1 ôn kỹ năng cơ bản / chưa deploy','g1-basic-rehearsal-latest.json'),
 ('Qwen4B / 48câu công khai mới','qwen-g1-state-reference-latest.json'),
 ('G1 trạng thái hội thoại / inference thật','g1-conversational-state-latest.json'),
 ('Antigravity / prompts đã rà soát','g1-state-prompts-reviewed.json'),
 ('ALIF time constants / diagnostic','p2-timescale-view-latest.json'),
 ('ALIF firing noise / diagnostic','p2-inference-dynamics-latest.json'),
 ('P2 cue/time/noise / chỉ chẩn đoán','p2-cue-information-latest.json'),
 ('P2 acquisition/retention / baseline','p2-paired-score-diagnostic-latest.json'),
 ('Paired-score / audit và attribution','p1-paired-score-v2-audit.json'),
 ('Paired-score / xác nhận P1','p1-paired-score-v2-latest.json'),
 ('Score eligibility / audit toán','p1-score-integrity-latest.json'),
 ('Audit optimizer và coverage / chỉ đối chứng','p1-control-factors-latest.json'),
 ('Audit local classifier / vòng cuối','p1-local-classifier-v2-audit.json'),
 ('Local classifier / vòng cuối','p1-local-classifier-v2-latest.json'),
 ('Trace thực trên lõi đang dùng','live-observatory-latest.json'),
 ('Tạo dữ liệu Antigravity','antigravity-generation-latest.json'),
 ('Local classifier / P1','p1-local-classifier-latest.json'),
 ('Ảnh thật / đối chứng đã audit','rgb-object-grounding-audit.json'),
 ('Bộ nhớ làm việc khi phát byte','byte-working-memory-latest.json'),
 ('Audit trạng thái decoder','decoder-state-retention-audit.json'),
 ('BPTT chỉ làm đối chứng','byte-bptt-control-latest.json'),
 ('Decoder UTF-8','byte-dialogue-latest.json'),
 ('Hội thoại và Qwen4B','qwen-dialogue-reference-latest.json'),
 ('Prototype hội thoại theo sự kiện','event-dialogue-latest.json'),
 ('Chuẩn hóa readout / 20 context','p2-conditioning-confirmation-latest.json'),
 ('Random feedback / đã dừng','p1-random-feedback-latest.json'))


def _safe(value,depth=0):
    if depth>5:return '[truncated]'
    if value is None or isinstance(value,(bool,int)):return value
    if isinstance(value,float):return value if value==value and abs(value)!=float('inf') else None
    if isinstance(value,str):return value[:800]
    if isinstance(value,dict):return {str(k)[:80]:_safe(v,depth+1) for k,v in list(value.items())[:48] if str(k).lower() not in ('token','password','api_key','authorization','secret','credentials','headers')}
    if isinstance(value,(list,tuple)):return [_safe(v,depth+1) for v in value[:128]]
    return str(value)[:160]


def _location(run):
    run=Path(run).resolve()
    if run.parent.name!='runs' or not RUN_ID.fullmatch(run.name):raise ValueError('Expected registered run directory')
    return run.parent.parent/'local/experiment-status'/(run.name+'.json')


def _write(path,value):
    raw=json.dumps(_safe(value),ensure_ascii=False,allow_nan=False).encode()
    if len(raw)>196608:raise ValueError('Experiment status exceeds192KB')
    path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        replace_with_retry(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def start_run(run,config):
    path=_location(run);proc=psutil.Process()
    record={'run_id':Path(run).name,'run':f'runs/{Path(run).name}','kind':config.get('observability_kind','research_clone'),
      'title':config.get('display_title',Path(run).name),'model_scope':config.get('model_scope','Bản thử nghiệm; không thay model trò chuyện'),
      'pid':proc.pid,'process_created':proc.create_time(),'started':time.time(),'updated':time.time(),'status':'running',
      'stage':'Đã đăng ký lượt chạy','config':_safe(config),'events':[],'progress':None,'neural':None}
    with _lock:_write(path,record);_owned[str(Path(run).resolve())]=path


def progress(run,stage,completed=None,total=None,metrics=None,neural=None,task=None):
    path=_location(run)
    with _lock:
        if not path.exists():return
        value=json.loads(path.read_text(encoding='utf-8'));value['updated']=time.time();value['stage']=str(stage)[:300]
        if value.get('pid')!=os.getpid():return # parent owns the journal; workers return measurements to it
        value['progress']={'completed':completed,'total':total}
        if metrics is not None:value['metrics']=_safe(metrics)
        if neural is not None:value['neural']=_safe(neural)
        if task is not None:value['task']=str(task)[:200]
        value['events']=(value.get('events',[])+[{'time':time.time(),'stage':stage,'progress':value['progress'],'metrics':_safe(metrics)}])[-40:]
        _write(path,value)


def complete_run(run,result):
    path=_location(run)
    with _lock:
        if not path.exists():return
        value=json.loads(path.read_text(encoding='utf-8'));value.update(status='completed',updated=time.time(),finished=time.time(),stage='Đã có kết quả')
        if value.get('pid')!=os.getpid():return
        value['result']=_summary(result);_write(path,value)


def _summary(value):
    keys=('decision','summary','selection','direct_gate','primary_representation_gate','normal_chat_or_vision_proven','normal_photo_qa_proven',
      'normal_chat_vision_proven','deployed','live_model_replaced','resources','screen_trials','confirmation_trials','metric_correction','image_grounding_gate',
      'successful_batches','expected_batches','trained','validation_pending','control_summary','all_matched','all_cold_exact',
      'new_training','cases','inferences','items','completed','expected','truncated','error','server_stopped',
      'all_order_passes','terminal_punctuation_overlap_count','all_state_unchanged','weights_unchanged',
      'stopped_identical_proposal','updates_before','updates_after','before_actual_neural','after_actual_neural','checks','pytest_summary',
      'gates','application_route_closed','paired_encoding_dev_gain','paired_hidden_dev_gain','paired_hidden_prior_gain','source_quality_decision',
      'control_only','interpretations','forecast_round_seconds','formal_training_permitted',
      'main_local_learning_success','weights_imported_into_scarx','scarx_uses_qwen_proxy','normal_chat_or_vision_equivalence',
      'credit_signal_rows','credit_noise_contrasts','all_source_checkpoints_unchanged','all_weights_learningstate_rng_unchanged',
      'proposal30_approved','zero_frozen_equivalence','formal_screen_permitted','forecast_screen_seconds','candidate_signal_locality','zero_frozen_exact',
      'word_objective_rows','corpus_structure','all_source_freeoutputs_and_teachermetrics_reproduced',
      'visual_information_rows','information_collapse_rows','diagnostic_flags','neural_training','neural_update_steps',
      'scene_count','question_count','question_context_channels_nonzero_total','all_scene_question_inputs_identical',
      'all_scene_question_outputs_identical','Qwen_vision_reference_measured')
    result={k:value[k] for k in keys if k in value}
    if isinstance(value.get('conditions'),list):
        trials=_product_trials(value['conditions'])
        if trials:result['product_trials']=trials
    return _safe(result)


def _product_trials(rows):
    trials=[]
    for row in rows[:24]:
        if not isinstance(row,dict) or not isinstance(row.get('states'),dict):continue
        trial={k:row[k] for k in ('mode','condition','curriculum_order_seed','surface_normalization','screen_pass','lost_old_correct_digits') if k in row}
        reset=row['states'].get('canonicalreset',{})
        if not isinstance(reset,dict):reset={}
        for name,key in (('known_reset','known'),('antigravity_reset','antigravity')):
            v=reset.get(key);trial[name]=v if type(v) in (int,float) and math.isfinite(v) else None
        trial['lost_guards_total']=sum(v.get('lost_personal_runtime_guards',v.get('lost_personal_guards',0))
            for v in row['states'].values() if isinstance(v,dict) and type(v.get('lost_personal_runtime_guards',v.get('lost_personal_guards',0))) in (int,float))
        trials.append(trial)
    return trials


def _on_exit():
    for _,path in list(_owned.items()):
        try:
            value=json.loads(path.read_text(encoding='utf-8'))
            if value['status']=='running':value.update(status='stopped_without_result',updated=time.time(),stage='Tiến trình đã dừng; chưa có kết quả đầy đủ');_write(path,value)
        except (OSError,ValueError):continue


atexit.register(_on_exit)


def _process(record):
    try:
        proc=psutil.Process(record['pid'])
        if abs(proc.create_time()-record['process_created'])>.25 or not proc.is_running():return False,{}
        processes=[proc]+proc.children(recursive=True);rss=commit=0
        for p in processes:
            try:info=p.memory_info();rss+=info.rss;commit+=getattr(info,'private',info.vms)
            except (psutil.NoSuchProcess,psutil.AccessDenied):continue
        return True,{'resident_tree_bytes':rss,'commit_tree_bytes':commit,'processes':len(processes)}
    except (psutil.NoSuchProcess,psutil.AccessDenied,KeyError):return False,{}


def dashboard(root):
    root=Path(root).resolve();directory=root/'local/experiment-status';jobs=[]
    files=sorted(directory.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:16] if directory.exists() else []
    for path in files:
        try:
            if path.stat().st_size>196608:continue
            value=json.loads(path.read_text(encoding='utf-8'));alive,memory=_process(value)
            value['process_alive']=alive;value['process_resources']=memory
            if value.get('status')=='running' and not alive:value['status']='stopped_without_result';value['stage']='Không còn tiến trình; chưa có kết quả đầy đủ'
            if value.get('status')=='completed' and value.get('result',{}).get('primary_representation_gate') is True:
                statistics=primary_statistics(root,value.get('run_id'))
                if statistics:value['result']['primary_statistics']=statistics
            if value.get('status')=='completed':
                retention=retention_statistics(root,value.get('run_id'))
                if retention:value.setdefault('result',{})['retention_statistics']=retention
                product=product_statistics(root,value.get('run_id'))
                if product:value.setdefault('result',{})['product_trials']=product
                words=word_statistics(root,value.get('run_id'))
                if words:value.setdefault('result',{})['word_trials']=words
                signals=credit_signal_statistics(root,value.get('run_id'))
                if signals:
                    value.setdefault('result',{})['credit_signal_rows']=signals['rows']
                    value['result']['credit_noise_contrasts']=signals['contrasts']
                association=association_statistics(root,value.get('run_id'))
                if association:value.setdefault('result',{})['association_statistics']=association
                vision=vision_path_statistics(root,value.get('run_id'))
                if vision:value.setdefault('result',{})['vision_path_rows']=vision
                vw=visual_word_statistics(root,value.get('run_id'))
                if vw:value.setdefault('result',{})['visual_word_statistics']=vw
                dw=direct_word_statistics(root,value.get('run_id'))
                if dw:value.setdefault('result',{})['direct_word_statistics']=dw
            jobs.append(value)
        except (OSError,ValueError):continue
    history=[]
    for title,name in CATALOG:
        path=root/'artifacts'/name
        try:
            if not path.is_file() or path.stat().st_size>5000000:continue
            value=json.loads(path.read_text(encoding='utf-8'));run=value.get('run')
            if not isinstance(run,str) or not run.startswith('runs/') or not RUN_ID.fullmatch(run[5:]):continue
            history.append({'title':title,'run_id':run[5:],'status':'completed','result':_summary(value),
              'report_url':'/api/research/report?run_id='+run[5:]})
        except (OSError,ValueError):continue
    return {'jobs':jobs,'history':history,'updated':time.time(),'note':'Kết quả và trace thí nghiệm tách khỏi phiên trò chuyện. Không có thao tác tự chạy lại job.'}


def primary_statistics(root,run_id):
    """Only finite registered confirmation metrics; no arbitrary result contents."""
    if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id):return {}
    base=(Path(root)/'runs').resolve();path=(base/run_id/'result.json').resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return {}
    try:
        data=json.loads(path.read_text(encoding='utf-8'));rows=data.get('confirmation_statistics',data.get('source_confirmation_statistics',{}));result={}
        if not isinstance(rows,dict):return {}
        def finite(value):return value if type(value) in (int,float) and math.isfinite(value) else None
        for task,row in list(rows.items())[:8]:
            if not isinstance(row,dict):continue
            gain=row.get('probe_gain',{});direct=row.get('direct',{})
            if not isinstance(gain,dict) or not isinstance(direct,dict):continue
            values={'gain':finite(gain.get('mean')),'lower95':finite(gain.get('lower95')),'upper95':finite(gain.get('upper95')),
                'direct':finite(direct.get('mean')),'seeds':finite(gain.get('n_seeds')),'holm_p':finite(row.get('holm_gain_p')),'pass':row.get('pass') is True}
            if all(values[k] is not None for k in ('gain','lower95','upper95','direct')):result[str(task)[:80]]=values
        return result
    except (OSError,ValueError,TypeError):return {}


def credit_signal_statistics(root,run_id):
    """Finite descriptive signal metrics from the registered readonly audit."""
    if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id):return {}
    base=(Path(root)/'runs').resolve();path=(base/run_id/'result.json').resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return {}
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data,dict) or data.get('decision')!='READONLY_CREDIT_SIGNAL_DIAGNOSTIC_ONLY' or data.get('new_training') is not False:return {}
        summary=data.get('summary',{});paired=data.get('paired_statistics',{})
        if not isinstance(summary,dict) or not isinstance(paired,dict):return {}
        def finite(v):return v if type(v) in (int,float) and math.isfinite(v) else None
        rows=[];contrasts={}
        for state in ('frozen_final','local_final','surrogate_final'):
            methods=summary.get(state,{})
            if not isinstance(methods,dict):continue
            for method in ('existing_score','crossfit_baseline_score','biased_surrogate'):
                values=methods.get(method)
                if isinstance(values,dict):rows.append({'source_state':state,'method':method,
                    **{k:finite(values.get(k)) for k in ('replicate_cosine','cross_signal_to_pair_noise','pair_noise_energy','sample_mean_norm')}})
            comparisons=paired.get(state,{})
            if not isinstance(comparisons,dict):continue
            contrasts[state]={}
            for key in ('crossfit_vs_existing_noise_ratio','surrogate_minus_existing_replicate_cosine'):
                value=comparisons.get(key)
                if isinstance(value,dict):contrasts[state][key]={k:finite(value.get(k)) for k in ('mean','lower95','upper95','n_seeds')}
        return {'rows':rows,'contrasts':contrasts} if rows else {}
    except (OSError,ValueError,TypeError):return {}


def association_statistics(root,run_id):
    """Bounded measured observation experiment summary, no fitted answers."""
    if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id):return {}
    base=(Path(root)/'runs').resolve();path=(base/run_id/'result.json').resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return {}
    try:
        data=json.loads(path.read_text(encoding='utf-8'));source=run_id
        if not isinstance(data,dict):return {}
        decisions=('STOP_OBSERVED_ASSOCIATION_V1','OBSERVED_ASSOCIATION_FRESH_VALIDATION_REQUIRED')
        if data.get('decision') not in decisions:
            linked=data.get('source_run')
            if not isinstance(linked,str) or not linked.startswith('runs/') or not RUN_ID.fullmatch(linked[5:]):return {}
            source=linked[5:];path=(base/source/'result.json').resolve()
            if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return {}
            data=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data,dict) or data.get('decision') not in decisions:return {}
        summary=data.get('summary',{});probes=data.get('probe_by_condition',{})
        if not isinstance(summary,dict) or not isinstance(probes,dict):return {}
        def finite(v):return v if type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 else None
        rows=[]
        for condition in ('frozen','paired','cross_family','erase_trace'):
            r=summary.get(condition,{})
            if not isinstance(r,dict):continue
            values={split:finite(r.get(split,{}).get('normalized_style_exact')) for split in ('train16','development','prior') if isinstance(r.get(split,{}),dict)}
            if values.get('development') is not None:rows.append({'condition':condition,**values,'prompt_probe':finite(probes.get(condition))})
        return {'source_run':source,'decision':data['decision'],'rows':rows} if rows else {}
    except (OSError,ValueError,TypeError):return {}


def vision_path_statistics(root,run_id):
    if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id):return []
    base=(Path(root)/'runs').resolve();path=(base/run_id/'result.json').resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return []
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data,dict) or data.get('new_training') is not False or data.get('decision') not in ('VISION_QUESTION_DROPPED_BEFORE_NEURAL_CORE','VISION_QUESTION_INPUT_PRESENT_REQUIRES_GROUNDING'):return []
        scenes=data.get('by_scene',{})
        if not isinstance(scenes,dict):return []
        rows=[]
        for i in range(1,13):
            identifier=f's{i:02d}';r=scenes.get(identifier)
            if not isinstance(r,dict):continue
            row={'scene':identifier}
            for key in ('cases','distinct_renderer_targets','prototype_distinct_multimodal_sequences'):
                v=r.get(key);row[key]=v if type(v) is int and 0<=v<=256 else None
            for key in ('all_questions_same_neural_input','all_questions_same_neural_probabilities'):
                v=r.get(key);row[key]=v if type(v) is bool else None
            rows.append(row)
        return rows
    except (OSError,ValueError,TypeError):return []


def visual_word_statistics(root,run_id):
    if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id):return {}
    base=(Path(root)/'runs').resolve();path=(base/run_id/'result.json').resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return {}
    try:
        data=json.loads(path.read_text(encoding='utf-8'));source=run_id
        if not isinstance(data,dict):return {}
        decisions=('STOP_VISUAL_WORD_READOUT_V1','FRESH_VISUAL_WORD_VALIDATION_REQUIRED','ISOLATED_VISUAL_WORD_CONTROL_ONLY','STOP_DIRECT_VISUAL_BRIDGE_V1','FRESH_DIRECT_VISUAL_VALIDATION_REQUIRED')
        if data.get('decision') not in decisions:
            linked=data.get('source_run','')
            if not isinstance(linked,str) or not linked.startswith('runs/') or not RUN_ID.fullmatch(linked[5:]):return {}
            source=linked[5:];path=(base/source/'result.json').resolve()
            if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return {}
            data=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data,dict) or data.get('decision') not in decisions:return {}
        control=data.get('decision')=='ISOLATED_VISUAL_WORD_CONTROL_ONLY'
        direct=data.get('decision') in ('STOP_DIRECT_VISUAL_BRIDGE_V1','FRESH_DIRECT_VISUAL_VALIDATION_REQUIRED')
        if direct:
            if data.get('proposal30_approved') is not True or data.get('deployed') is not False:return {}
        elif control:
            if data.get('control_only') is not True or data.get('weights_imported_into_scarx') is not False:return {}
        elif data.get('hidden_representation_learning') is not False:return {}
        summary=data.get('summary',{})
        if not isinstance(summary,dict):return {}
        def finite(v):return v if type(v) in (int,float) and math.isfinite(v) else None
        rows=[]
        for condition in (('correct','shuffled','zero','frozen','no_image','no_question','bptt_control') if direct else ('bptt_control','frozen_full','frozen_no_image','frozen_no_question') if control else ('full','no_image','no_question')):
            r=summary.get(condition,{})
            if not isinstance(r,dict):continue
            kinds=r.get('by_kind',{});kinds=kinds if isinstance(kinds,dict) else {}
            rows.append({'condition':condition,'macro_exact':finite(r.get('macro_exact')),'novel_composition_macro':finite(r.get('novel_composition_macro')),'macro_train':finite(r.get('macro_train')),'probe_gain':finite(r.get('probe_gain')),
                **{k:finite(kinds.get(k)) for k in ('single_color','single_shape','object_count','left_color')}})
        contrasts={}
        for group in ('paired_trained_control_gains','paired_causal_drops','paired_gains'):
            values=data.get(group,{})
            if not isinstance(values,dict):continue
            contrasts[group]={}
            for key in ('frozen_full','frozen','shuffled','no_image','no_question','blank_image','blank_question','swapped_image'):
                r=values.get(key)
                if isinstance(r,dict):contrasts[group][key]={k:finite(r.get(k)) for k in ('mean','lower95','upper95','n_seeds')}
        return {'source_run':source,'decision':data['decision'],'rows':rows,'contrasts':contrasts,'control_only':control,'direct_eligibility':direct,'interpretation':data.get('interpretation') if control else None}
    except (OSError,ValueError,TypeError):return {}


def direct_word_statistics(root,run_id):
    """Bounded typed wordpilot metrics, one sourcehop for Qwenreference jobs."""
    if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id):return {}
    base=(Path(root)/'runs').resolve();run=(base/run_id).resolve()
    if not run.is_relative_to(base):return {}
    try:
        path=run/'result.json'
        if not path.is_file():path=run/'answer-source.json'
        if not path.is_file() or path.stat().st_size>500000:return {}
        data=json.loads(path.read_text(encoding='utf-8'));decisions=('STOP_DIRECT_WORD_BRIDGE_V1','WORD_BRIDGE_FRESH_VALIDATION_REQUIRED')
        if data.get('decision') not in decisions:
            linked=data.get('source_run','')
            if not isinstance(linked,str) or not linked.startswith('runs/') or not RUN_ID.fullmatch(linked[5:]):return {}
            run_id=linked[5:];path=(base/run_id/'result.json').resolve()
            if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>500000:return {}
            data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('decision') not in decisions or data.get('deployed') is not False:return {}
        values=data.get('summary',{});summary={}
        if not isinstance(values,dict):return {}
        for method in ('correct','shuffled','zero','frozen','bptt_control'):
            r=values.get(method)
            if not isinstance(r,dict):continue
            summary[method]={k:(r[k] if type(r.get(k)) in (int,float) and math.isfinite(r[k]) else None) for k in ('macro_development','macro_train','probe_gain','restore_initial_hidden_dev')}
        return {'source_run':run_id,'decision':data['decision'],'summary':summary}
    except (OSError,ValueError,TypeError,KeyError,AttributeError):return {}


def product_statistics(root,run_id):
    if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id):return []
    base=(Path(root)/'runs').resolve();path=(base/run_id/'result.json').resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return []
    try:
        value=json.loads(path.read_text(encoding='utf-8'));rows=value.get('conditions')
        return _safe(_product_trials(rows)) if isinstance(rows,list) else []
    except (OSError,ValueError,TypeError):return []


def word_statistics(root,run_id):
    """Bounded actual word-transfer metrics; never expose model/data paths."""
    if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id):return []
    base=(Path(root)/'runs').resolve();path=(base/run_id/'result.json').resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return []
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data,dict):return []
        summary=data.get('summary',{});source=run_id;control_requested=data.get('control_only') is True
        if not isinstance(summary,dict):return []
        if not any(k in summary for k in ('legacy','known-sparse128-v1','known-digest128-v1')):
            linked=data.get('source_run')
            if not isinstance(linked,str) or not linked.startswith('runs/') or not RUN_ID.fullmatch(linked[5:]):return []
            source=linked[5:];path=(base/source/'result.json').resolve()
            if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return []
            data=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data,dict):return []
            if control_requested and data.get('control_only') is not True:return []
            summary=data.get('summary',{})
            if not isinstance(summary,dict):return []
        decision=data.get('decision')
        if decision not in ('STOP_WORD_SCORE_INPUT_TRANSFER_V2','WORD_SCORE_INPUT_FRESH_VALIDATION_REQUIRED','ISOLATED_WORD_LEARNABILITY_CONTROL_ONLY'):decision=None
        rows=[]
        def metric(row,split,key):
            r=row.get(split,{});v=r.get(key) if isinstance(r,dict) else None
            return v if type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 else None
        for encoding in ('legacy','known-sparse128-v1','known-digest128-v1'):
            variants=summary.get(encoding,{})
            if not isinstance(variants,dict):continue
            for condition in ('frozen','full','shuffled','local_reference'):
                if condition=='local_reference' and data.get('control_only') is not True:continue
                row=variants.get(condition)
                if not isinstance(row,dict):continue
                dev=metric(row,'development','normalized_style_exact')
                if dev is None:continue
                rows.append({'input_encoding':encoding,'condition':condition,'development':dev,
                    'prior':metric(row,'prior','normalized_style_exact'),'train16':metric(row,'train16','normalized_style_exact'),
                    'teacher_token_accuracy':metric(row,'development','teacher_token_accuracy'),'source_run':source,'source_quality_decision':decision,
                    'control_only':data.get('control_only') is True,
                    'method':('surrogate-BPTT · đối chứng' if condition=='full' else 'readout-only · baseline' if condition=='frozen' else 'local-score v2 · đã STOP') if data.get('control_only') is True else condition})
        return rows
    except (OSError,ValueError,TypeError):return []


def retention_statistics(root,run_id):
    """Finite P2 metrics, optionally from the one linked source run of an audit."""
    base=(Path(root)/'runs').resolve()
    def read(identifier):
        if not isinstance(identifier,str) or not RUN_ID.fullmatch(identifier):return None
        path=(base/identifier/'result.json').resolve()
        if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return None
        return json.loads(path.read_text(encoding='utf-8'))
    try:
        data=read(run_id)
        if not isinstance(data,dict):return {}
        summary=data.get('summary',{})
        def is_retention(value):return isinstance(value,dict) and any(isinstance(r,dict) and 'mean_forgetting' in r and 'final' in r for r in value.values())
        source=run_id
        if not is_retention(summary):
            linked=data.get('source_run')
            if not isinstance(linked,str) or not linked.startswith('runs/'):return {}
            source=linked[5:];data=read(source)
            if not isinstance(data,dict):return {}
            summary=data.get('summary',{})
        if not is_retention(summary):return {}
        rows={};gates=data.get('absolute_gates',{})
        if not isinstance(gates,dict):gates={}
        for name,row in list(summary.items())[:12]:
            if not isinstance(row,dict) or 'final' not in row:continue
            values={k:(row[k] if type(row.get(k)) in (int,float) and math.isfinite(row[k]) else None)
                for k in ('acquisition','final','mean_forgetting','worst_forgetting','joint_refit_probe')}
            if values['final'] is not None:
                rows[str(name)[:100]]={**values,'gate':gates.get(name) is True,'offline':str(name).startswith('offline')}
        return {'source_run':source,'rows':rows} if rows else {}
    except (OSError,ValueError,TypeError):return {}


def report(root,run_id):
    if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id):raise ValueError('Invalid run ID')
    base=(Path(root)/'runs').resolve();path=(base/run_id/'report.md').resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>400000:raise ValueError('Report unavailable or too large')
    return path.read_bytes()
