"""Read real inference/learning transactions. No training or job-launch on GET."""
import json,time,hashlib,math,base64
from datetime import datetime,timezone
from pathlib import Path
from ..runtime.work_trace import read
from ..runtime.experiment_status import RUN_ID


def answers_case(root, identifier):
    """Pinned public corpus/reference; labels never enter neural input."""
    if not isinstance(identifier,str) or len(identifier)>80:raise ValueError('Invalid case ID')
    root=Path(root).resolve()
    reference=json.loads((root/'artifacts/qwen-dialogue-reference-latest.json').read_text(encoding='utf-8'))
    data=(root/reference['data']).resolve()
    if not data.is_relative_to(root/'data/candidates'):raise ValueError('Invalid corpus path')
    records=[json.loads(line) for line in (data/'experiences.jsonl').read_text(encoding='utf-8').splitlines()]
    case=next((r for r in records if r['id']==identifier and r['split']=='development'),None)
    if case is None:raise ValueError('Unknown public development case')
    qwen=next((r for r in reference['outputs'] if r['id']==identifier),None)
    if qwen is None or qwen['messages']!=case['messages'] or qwen['one_reference_target']!=case['target']:
        raise ValueError('Reference and dataset disagree')
    manifest=(root/reference['run']/'manifest.json').resolve()
    if not manifest.is_relative_to(root/'runs'):raise ValueError('Invalid reference run')
    measured=json.loads(manifest.read_text(encoding='utf-8'))['created_utc']
    return case,{'generated':qwen['generated'],'finish_reason':qwen['finish_reason'],'measured_at_utc':measured,
        'run':reference['run'],'scope':'archived measured Qwen3-4B; full prefix; not live or an oracle'}


def observe_answer(app,root,data):
    dataset=data.get('dataset')
    if dataset=='g1_state':
        return observe_reviewed_state_prompt(app,root,data)
    if dataset is not None:raise ValueError('Unknown observation dataset')
    case,reference=answers_case(root,data.get('case_id'))
    prompt=next(m['text'] for m in reversed(case['messages']) if m['role']=='user')
    return app.chat(prompt,record_experience=False,observation={'case_id':case['id'],
        'messages':case['messages'],'target':case['target'],'qwen':reference,
        'input_scope':'SCAR-X receives latest user text only; Qwen received full prefix. Not an equal-context quality benchmark.'})


def observe_reviewed_state_prompt(app,root,data):
    root=Path(root).resolve();identifier=data.get('case_id')
    if not isinstance(identifier,str) or len(identifier)>80:raise ValueError('Invalid public case ID')
    info=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'))
    path=(root/info['data']).resolve()
    if not path.is_relative_to(root/'data/candidates') or not path.is_file() or path.stat().st_size>1000000:raise ValueError('Invalid public dataset')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=info['sha256']:raise ValueError('Reviewed dataset checksum mismatch')
    case=next((r for r in json.loads(raw)['items'] if r['id']==identifier),None)
    if case is None:raise ValueError('Unknown reviewed public case')
    return app.chat(case['prompt'],record_experience=False,observation={'target':case['target'],
        'case_id':'g1-state/'+identifier,'source_sha256':info['sha256'],
        'input_scope':'Actual live SCAR-X, public prompt only, continuous neural state. Archived Qwen used this single prompt with its own fresh context. Targets not fed into neural input/learning.'})


def feed(app,root):
    with app.lock:
        rows=app.db.execute("SELECT id,created,action,details FROM events WHERE action IN ('answer-trace','teach','online-learn','online-reject') ORDER BY id DESC LIMIT 180").fetchall()
        controller={'accepted':app.dialogue.accepted_updates,'rejected':app.dialogue.rejected_updates,
            'queued':sum(l['status']=='queued' for l in app.dialogue.lessons),'updates':app.model.core.updates}
        live=[]
        for identifier,created,action,details in rows:
            item=json.loads(details);stamp=datetime.fromisoformat(created).timestamp()
            live.append({**item,'event_id':f'live-{identifier}','time':stamp,'kind':'answer' if action=='answer-trace' else 'learning',
                'action':action,'scope':'Lõi đang dùng','run_id':None})
    records=live+read(Path(root).resolve())
    records.sort(key=lambda r:r['time'],reverse=True)
    references=archived_single_prompt_references(root)
    for row in records:
        if row.get('kind')=='answer' and not row.get('qwen') and isinstance(row.get('prompt'),str) and row['prompt'] in references:
            row['qwen']=references[row['prompt']]
    return {'activities':[r for r in records if r['kind']=='activity'][:24],
        'answers':[r for r in records if r['kind']=='answer'][:60],
        'learning':[r for r in records if r['kind']=='learning'][:100],
        'controller':controller,'updated':time.time()}


def archived_single_prompt_references(root):
    out={}
    for name in ('qwen-g1-state-reference-latest.json','qwen-word-control-reference-latest.json','qwen-direct-word-reference.json'):
        path=Path(root)/'artifacts'/name
        if not path.is_file() or path.stat().st_size>2000000:continue
        try:
            value=json.loads(path.read_text(encoding='utf-8'));run=value.get('run')
            if not isinstance(run,str) or not run.startswith('runs/') or not RUN_ID.fullmatch(run[5:]):continue
            rows=value.get('outputs')
            if not isinstance(rows,list):continue
            for row in rows[:128]:
                if not isinstance(row,dict):continue
                prompt=row.get('prompt');generated=row.get('generated')
                if not isinstance(prompt,str) or len(prompt)>2048 or not isinstance(generated,str) or len(generated)>8192:continue
                out[prompt]={'generated':generated,'finish_reason':row.get('finish_reason'),'run':run,'measured_at_utc':row.get('measured_at_utc'),
                    'scope':'Actual archived Qwen3-4B on this exact single public user prompt; no tools/history. Different training/RAM budget; not a truth oracle or serving proxy.'}
        except (OSError,ValueError,TypeError,AttributeError):continue
    return out


def control_answer_archive(root,identifier):
    """Latest measured public experiment file; no rerun or score selection."""
    if not isinstance(identifier,str) or not RUN_ID.fullmatch(identifier):return {}
    root=Path(root).resolve();base=(root/'runs').resolve();run=(base/identifier).resolve()
    if not run.is_relative_to(base):return {}
    def measured_files(directory):
        candidates=list(directory.glob('control-*.json'))
        for condition in ('frozen','paired','cross_family','erase_trace'):candidates.extend(directory.glob('*-'+condition+'.json'))
        return sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True)[:24]
    try:
        files=measured_files(run)
        if not files:
            source=run/'result.json'
            if not source.is_file():source=run/'answer-source.json'
            if not source.is_file() or source.stat().st_size>262144:return {}
            pointer=json.loads(source.read_text(encoding='utf-8'));linked=pointer.get('source_run',pointer.get('scarx_control_run',''))
            if not isinstance(linked,str) or not linked.startswith('runs/') or not RUN_ID.fullmatch(linked[5:]):return {}
            identifier=linked[5:];run=(base/identifier).resolve()
            if not run.is_relative_to(base):return {}
            files=measured_files(run)
        if not files:return {}
        corpus=root/'data/dialogue/generation1.json'
        if not corpus.is_file() or corpus.stat().st_size>1_000_000:return {}
        pairs={(p,f['answer']) for f in json.loads(corpus.read_text(encoding='utf-8'))['families'] for p in f['train']}
        for name in ('g1-curriculum-reviewed.json','g1-state-prompts-reviewed.json'):
            info_path=root/'artifacts'/name
            if not info_path.is_file():continue
            info=json.loads(info_path.read_text(encoding='utf-8'));path=(root/info['data']).resolve()
            if not path.is_relative_to(root/'data/candidates') or not path.is_file() or path.stat().st_size>1_000_000:return {}
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=info['sha256']:return {}
            pairs.update((r['prompt'],r['target']) for r in json.loads(raw)['items'])
        references=archived_single_prompt_references(root)
        for path in files:
            if not path.resolve().is_relative_to(run) or path.stat().st_size>1_000_000:continue
            raw=path.read_bytes();data=json.loads(raw);record=data.get('record',{});outputs=data.get('outputs',{})
            seed=record.get('seed');encoding=record.get('input_encoding')
            condition=record.get('condition');mechanism=None
            if record.get('method')=='isolated_surrogate_bptt' and record.get('responses')==1024:
                if seed not in (28001,28019,28027) or encoding not in ('legacy','known-digest128-v1') or path.name!=f'control-{seed}-{encoding}.json':continue
                mechanism='isolated_bptt'
            elif record.get('experiences')==1024 and record.get('observations')==1024:
                if seed not in (31013,31019,31033) or condition not in ('frozen','paired','cross_family','erase_trace') or path.name!=f'{seed}-{condition}.json':continue
                mechanism='observed_association';encoding='known-digest128-v1'
            if mechanism is None:continue
            rows=[]
            for split in ('development','prior','train16'):
                values=outputs.get(split,[])
                if not isinstance(values,list) or len(values)>256:continue
                for index,row in enumerate(values):
                    if not isinstance(row,dict):continue
                    prompt=row.get('prompt');target=row.get('target');generated=row.get('generated')
                    if not isinstance(prompt,str) or not isinstance(target,str) or (prompt,target) not in pairs or not isinstance(generated,str) or len(generated)>2048:continue
                    rows.append({'case_id':split+'/'+str(index+1),'split':split,'prompt':prompt,'target':target,'generated':generated,
                        'style_match':row.get('normalized_style_exact') if type(row.get('normalized_style_exact')) is bool else None,
                        'qwen':references.get(prompt)})
            if rows:return {'run_id':identifier,'file':path.name,'signature':hashlib.sha256(raw).hexdigest(),
                'seed':seed,'input_encoding':encoding,'condition':condition,'learning_mechanism':mechanism,'case_count':len(rows),'rows':rows,
                'record_saved_at_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                'scope':('ARCHIVED ACTUAL ISOLATED BPTT CONTROL' if mechanism=='isolated_bptt' else 'ARCHIVED OBSERVED-ASSOCIATION EXPERIMENT; labelconditionedtraining, NO goldinput at inference')+'; not live/main; file timestamp, not individual inference time.'}
        return {}
    except (OSError,ValueError,TypeError,KeyError,AttributeError):return {}


def vision_answer_archive(root,identifier):
    """Owned synthetic audit images/actual outputs only; never arbitrary photos."""
    if not isinstance(identifier,str) or not RUN_ID.fullmatch(identifier):return {}
    root=Path(root).resolve();run=(root/'runs'/identifier).resolve()
    if not run.is_relative_to(root/'runs'):return {}
    try:
        result_path=run/'result.json';output_path=run/'outputs.json'
        if not result_path.is_file() or result_path.stat().st_size>262144 or not output_path.is_file() or output_path.stat().st_size>2_000_000:return {}
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if not isinstance(result,dict) or result.get('decision') not in ('VISION_QUESTION_DROPPED_BEFORE_NEURAL_CORE','VISION_QUESTION_INPUT_PRESENT_REQUIRES_GROUNDING') or result.get('new_training') is not False:return {}
        dataset=(root/result['data']).resolve();candidate_base=(root/'data/candidates').resolve();expected_dir=(candidate_base/identifier).resolve()
        if not expected_dir.is_relative_to(candidate_base) or not dataset.is_relative_to(expected_dir) or dataset!=(expected_dir/'qa-fixtures.json').resolve() or not dataset.is_file() or dataset.stat().st_size>1_000_000:return {}
        source=dataset.read_bytes()
        if hashlib.sha256(source).hexdigest()!=result['data_sha256']:return {}
        data=json.loads(source);cases={r['id']:r for r in data['cases']};images={}
        if data.get('automatic_learning_allowed') is not False:return {}
        from ..evaluation.vision_fixtures import image_uri,pixel_thumbnail
        from .pilots import PilotLab
        for scene in data['scenes']:
            filename=scene['file']
            if filename not in {f's{i:02d}.png' for i in range(1,13)}:return {}
            path=(dataset.parent/filename).resolve()
            if not path.is_relative_to(dataset.parent) or not path.is_file() or path.stat().st_size>10000:return {}
            raw=path.read_bytes()
            if not raw.startswith(b'\x89PNG\r\n\x1a\n') or hashlib.sha256(raw).hexdigest()!=scene['sha256']:return {}
            uri=image_uri(raw);pixels,_=PilotLab.pixels(uri);images[scene['id']]=(uri,pixel_thumbnail(pixels))
        raw_outputs=output_path.read_bytes();outputs=json.loads(raw_outputs);rows=[]
        if not isinstance(outputs,list) or len(outputs)>240:return {}
        for record in outputs:
            case=cases.get(record.get('id'))
            if case is None or any(record.get(k)!=case.get(k) for k in ('prompt','target','image_sha256','scene_id')):continue
            digit=record.get('digit');p=record.get('probabilities')
            if type(digit) is not int or not 0<=digit<=9 or not isinstance(p,list) or len(p)!=10 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in p):continue
            image,encoded=images[case['scene_id']]
            rows.append({'case_id':case['id'],'prompt':case['prompt'],'target':case['target'],
                'generated':json.dumps({'digit':digit,'probabilities':p}), 'style_match':None,'qwen':None,
                'output_type':'digit_classifier','image_data_uri':image,'encoded_image_data_uri':encoded,
                'image_source':'Owned synthetic renderer '+case['scene_id'],
                'qwen_missing_reason':result['reason_missing_Qwen_vision']})
        return {'run_id':identifier,'signature':hashlib.sha256(raw_outputs+source).hexdigest(),'learning_mechanism':'vision_path_audit',
            'case_count':len(rows),'rows':rows,'scope':'Actual publicrelease vision calls; rawdigitreadout, not VQA or languagegeneration.',
            'record_saved_at_utc':datetime.fromtimestamp(output_path.stat().st_mtime,timezone.utc).isoformat()} if rows else {}
    except (OSError,ValueError,TypeError,KeyError,AttributeError):return {}


def direct_visual_answer_archive(root,identifier):
    """37complete864DEVrows/96ownedimages from latestfile, never infer on GET."""
    if not isinstance(identifier,str) or not RUN_ID.fullmatch(identifier):return {}
    root=Path(root).resolve();base=(root/'runs').resolve();run=(base/identifier).resolve()
    if not run.is_relative_to(base):return {}
    try:
        path=run/'manifest.json'
        if not path.is_file() or path.stat().st_size>262144:return {}
        config=json.loads(path.read_text(encoding='utf-8')).get('config',{})
        if config.get('protocol')!='experiments/direct-visual-bridge-v1.md' or config.get('experiences')!=1536 or config.get('proposal30_approved') is not True:return {}
        dataset=(root/config['dataset']).resolve()
        if not dataset.is_relative_to(root/'data/candidates') or dataset.name!='dataset.json' or not dataset.is_file() or dataset.stat().st_size>2_000_000:return {}
        raw_data=dataset.read_bytes()
        if hashlib.sha256(raw_data).hexdigest()!=config['dataset_sha256']:return {}
        data=json.loads(raw_data)
        if data.get('automatic_learning_allowed') is not False:return {}
        cases={r['id']:r for r in data['cases'] if r['split']=='development'};scenes={s['id']:s for s in data['scenes'] if s['split']=='development'}
        if len(cases)!=864:return {}
        methods=('correct','shuffled','zero','frozen','no_image','no_question','bptt_control')
        files=[run/f'visual37-{s}-{m}.json' for s in (37313,37321,37339) for m in methods];files=sorted((p for p in files if p.is_file()),key=lambda p:p.stat().st_mtime,reverse=True)
        for path in files:
            if path.stat().st_size>6_000_000:continue
            raw=path.read_bytes();value=json.loads(raw);record=value.get('record',{});outputs=value.get('outputs',{}).get('development');seed=record.get('seed');method=record.get('method')
            if seed not in (37313,37321,37339) or method not in methods or path.name!=f'visual37-{seed}-{method}.json' or record.get('experiences')!=1536 or not isinstance(outputs,list) or len(outputs)!=864:continue
            if record.get('hidden_updates')!=(0 if method in ('zero','frozen') else 1536):continue
            rows=[];images={};seen=set()
            for output in outputs:
                case=cases.get(output.get('id'));scene=scenes.get(output.get('scene_id'));generated=output.get('generated')
                if case is None or scene is None or case['id'] in seen or any(output.get(k)!=case.get(k) for k in ('prompt','target','kind','language','scene_id')) or not isinstance(generated,str) or len(generated)>2048:return {}
                seen.add(case['id']);sid=scene['id']
                if sid not in images:
                    filename=scene['file']
                    if filename not in {f'development-{i:03d}.png' for i in range(1,97)}:return {}
                    image=(dataset.parent/filename).resolve()
                    if not image.is_relative_to(dataset.parent) or not image.is_file() or image.stat().st_size>10000:return {}
                    pixels=image.read_bytes()
                    if not pixels.startswith(b'\x89PNG\r\n\x1a\n') or hashlib.sha256(pixels).hexdigest()!=scene['sha256']:return {}
                    images[sid]={'image_data_uri':'data:image/png;base64,'+base64.b64encode(pixels).decode('ascii'),'image_source':'Ownedscene '+sid+' · method '+method+' (missingmodality inputs zeroed if named)'}
                rows.append({'case_id':case['id'],'scene_id':sid,'prompt':case['prompt'],'target':case['target'],'generated':generated,
                    'style_match':output.get('exact') if type(output.get('exact')) is bool else None,'qwen':None,
                    'qwen_missing_reason':'Qwen4Btext-only NOT visionreference; no image-as-text substitute. ActualQwentext36 separate.',
                    'novel_composition':case.get('novel_composition') is True})
            return {'run_id':identifier,'file':path.name,'signature':hashlib.sha256(raw+raw_data).hexdigest(),'learning_mechanism':'direct_visual_bridge',
                'case_count':864,'rows':rows,'images':images,'method':method,'seed':seed,'record_saved_at_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                'scope':'Actual37 autoregressiveALIFvisualwords/ownedDEV, proposal30localcacheeligibility; BPTT onlyisolatedcontrol, NOT main/normalvision.'}
        return {}
    except (OSError,ValueError,TypeError,KeyError,AttributeError):return {}


def direct_word_answer_archive(root,identifier):
    """Actual complete64DEVutterances, validated against reviewedpublicdataset."""
    if not isinstance(identifier,str) or not RUN_ID.fullmatch(identifier):return {}
    root=Path(root).resolve();base=(root/'runs').resolve();run=(base/identifier).resolve()
    if not run.is_relative_to(base):return {}
    try:
        path=run/'manifest.json'
        if not path.is_file() or path.stat().st_size>262144:return {}
        config=json.loads(path.read_text(encoding='utf-8')).get('config',{})
        if config.get('protocol')!='experiments/direct-word-bridge-v1.md' or config.get('responses')!=1024:
            pointer=run/'answer-source.json'
            if not pointer.is_file() or pointer.stat().st_size>65536:return {}
            linked=json.loads(pointer.read_text(encoding='utf-8')).get('source_run','')
            if not isinstance(linked,str) or not linked.startswith('runs/') or not RUN_ID.fullmatch(linked[5:]):return {}
            identifier=linked[5:];run=(base/identifier).resolve();path=run/'manifest.json'
            if not run.is_relative_to(base) or not path.is_file() or path.stat().st_size>262144:return {}
            config=json.loads(path.read_text(encoding='utf-8')).get('config',{})
        if config.get('protocol')!='experiments/direct-word-bridge-v1.md' or config.get('responses')!=1024:return {}
        dataset=(root/config['dataset']).resolve()
        if not dataset.is_relative_to(root/'data/candidates') or dataset.name!='reviewed.json' or dataset.stat().st_size>1_000_000:return {}
        raw_data=dataset.read_bytes()
        if hashlib.sha256(raw_data).hexdigest()!=config.get('dataset_sha256'):return {}
        data=json.loads(raw_data)
        if data.get('format')!='scarx-direct-word-style-dataset-v1' or data.get('all_rows_read') is not True or data.get('automatic_learning_allowed') is not False:return {}
        from ..evaluation.direct_word_data import TARGETS
        cases={c['id']:c for c in data['items'] if c['split']=='development'}
        if len(cases)!=64 or any(c['family'] not in TARGETS or c['target']!=TARGETS[c['family']] for c in cases.values()):return {}
        files=[run/f'word-{s}-{c}.json' for s in (36313,36319,36341) for c in ('correct','shuffled','zero','frozen','bptt_control')]
        files=sorted((p for p in files if p.is_file()),key=lambda p:p.stat().st_mtime,reverse=True);references=archived_single_prompt_references(root)
        for path in files:
            if path.stat().st_size>1_000_000:continue
            raw=path.read_bytes();value=json.loads(raw);r=value.get('record',{});outputs=value.get('outputs',{}).get('development');seed=r.get('seed');method=r.get('method')
            if seed not in (36313,36319,36341) or method not in ('correct','shuffled','zero','frozen','bptt_control') or path.name!=f'word-{seed}-{method}.json' or r.get('experiences')!=1024 or not isinstance(outputs,list) or len(outputs)!=64:continue
            rows=[];seen=set()
            for out in outputs:
                c=cases.get(out.get('id'));generated=out.get('generated')
                if c is None or c['id'] in seen or any(c[k]!=out.get(k) for k in ('prompt','target','family','language')) or not isinstance(generated,str) or len(generated)>2048:return {}
                seen.add(c['id']);rows.append({'case_id':c['id'],'prompt':c['prompt'],'target':c['target'],'generated':generated,
                    'style_match':out.get('exact') if type(out.get('exact')) is bool else None,'family':c['family'],'qwen':references.get(c['prompt'])})
            return {'run_id':identifier,'file':path.name,'signature':hashlib.sha256(raw+json.dumps(references,sort_keys=True).encode()).hexdigest(),
                'learning_mechanism':'direct_word_bridge','case_count':64,'rows':rows,'seed':seed,'method':method,
                'record_saved_at_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                'scope':'Actual autoregressiveALIFwords on64heldoutclosedstyleprompts; not normalchat/P1/P2/main. BPTTonlyseparatecontrol.'}
        return {}
    except (OSError,ValueError,TypeError,KeyError,AttributeError):return {}


def eligibility_answer_archive(root,identifier):
    """Latest complete registered numeric trial; regenerate public input checks only."""
    if not isinstance(identifier,str) or not RUN_ID.fullmatch(identifier):return {}
    root=Path(root).resolve();base=(root/'runs').resolve();run=(base/identifier).resolve()
    if not run.is_relative_to(base):return {}
    try:
        manifest_path=run/'manifest.json'
        if not manifest_path.is_file() or manifest_path.stat().st_size>262144:return {}
        config=json.loads(manifest_path.read_text(encoding='utf-8')).get('config',{})
        if config.get('protocol')!='experiments/direct-readout-eligibility-v1.md' or config.get('proposal30_approved') is not True or config.get('examples_per_trial')!=1024:return {}
        import numpy as np
        from ..evaluation.tasks21 import representation
        from ..evaluation.binding_audit21 import strict_binding
        tasks=('order','xor','strict_binding');methods=('correct','shuffled','zero','frozen','bptt_control')
        seeds={'development':(35113,35119,35141),'confirmation':(35201,35219,35227,35233,35239,35243,35257,35261)}
        files=[run/f'{phase}-{task}-{seed}-{method}.json' for phase,ss in seeds.items() for task in tasks for seed in ss for method in methods]
        files=sorted((p for p in files if p.is_file()),key=lambda p:p.stat().st_mtime,reverse=True)
        for path in files:
            if path.stat().st_size>5_000_000:continue
            raw=path.read_bytes();value=json.loads(raw);record=value.get('record',{});outputs=value.get('outputs')
            phase=record.get('phase');task=record.get('task');seed=record.get('seed');method=record.get('method')
            if phase not in seeds or task not in tasks or seed not in seeds[phase] or method not in methods or path.name!=f'{phase}-{task}-{seed}-{method}.json' or record.get('experiences')!=1024 or record.get('head_updates')!=1024 or record.get('proposal30_approved') is not True:continue
            count=256 if phase=='development' else 512
            if not isinstance(outputs,list) or len(outputs)!=count:continue
            expected_x,expected_y=strict_binding(seed+1390000,count) if task=='strict_binding' else representation(task,seed+1390000,count)
            if hashlib.sha256(expected_x.tobytes()).hexdigest()!=record.get('test_inputs_sha256') or hashlib.sha256(expected_y.astype(np.int32).tobytes()).hexdigest()!=record.get('test_targets_sha256'):continue
            inputs=np.asarray([r.get('input') for r in outputs],np.float32)
            if inputs.shape!=(count,12,8) or not np.isfinite(inputs).all() or not np.array_equal(inputs,expected_x):continue
            rows=[]
            for i,r in enumerate(outputs):
                prediction=r.get('prediction');target=r.get('target');probabilities=r.get('probabilities');case_id=f'{phase}-{task}-{seed}-{method}/{i}'
                if r.get('id')!=case_id or type(prediction) is not int or prediction not in (0,1) or type(target) is not int or target!=int(expected_y[i]) or not isinstance(probabilities,list) or len(probabilities)!=2 or any(type(p) not in (int,float) or not np.isfinite(p) or not 0<=p<=1 for p in probabilities) or abs(sum(probabilities)-1)>1e-5:return {}
                if prediction!=int(np.argmax(probabilities)):return {}
                rows.append({'case_id':case_id,'prompt':json.dumps(expected_x[i].tolist()),'generated':str(prediction),'target':str(target),'style_match':prediction==target,
                    'qwen':None,'qwen_missing_reason':'Numeric tensorinput: no comparableQwenmeasurement. Existingtextanswers are not substituted.'})
            if rows:return {'run_id':identifier,'file':path.name,'signature':hashlib.sha256(raw).hexdigest(),'learning_mechanism':'direct_readout_eligibility',
                'case_count':len(rows),'rows':rows,'task':task,'phase':phase,'method':method,'seed':seed,
                'record_saved_at_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                'scope':'Actual numericALIFclassifications, proposal30research only; not normalchat/vision/main. BPTT only isolatedcontrol method.'}
        return {}
    except (OSError,ValueError,TypeError,KeyError,AttributeError):return {}


def visual_word_answer_archive(root,identifier):
    """All864actualDEVwords from latest completed visualword trial, not best."""
    if not isinstance(identifier,str) or not RUN_ID.fullmatch(identifier):return {}
    root=Path(root).resolve();base=(root/'runs').resolve();run=(base/identifier).resolve()
    if not run.is_relative_to(base):return {}
    try:
        result_path=run/'result.json'
        if not result_path.is_file() or result_path.stat().st_size>262144:return {}
        result=json.loads(result_path.read_text(encoding='utf-8'))
        decisions=('STOP_VISUAL_WORD_READOUT_V1','FRESH_VISUAL_WORD_VALIDATION_REQUIRED','ISOLATED_VISUAL_WORD_CONTROL_ONLY')
        if result.get('decision') not in decisions:
            linked=result.get('source_run','')
            if not isinstance(linked,str) or not linked.startswith('runs/') or not RUN_ID.fullmatch(linked[5:]):return {}
            identifier=linked[5:];run=(base/identifier).resolve();result_path=run/'result.json'
            if not run.is_relative_to(base) or not result_path.is_file() or result_path.stat().st_size>262144:return {}
            result=json.loads(result_path.read_text(encoding='utf-8'))
        control=result.get('decision')=='ISOLATED_VISUAL_WORD_CONTROL_ONLY'
        if result.get('decision') not in decisions:return {}
        if control:
            if result.get('control_only') is not True or result.get('weights_imported_into_scarx') is not False:return {}
        elif result.get('hidden_representation_learning') is not False:return {}
        dataset=(root/result['dataset']).resolve();candidate_base=(root/'data/candidates').resolve()
        if not dataset.is_relative_to(candidate_base) or dataset.name!='dataset.json' or not dataset.is_file() or dataset.stat().st_size>2_000_000:return {}
        raw_data=dataset.read_bytes()
        if hashlib.sha256(raw_data).hexdigest()!=result['dataset_sha256']:return {}
        data=json.loads(raw_data);cases={r['id']:r for r in data['cases'] if r['split']=='development'};scenes={s['id']:s for s in data['scenes'] if s['split']=='development'}
        files=[run/f'control-visual-{s}.json' for s in (33113,33119,33133)] if control else [run/f'{s}-{c}.json' for s in (33113,33119,33133) for c in ('full','no_image','no_question')]
        files=sorted((p for p in files if p.is_file()),key=lambda p:p.stat().st_mtime,reverse=True)
        for path in files:
            if path.stat().st_size>5_000_000:continue
            raw=path.read_bytes();value=json.loads(raw);record=value.get('record',{});outputs=value.get('outputs',{}).get('development',[])
            seed=record.get('seed');condition=record.get('condition')
            if seed not in (33113,33119,33133) or condition not in ('full','no_image','no_question') or record.get('experiences')!=1536 or not isinstance(outputs,list) or len(outputs)>864:continue
            if control:
                if condition!='full' or record.get('method')!='isolated_visual_surrogate_bptt' or record.get('control_only') is not True or record.get('hidden_unchanged') is not False or path.name!=f'control-visual-{seed}.json':continue
            elif record.get('hidden_unchanged') is not True or path.name!=f'{seed}-{condition}.json':continue
            rows=[];images={}
            for output in outputs:
                case=cases.get(output.get('id'));scene=scenes.get(output.get('scene_id'));generated=output.get('generated')
                if case is None or scene is None or any(output.get(k)!=case.get(k) for k in ('prompt','target','kind','language','scene_id')) or not isinstance(generated,str) or len(generated)>2048:continue
                scene_id=scene['id']
                if scene_id not in images:
                    filename=scene['file']
                    if filename not in {f'development-{i:03d}.png' for i in range(1,97)}:return {}
                    image_path=(dataset.parent/filename).resolve()
                    if not image_path.is_relative_to(dataset.parent) or not image_path.is_file() or image_path.stat().st_size>10000:return {}
                    pixels=image_path.read_bytes()
                    if not pixels.startswith(b'\x89PNG\r\n\x1a\n') or hashlib.sha256(pixels).hexdigest()!=scene['sha256']:return {}
                    images[scene_id]={'image_data_uri':'data:image/png;base64,'+base64.b64encode(pixels).decode('ascii'),'image_source':'Owned renderedscene '+scene_id+' · neuralinputcondition '+condition}
                rows.append({'case_id':case['id'],'scene_id':scene_id,'prompt':case['prompt'],'target':case['target'],'generated':generated,
                    'style_match':output.get('exact') if type(output.get('exact')) is bool else None,'qwen':None,
                    'qwen_missing_reason':'ExistingQwen4Btext-only, not a realvisionbaseline. No image-as-text substitute.','novel_composition':case.get('novel_composition') is True})
            if rows:return {'run_id':identifier,'file':path.name,'signature':hashlib.sha256(raw+raw_data).hexdigest(),
                'learning_mechanism':'isolated_visual_word_bptt' if control else 'visual_word_readout','case_count':len(rows),'rows':rows,'images':images,'condition':condition,'seed':seed,
                'record_saved_at_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                'scope':'Actual isolated ALIF-BPTT CONTROL words; trainedhidden, NOT locallearning/main/normalvision.' if control else 'Actual autoregressiveneuralwords, hiddenFROZEN; syntheticDEV, not normalvision/main. Missinginputs are zeroedaccordingto condition.'}
        return {}
    except (OSError,ValueError,TypeError,KeyError,AttributeError):return {}
