"""Actual Generation.vision on public release clone; no modeltraining."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.app.generation import Generation
from scarx.app.pilots import PilotLab
from scarx.neural.input_events import InputEvents
from scarx.evaluation.vision_fixtures import scenes,render,image_uri,expected,eligible,pixel_thumbnail
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append,activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();info=json.loads((root/'artifacts/vision-questions-reviewed.json').read_text(encoding='utf-8'))
    path=(root/info['data']).resolve()
    if not path.is_relative_to(root/'data/candidates') or hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Reviewed source mismatch')
    reviewed=json.loads(path.read_text(encoding='utf-8'));questions=reviewed['items'];assert len(questions)==32
    release=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'))
    if hashlib.sha256((root/release['checkpoint']).read_bytes()).hexdigest()!=release['sha256']:raise ValueError('Release weights checksum')
    run=new_run(root,'vision-question-path-audit-v1',{'protocol':'experiments/vision-question-path-audit-v1.md',
        'display_title':'Vision thật · ảnh/câu hỏi có tới neuron không? · KHÔNG HỌC',
        'model_scope':'Isolated publicrelease Generation.vision, canonicalrestore and48active. Not private/live; no proposal30implementation.',
        'scenes':12,'questions':32,'cases':240,'new_training':False,'model_checkpoint':release['checkpoint']},
        data_description='Owned12geometricRGBfixtures +32actualreviewedAGYquestions. Labels solelyrenderer spec, not teacher/model.',data_files=[info['data'],release['checkpoint']])
    dataset=root/'data/candidates'/run.name;dataset.mkdir(parents=True)
    scene_map={};outputs=[];cases=[]
    for scene in scenes():
        payload=render(scene);filename=scene['id']+'.png';(dataset/filename).write_bytes(payload)
        pixels,digest=PilotLab.pixels(image_uri(payload));scene_map[scene['id']]={**scene,'file':filename,'sha256':digest,
            'image_uri':image_uri(payload),'pixels':pixels,'pixel_thumbnail':pixel_thumbnail(pixels)}
        for q in questions:
            if eligible(scene,q['kind']):cases.append({'id':scene['id']+'/'+q['id'],'scene_id':scene['id'],'question_id':q['id'],
                'kind':q['kind'],'language':q['language'],'prompt':q['question'],'target':expected(scene,q['kind'],q['language']),
                'image_file':filename,'image_sha256':digest,'target_source':'owned renderer manifest, not teacher answer'})
    assert len(cases)==240
    scene_manifest=[{k:v for k,v in row.items() if k not in ('image_uri','pixels','pixel_thumbnail')} for row in scene_map.values()]
    data={'cases':cases,'scenes':scene_manifest,'questions_source':info['data'],'questions_sha256':info['sha256'],
        'automatic_learning_allowed':False,'trained':False,'scope':'Synthetic pathdiagnostic only, not normalphotoQAbenchmark'}
    write_json(dataset/'qa-fixtures.json',data)
    (run/'report.md').write_text('# Vision question path\n\nActual publicrelease clone inference in progress; no training/privateinputs/QAabilityclaim.\n',encoding='utf-8')
    activity(run,'Bắt đầu240visioncalls trên bản sao release','Ảnhhình học tựrender, AGYquestionđãreview; đápán từrenderer. Cùngstate/48neurons, realGeneration.vision path, khôngnhậpgold hoặc sửaweights.',status='started',evidence={'cases':240,'new_training':False,'private_model_loaded':False,'proposal30_approved':False})
    app=Generation(root/'local/vision-question-audit'/run.name,background=False)
    try:
        baseline=app._pack();fingerprint=app.model.core.fingerprint();updates=app.model.core.updates;dialogue=app.dialogue.dumps();encoder=InputEvents(len(app.model.vocabulary))
        assert app.model.core.config.hidden==48 and app.model.active_count==48 and encoder.dimensions==app.model.core.config.inputs
        source_hashes={s:hashlib.sha256(np.asarray(row['pixels'],np.float32).tobytes()).hexdigest() for s,row in scene_map.items()}
        with ResourceMonitor(interval=.1) as monitor:
            for index,case in enumerate(cases):
                app._unpack(baseline);scene=scene_map[case['scene_id']];core=app.model.core;original_step=core.step;actual_inputs=[];start_tick=app.model.ticks
                def record_step(x,track=True):
                    actual_inputs.append(np.asarray(x,np.float32).copy());return original_step(x,track)
                core.step=record_step;start=time.perf_counter()
                try:answer=app.vision({'image':scene['image_uri'],'prompt':case['prompt'],'neuron_budget':48,'request_id':f'visionaudit-{index}'})
                finally:del core.step
                elapsed=time.perf_counter()-start;actual=np.stack(actual_inputs)
                assert len(actual)==app.model.ticks-start_tick==16 and answer['workload']['active_neurons']==48
                assert app.model.core.fingerprint()==fingerprint and app.model.core.updates==updates and app.dialogue.dumps()==dialogue
                prototype=encoder.multimodal(case['prompt'],(dataset/scene['file']).read_bytes())
                record={**case,'digit':answer['digit'],'probabilities':answer['probabilities'],'presentation':answer['presentation'],
                    'neural_ticks':len(actual),'seconds':elapsed,'active':48,'allocated':48,
                    'neural_input_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
                    'neural_question_context_nonzero':int(np.count_nonzero(actual[:,len(app.model.vocabulary):app.model.image_start])),
                    'workload_prompt_features_nonzero':answer['workload']['prompt_features_nonzero'],
                    'prototype_multimodal_sha256':hashlib.sha256(prototype.tobytes()).hexdigest(),
                    'prototype_shape':list(prototype.shape),'weight_updates':0}
                outputs.append(record)
                # This is an actual neural digitreadout, NEVER formatted as the
                # renderer's correct natural-language answer.
                append(run,'answer',{'scope':'ACTUAL VISION PATH · publicrelease clone · digits-only, NOT VISUAL QA',
                    'case_id':case['id'],'prompt':case['prompt'],'target':case['target'],
                    'generated':json.dumps({'digit':answer['digit'],'probabilities':answer['probabilities']},ensure_ascii=False),
                    'output_type':'digit_classifier','input_scope':'Same RGBimage + actualuserquestion passed to Generation.vision. Canonical48ALIFclone, no goldanswer/training. Presentation is JSONdigitclassification, not languagegeneration.',
                    'image_data_uri':scene['image_uri'],'encoded_image_data_uri':scene['pixel_thumbnail'],
                    'image_source':'Owned deterministicrenderer fixture '+scene['id'],'neural_ticks':len(actual),'seconds':elapsed,'active':48,'allocated':48,
                    'qwen':None,'qwen_missing_reason':'Qwen3-4B hiện có là text-only, chưa có phép đo QwenVL trên ảnh này. Không dùng text-onlyoutput giả làm visionbaseline.',
                    'checkpoint':release['checkpoint']})
                if (index+1)%16==0 or index+1==len(cases):
                    progress(run,'Actual imagequestioncalls · '+case['scene_id'],index+1,len(cases),metrics={'context_channels_nonzero':record['neural_question_context_nonzero'],
                        'active':48,'last_digit':answer['digit'],'seconds':elapsed},neural={'computed':48,'allocated':48,'tick':app.model.ticks,'spikes':core.spikes.astype(int).tolist(),'voltage':np.round(core.voltage,4).tolist(),'sample_kind':'actual publicrelease visioninference, NO learning'})
                    write_json(run/'outputs.json',outputs);monitor.check()
                if time.perf_counter()-began>120:raise TimeoutError('Registered120s visionpath audit cap')
            by_scene={}
            for identifier in scene_map:
                rows=[r for r in outputs if r['scene_id']==identifier]
                same_inputs=len({r['neural_input_sha256'] for r in rows})==1
                same_probs=all(np.array_equal(np.asarray(rows[0]['probabilities']),np.asarray(r['probabilities'])) for r in rows)
                by_scene[identifier]={'cases':len(rows),'question_kinds':sorted({r['kind'] for r in rows}),'distinct_renderer_targets':len({r['target'] for r in rows}),
                    'all_questions_same_neural_input':same_inputs,'all_questions_same_neural_probabilities':same_probs,
                    'prototype_distinct_multimodal_sequences':len({r['prototype_multimodal_sha256'] for r in rows})}
            color_pairs=[]
            fixtures=scenes()
            for i in range(0,12,2):
                a,b=scene_map[fixtures[i]['id']],scene_map[fixtures[i+1]['id']]
                same_pixels=np.array_equal(a['pixels'],b['pixels'])
                proto_different=not np.array_equal(encoder.image((dataset/a['file']).read_bytes()),encoder.image((dataset/b['file']).read_bytes()))
                assert same_pixels and proto_different
                color_pairs.append({'first':a['id'],'second':b['id'],'same_grayscale_pixels':same_pixels,'RGB_prototype_inputs_differ':proto_different})
            # Restore entire discarded session before shutdown, including RNG.
            app._unpack(baseline);assert app._pack()==baseline
            dropped=all(x['all_questions_same_neural_input'] for x in by_scene.values()) and not any(r['neural_question_context_nonzero'] for r in outputs)
            result={'run':run.relative_to(root).as_posix(),'decision':'VISION_QUESTION_DROPPED_BEFORE_NEURAL_CORE' if dropped else 'VISION_QUESTION_INPUT_PRESENT_REQUIRES_GROUNDING',
                'data':(dataset/'qa-fixtures.json').relative_to(root).as_posix(),'data_sha256':hashlib.sha256((dataset/'qa-fixtures.json').read_bytes()).hexdigest(),
                'scene_count':12,'question_count':32,'inferences':240,'by_scene':by_scene,'color_pairs':color_pairs,
                'all_scene_question_inputs_identical':all(x['all_questions_same_neural_input'] for x in by_scene.values()),
                'all_scene_question_outputs_identical':all(x['all_questions_same_neural_probabilities'] for x in by_scene.values()),
                'question_context_channels_nonzero_total':sum(r['neural_question_context_nonzero'] for r in outputs),
                'source_release_sha256':release['sha256'],'all_source_weights_updates_dialogue_unchanged':True,'discarded_session_restored_exact':True,
                'source_encoder_pixel_sha256':source_hashes,'new_training':False,'deployed':False,'normal_photo_qa_proven':False,'learned_caption_or_language_proven':False,
                'Qwen_vision_reference_measured':False,'reason_missing_Qwen_vision':'Installed Qwen3-4B-Instruct text-only; no QwenVLstarted/downloaded or image-as-text substitute.',
                'proposal30_approved':False,'resources':monitor.report(),'cap':cap,
                'limits':['One fixedpubliccheckpoint, syntheticfixturecounterexamples, not240models/qualitybenchmark.',
                    'Colorloss repeats knownreport13; newend-to-end questioninsensitivity audit, no novelvisionrepresentationclaim.',
                    'Different prototypeinputsequences do not prove learnedvision or semanticquestionunderstanding.',
                    'Question mayaffect defaultcapacity heuristics; here48capacityfixed to isolate actualsemanticinput.']}
            (run/'report.md').write_text('# Vision question path audit\n\n240actual Generation.vision calls on an isolatedpublicrelease clone. Questionsemanticchannels zero, identicalimage/state/budget gives identicalneuralinputs/output across differentquestionkinds. NOT training/generalvision success. Detailed outputs.json includes allwrong/rightdigits, targets onlyrendererannotations.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
            write_json(run/'result.json',result);write_json(root/'artifacts/vision-question-path-latest.json',result)
            activity(run,'Hoàn tất240visioncalls thật','12/12ảnh có cùngneuralinput/probabilities khi đổicâuhỏi; contextchannels luôn0. Giữcore/weights, đây làpathlimitation không qualitysuccess. RGBprototype chỉpreserveinput, chưatrained.',evidence={'inferences':240,'scene_groups':12,'context_channels_nonzero':0,'new_training':False,'proposal30_approved':False})
            print(json.dumps({'run':result['run'],'decision':result['decision'],'inferences':240,'same_question_inputs_by_scene':12,'question_context_nonzero':0,'resources':result['resources']},ensure_ascii=False))
    finally:app.close()


if __name__=='__main__':main()
