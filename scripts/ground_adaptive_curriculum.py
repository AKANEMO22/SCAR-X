"""Data-only reference/image grounding, plus compatibility/information audit."""
from pathlib import Path
import os,sys,json,time,hashlib,io,zipfile,urllib.request
from collections import Counter,defaultdict
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from PIL import Image,ImageDraw
from scarx.evaluation.adaptive_grounding import source_receipts,scene_bank,pixel_objects,eligible,semantic_value,reference_sentence,distribution_ceiling,chat_examples,model_input
from scarx.evaluation.vision_fixtures import render,image_uri
from scarx.neural.language import tokenize
from scarx.app.pilots import PilotLab
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity


def metadata_only(session):
    """Never instantiate a model; never publish private vocabulary/fingerprints."""
    if len(session)>3_000_000:raise ValueError('Session bound')
    with zipfile.ZipFile(io.BytesIO(session)) as z:
        neural=z.read('model.zip');meta=json.loads(z.read('session.json'))
    if hashlib.sha256(neural).hexdigest()!=meta['sha256']:raise ValueError('Session checksum')
    return neural_metadata(neural)


def neural_metadata(payload):
    if len(payload)>2_000_000:raise ValueError('Neuralpayload bound')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        meta=json.loads(z.read('metadata.json'));core=z.read('core.npz')
    vocabulary=meta['vocabulary']
    if not isinstance(vocabulary,list) or not 3<=len(vocabulary)<=256 or any(not isinstance(t,str) for t in vocabulary):raise ValueError('Vocabulary metadata')
    digest=hashlib.sha256()
    with zipfile.ZipFile(io.BytesIO(core)) as z:
        for name in ('w_in','w_rec','w_out','bias','stable_in','stable_rec','stable_out','stable_bias','indices'):
            raw=z.read(name+'.npy');digest.update(name.encode());digest.update(raw)
    return set(vocabulary),digest.hexdigest()


def live_metadata():
    try:return metadata_only(urllib.request.urlopen('http://127.0.0.1:8765/api/export',timeout=15).read())
    except (OSError,ValueError,KeyError,zipfile.BadZipFile):return None,None


def vocabulary_audit(cases,known):
    tokens=[tokenize(r['reference_answer']) for r in cases];all_tokens=set(t for ts in tokens for t in ts)
    result={'references':len(cases),'distinct_reference_tokens':len(all_tokens),'max_reference_tokens':max(map(len,tokens)),'longer_than_24':sum(len(ts)>24 for ts in tokens),'vocabulary_cap_in_current_code':256,'target_token_limit_in_current_code':24,'scope':'Lexical/length compatibility only, not proof of learnability or API modality support'}
    if known is None:result['metadata_available']=False;return result
    result.update(metadata_available=True,existing_vocabulary_size=len(known),needed_vocabulary_size=len(known|all_tokens),exceeds_256=len(known|all_tokens)>256,fully_known_reference_count=sum(set(ts)<=known for ts in tokens))
    return result


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    started=time.perf_counter();cap=apply_memory_limit();info=json.loads((root/'artifacts/adaptive-curriculum-reviewed.json').read_text(encoding='utf-8'));source=(root/info['data']).resolve()
    if not source.is_relative_to(root/'data/candidates') or hashlib.sha256(source.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Reviewedwording sha')
    words=json.loads(source.read_text(encoding='utf-8'));assert words['all_rows_read'] and len(words['items'])==127
    labels='data/fixtures/adaptive_chat_references_v1.json';references=json.loads((root/labels).read_text(encoding='utf-8'));sources=source_receipts(root)
    code=(root/'src/scarx/neural/unified.py').read_text(encoding='utf-8');assert 'len(vocabulary) <= 256' in code and 'len(tokens) <= 24' in code
    run=new_run(root,'adaptive-curriculum-grounding-v1',{'protocol':'experiments/adaptive-curriculum-grounding-v1.md','display_title':'Ground data ·95chat/576imageQA · không train','observability_kind':'data_generation','model_scope':'Development references/pixeloracle only. No neural inference/training, U/P2pilot/mainchange or newteacher call.','image_seed':43301,'images':36,'chat_references':95,'vision_qa':576,'new_training':False},data_description='ActualAGYwordings41 + projectauthored source-backed references + ownedRGB/pixeloracle; no privateweights or teacheranswers.',data_files=[info['data'],labels]+sorted({v['path'] for v in sources.values()}))
    directory=root/'data/candidates'/run.name;(directory/'images').mkdir(parents=True)
    (run/'report.md').write_text('# Ground developmentdata\n\nBuilding offline references, not training or pilot.\n',encoding='utf-8')
    activity(run,'Đang hoàn thiện reference và nhãn ảnh','Dùng wording AGY đã review, đáp án chat theo nguồn dự án, ảnh do renderer tạo. Không chạy learner hoặc nạp data vào Hermes.',status='started',evidence={'protocol':'experiments/adaptive-curriculum-grounding-v1.md','new_training':False})
    initial_live,initial_weights=live_metadata();error=None;result=None
    with ResourceMonitor(interval=.05) as monitor:
        try:
            chat=chat_examples(words['items'],references);questions=[r for r in words['items'] if r['modality']=='vision'];assert len(questions)==32
            scenes=[];cases=[];seen=set();raw_images=[];legacy_pixels=[]
            for index,scene in enumerate(scene_bank(),1):
                raw=render(scene);digest=hashlib.sha256(raw).hexdigest()
                if digest in seen:raise ValueError('Duplicate owned image, no automatic reseed/repair')
                seen.add(digest);objects=pixel_objects(raw)
                if objects!=sorted(scene['objects'],key=lambda o:o['x']):raise ValueError('Independent pixeloracle/manifest disagreement')
                relative='images/'+digest[:16]+'.png';(directory/relative).write_bytes(raw)
                pixels,_=PilotLab.pixels(image_uri(raw));encoded=hashlib.sha256(pixels.tobytes()).hexdigest()
                stored={**scene,'file':relative,'sha256':digest,'legacy64_sha256':encoded,'pixel_oracle_matches_manifest':True};scenes.append(stored);raw_images.append(raw);legacy_pixels.append(pixels)
                for question in questions:
                    kind=question['family'];language=question['language']
                    if not eligible(objects,kind):continue
                    value=semantic_value(objects,kind,language);assert value==semantic_value(scene['objects'],kind,language)
                    case={'id':scene['id']+'/'+question['id'],'input':{'prompt':question['prompt'],'image_file':relative},'image_sha256':digest,'legacy64_sha256':encoded,'scene_id':scene['id'],'question_id':question['id'],'kind':kind,'language':language,'semantic_target':value,'reference_answer':reference_sentence(value,kind,language),'bag_of_objects':sorted((o['shape'],o['color']) for o in objects),'label_basis':'Ownedrenderer manifest independently checked against raster; not modelanswer','automatic_learning_allowed':False}
                    visible=model_input(case,directory);assert set(visible)=={'prompt','image_bytes'} and visible['image_bytes']==raw;cases.append(case)
                progress(run,'Đã kiểm pixel/nhãn ảnh',index,36,metrics={'owned_images':index,'qa_written':len(cases),'main_updates':0});monitor.check()
                if time.perf_counter()-started>120:raise TimeoutError('Grounding120s cap')
            assert len(chat)==95 and len(cases)==576
            family={kind:[r for r in cases if r['kind']==kind] for kind in ('single_color','single_shape','object_count','left_color')}
            question_only={k:distribution_ceiling(v,lambda r:r['input']['prompt']) for k,v in family.items()}
            encoded_ceiling={k:distribution_ceiling(v,lambda r:(r['input']['prompt'],r['legacy64_sha256'])) for k,v in family.items()}
            left_bag=distribution_ceiling(family['left_color'],lambda r:(r['input']['prompt'],tuple(map(tuple,r['bag_of_objects']))))
            bygroup=defaultdict(list)
            for scene in scenes:bygroup[scene['counterfactual_group']].append(scene)
            equal_groups=sum(len({s['legacy64_sha256'] for s in group})==1 for group in bygroup.values())
            release=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'));raw=(root/release['checkpoint']).read_bytes();assert hashlib.sha256(raw).hexdigest()==release['sha256'];public_vocab,_=neural_metadata(raw)
            audit={'chat_public_release':vocabulary_audit(chat,public_vocab),'vision_public_release':vocabulary_audit(cases,public_vocab),'combined_public_release':vocabulary_audit(chat+cases,public_vocab),'combined_live_aggregate':vocabulary_audit(chat+cases,initial_live),'private_vocabulary_or_fingerprint_published':False,'current_vision_api_is_digits_only':True}
            metadata={'format':'scarx-grounded-development-v1','wording_source':info,'sources':sources,'chat':chat,'scenes':scenes,'vision':cases,'input_annotation_separation_checked':True,'automatic_learning_allowed':False,'all_development':True,'new_training':False,'limits':['Source-backed projectreferences are not generalfactverification; greetings are style.','36controlledgeometricimages are not naturalphotos/screenshots/OCR.','References are not modelanswers or knowledge already in synapses.','No finaltest or newquality score.']}
            path=directory/'dataset.json';write_json(path,metadata);write_json(run/'compatibility.json',audit)
            # A preview for human inspection, never an image input or traincase.
            preview=Image.new('RGB',(6*100,6*122),'#eeeeee');draw=ImageDraw.Draw(preview)
            for i,raw in enumerate(raw_images):
                with Image.open(io.BytesIO(raw)) as image:preview.paste(image.resize((96,96)),((i%6)*100+2,(i//6)*122))
                draw.text(((i%6)*100+2,(i//6)*122+99),scenes[i]['id'],fill='black')
            preview.save(root/'artifacts/adaptive-grounding-preview.png')
            pair=Image.new('RGB',(560,190),'white');draw=ImageDraw.Draw(pair)
            for index in range(2):
                with Image.open(io.BytesIO(raw_images[index])) as image:pair.paste(image,(index*140,35))
                gray=Image.fromarray(np.rint(legacy_pixels[index]/16*255).clip(0,255).astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST).convert('RGB');pair.paste(gray,((index+2)*140,35))
            for x,label in enumerate(('RGB A','RGB B','Actual64 A','Actual64 B')):draw.text((x*140+3,15),label,fill='black')
            draw.text((3,172),'Changed color, same question. Equal64: '+str(np.array_equal(*legacy_pixels[:2])),fill='black');pair.save(root/'artifacts/adaptive-grounding-color-loss.png')
            final_live,final_weights=live_metadata()
            result={'run':run.relative_to(root).as_posix(),'decision':'DATA_GROUNDED_NOT_TRAINING_READY','data':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'chat_references':95,'owned_images':36,'vision_qa':576,'independent_pixel_checks':36,'pixel_manifest_disagreements':0,'question_only_ceiling_by_kind':question_only,'legacy64_conditional_ceiling_by_kind':encoded_ceiling,'left_bag_of_objects_ceiling':left_bag,'counterfactual_groups':len(bygroup),'legacy64_equal_color_groups':equal_groups,'compatibility':audit,'preview':'artifacts/adaptive-grounding-preview.png','color_loss_preview':'artifacts/adaptive-grounding-color-loss.png','live_metadata_available':initial_live is not None and final_live is not None,'live_weights_unchanged_during_build':initial_weights==final_weights if initial_weights is not None and final_weights is not None else None,'automatic_learning_allowed':False,'pilot_run':False,'main_training':False,'main_restored':False,'neural_instances_created':0,'normal_chat_or_vision_proven':False,'new_teacher_or_qwen_call':False,'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-started,'error':None}
            monitor.check()
        except Exception as exc:error=type(exc).__name__+': '+str(exc)[:200]
    if result is None:result={'run':run.relative_to(root).as_posix(),'decision':'DATA_GROUNDING_FAILED','error':error,'main_training':False,'pilot_run':False,'automatic_learning_allowed':False}
    if error:result.update(decision='DATA_GROUNDING_FAILED',error=error)
    write_json(run/'result.json',result);write_json(root/'artifacts/adaptive-curriculum-grounding.json',result)
    (run/'report.md').write_text('# Grounded developmentreferences\n\nOffline data/reference/encoder audit only. No modellearning or normalvision claim.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    activity(run,'Hoàn tất kiểm data' if error is None else 'Data chưa qua kiểm tra',result['decision']+'; không huấn luyện, không nạp model hoặc tự mở pilot.',status='completed' if error is None else 'failed',evidence={'receipt':'artifacts/adaptive-curriculum-grounding.json','error':error,'main_training':False})
    print(json.dumps({k:result.get(k) for k in ('run','decision','chat_references','owned_images','vision_qa','legacy64_equal_color_groups','seconds','error')}))
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
