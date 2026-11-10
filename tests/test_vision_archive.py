import hashlib,json,io
from pathlib import Path
from scarx.app.observatory import vision_answer_archive
from scarx.runtime.experiment_status import vision_path_statistics
from scarx.evaluation.vision_fixtures import scenes,render,image_uri
from scarx.neural.unified import UnifiedCore
from scarx.app.pilots import PilotLab

ROOT=Path(__file__).resolve().parents[1]


def test_owned_vision_archive_exposes_actual_pixels_and_readout_without_private_fields(tmp_path):
    run=tmp_path/'runs'/'visionfixture';run.mkdir(parents=True);candidate=tmp_path/'data/candidates/visionfixture';candidate.mkdir(parents=True)
    raw=render(scenes()[0]);(candidate/'s01.png').write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
    case={'id':'s01/q','prompt':'What color is the shape?','target':'red','image_sha256':sha,'scene_id':'s01'}
    data={'cases':[case],'scenes':[{'id':'s01','file':'s01.png','sha256':sha}],'automatic_learning_allowed':False}
    fixture=candidate/'qa-fixtures.json';fixture.write_text(json.dumps(data),encoding='utf-8')
    release=json.loads((ROOT/'artifacts/generation1-release.json').read_text(encoding='utf-8'));m=UnifiedCore.loads((ROOT/release['checkpoint']).read_bytes())
    pixels,_=PilotLab.pixels(image_uri(raw));reply=m.image(pixels)
    output={**case,'digit':reply['digit'],'probabilities':reply['probabilities'],'private':'never expose'}
    (run/'outputs.json').write_text(json.dumps([output]),encoding='utf-8')
    result={'decision':'VISION_QUESTION_DROPPED_BEFORE_NEURAL_CORE','new_training':False,'data':fixture.relative_to(tmp_path).as_posix(),
        'data_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'reason_missing_Qwen_vision':'Text-only reference, not VQA.',
        'by_scene':{'s01':{'cases':1,'distinct_renderer_targets':1,'all_questions_same_neural_input':True,'all_questions_same_neural_probabilities':True,'prototype_distinct_multimodal_sequences':1,'private':'omit'}}}
    (run/'result.json').write_text(json.dumps(result),encoding='utf-8')
    archive=vision_answer_archive(tmp_path,'visionfixture');assert archive['case_count']==1 and archive['learning_mechanism']=='vision_path_audit'
    row=archive['rows'][0];assert row['output_type']=='digit_classifier' and row['qwen'] is None and json.loads(row['generated'])['digit']==reply['digit']
    assert row['image_data_uri'].startswith('data:image/png;base64,') and row['encoded_image_data_uri']!=row['image_data_uri']
    assert 'private' not in json.dumps(archive) and vision_path_statistics(tmp_path,'visionfixture')[0]['all_questions_same_neural_input'] is True
    (candidate/'s01.png').write_bytes(b'not the source image');assert vision_answer_archive(tmp_path,'visionfixture')=={}
    assert not vision_path_statistics(tmp_path,'../outside') and not vision_answer_archive(tmp_path,'../outside')
