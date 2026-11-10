"""Record actual editorial review, not automatic factual verification."""
from pathlib import Path
import sys,json,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from generate_vision_questions import KINDS
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.work_trace import activity


def main():
    info=json.loads((root/'artifacts/antigravity-vision-questions-latest.json').read_text(encoding='utf-8'));path=(root/info['data']).resolve()
    if not path.is_relative_to(root/'data/candidates'):raise ValueError('Question sourcepath')
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==info['sha256'];items=json.loads(raw)['items']
    assert len(items)==32 and {r['id'] for r in items}=={f'{k}_{i:02d}' for k in KINDS for i in range(1,9)}
    assert all(len(r['question'].split())<=16 for r in items)
    # Editorial decision made after reading EVERY emitted row, not inferred from
    # these count/schema checks. The archive retains the exact original wording.
    notes={'single_shape_01':'Vietnamese slightly stiff; asks the sole geometricshape unambiguously under fixture.',
        'single_shape_04':'Vietnamese slightly stiff, same single-shape attribute.',
        'left_color_04':'Interpret outer-left edge phrasing as uniqueleftmostobject in the controlledfixture; not a broadsemantic test.'}
    run=new_run(root,'vision-question-semantic-review-v1',{'protocol':'experiments/vision-question-path-audit-v1.md',
        'display_title':'Rà soát32câu hỏi ảnh · nhãn từrenderer, khôngteacher',
        'observability_kind':'data_generation','model_scope':'Editorial publicwording review only; no neuraltraining/proposal30approval.',
        'source_run':info['run'],'items':32},data_description='All32actualCLIquestions read; geometricquestionkinds reviewed, no teacheranswers.',data_files=[info['data']])
    reviewed={'items':items,'editorial_notes':notes,'source_generation_run':info['run'],'automatic_learning_allowed':False,
        'semantic_review_pending':False,'review_scope':'Controlled geometricquestion types only; not a guarantee of naturallanguage quality/visionperformance.'}
    destination=path.parent/'reviewed-questions.json';write_json(destination,reviewed)
    receipt={'run':run.relative_to(root).as_posix(),'source_generation_run':info['run'],'data':destination.relative_to(root).as_posix(),
        'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'items':32,'decision':'QUESTIONS_REVIEWED_FOR_PATH_AUDIT_ONLY',
        'editorial_notes':notes,'automatic_learning_allowed':False,'trained':False,'teacher_answers_used':False,'proposal30_approved':False}
    (run/'report.md').write_text('# Public visual-question editorial review\n\nAll32actualCLIquestions read; attributeintent matches controlledfixtures. Threewording caveats retained. No targettruthcertification or trainingimport.\n\n'+json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    write_json(run/'result.json',receipt);write_json(root/'artifacts/vision-questions-reviewed.json',receipt)
    activity(run,'Đã ràsoát32câu hỏi ảnh','Giữ câu chữ gốc và3editorialcaveats. Chỉ đủchoauditfixturehình học; chưatraining, khôngteacheranswer hoặc mở proposal30.',evidence={'reviewed':32,'sha256':receipt['sha256'],'trained':False})
    print(json.dumps({'reviewed':32,'editorial_caveats':len(notes),'trained':False}))


if __name__=='__main__':main()
