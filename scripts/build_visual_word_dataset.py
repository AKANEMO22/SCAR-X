"""Actual owned image/case build, fixed splits before neural measurements."""
from pathlib import Path
import json,sys,hashlib,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.visual_word_data import scene_split,signature,qa_cases
from scarx.evaluation.vision_fixtures import render
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity


def main():
    began=time.perf_counter();info=json.loads((root/'artifacts/vision-questions-reviewed.json').read_text(encoding='utf-8'));path=(root/info['data']).resolve()
    if not path.is_relative_to(root/'data/candidates') or hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Reviewedquestion source')
    questions=json.loads(path.read_text(encoding='utf-8'))['items'];assert len(questions)==32
    run=new_run(root,'visual-word-dataset-v1',{'protocol':'experiments/visual-word-readout-v1.md',
        'display_title':'Dataset jointvisionword · 96train/96dev · chưatraining',
        'observability_kind':'data_generation','model_scope':'Owned renderer + actualAGYwording; no teacheranswers or proposal30approval.',
        'image_seeds':[33101,33103],'scene_counts':[96,96]},data_description='Synthetic geometricimages ownedbyproject, exactfixedcombinations/geometry and reviewedAGYquestions; not generalphotos.',data_files=[info['data']])
    candidate=root/'data/candidates'/run.name;candidate.mkdir(parents=True);train=scene_split('train');dev=scene_split('development');signatures={signature(s) for s in train}
    hashes={};scene_rows=[]
    for index,scene in enumerate(train+dev,1):
        raw=render(scene);digest=hashlib.sha256(raw).hexdigest()
        if digest in hashes:raise ValueError('Exactimage duplicate; stopfixedsplit, no repair')
        hashes[digest]=scene['id'];name=scene['id']+'.png';(candidate/name).write_bytes(raw)
        scene_rows.append({**scene,'file':name,'sha256':digest,'novel_composition':signature(scene) not in signatures})
        if index%24==0:progress(run,'Render/validate ảnh cómanifest',index,192,metrics={'owned_images':index})
        if time.perf_counter()-began>120:raise TimeoutError('Registered120s datasetbuildcap')
    cases=qa_cases(train,questions,signatures)+qa_cases(dev,questions,signatures)
    assert sum(r['split']=='train' for r in cases)==sum(r['split']=='development' for r in cases)==864
    data={'scenes':scene_rows,'cases':cases,'question_source':info['data'],'question_sha256':info['sha256'],
        'automatic_learning_allowed':False,'scope':'Explicit supervisedresearch only; never autoimport into live model.'}
    write_json(candidate/'dataset.json',data)
    result={'run':run.relative_to(root).as_posix(),'data':(candidate/'dataset.json').relative_to(root).as_posix(),
        'sha256':hashlib.sha256((candidate/'dataset.json').read_bytes()).hexdigest(),'decision':'VISUAL_WORD_DATASET_READY_FOR_PREREGISTERED_PILOT',
        'train_scenes':96,'development_scenes':96,'train_cases':864,'development_cases':864,
        'development_novel_composition_cases':sum(r['split']=='development' and r['novel_composition'] for r in cases),
        'exact_image_hash_overlap':0,'new_teacher_call':False,'trained':False,'proposal30_approved':False,'seconds':time.perf_counter()-began}
    (run/'report.md').write_text('# Owned jointvisualword dataset\n\nFixedcompositions/geometry/question split, all192renderedimages validated, no neuraltraining or teacheranswers.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    write_json(run/'result.json',result);write_json(root/'artifacts/visual-word-dataset-latest.json',result)
    activity(run,'Đã dựng192ảnh/1728QA cónguồnmanifest','Train/dev ảnh vàwordings tách biệt, cóheldoutcompositions; chưatraining. SourceAGYquestionsđãreview, no newteacher/labels/proposal30approval.',evidence={'train_cases':864,'development_cases':864,'image_overlap':0,'novel_dev_cases':result['development_novel_composition_cases']})
    print(json.dumps(result))


if __name__=='__main__':main()
