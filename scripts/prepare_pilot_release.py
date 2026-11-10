import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json
from scarx.neural.language import WordLanguageCore,tokenize
from scarx.evaluation.runs import new_run,write_json

root=Path(__file__).resolve().parents[1]
source=json.loads((root/'runs/20261108T190315Z-language-vision-pilot-v2-6a889e/language-selection.json').read_text(encoding='utf-8'))['chosen']
lesson=json.loads((root/'data/dialogue/greeting-lesson.json').read_text(encoding='utf-8'))
run=new_run(root,'v02-greeting-product-lesson',{'source_checkpoint':source['checkpoint'],'repeats':16,'training_point_only':True},
 data_description='Project-authored supervised greeting lesson; not a heldout quality test',data_files=['data/dialogue/greeting-lesson.json'])
model=WordLanguageCore.load(root/source['checkpoint']);before=model.generate(lesson['prompt']);old=model.core.fingerprint()
for _ in range(16):model.train_pair(lesson['prompt'],lesson['answer'],mode='full')
after=model.generate(lesson['prompt']);destination=root/'artifacts/pilots'/run.name/'language';model.save(destination)
assert model.core.fingerprint()!=old and tokenize(after['text'])==tokenize(lesson['answer'])
assert WordLanguageCore.load(destination).generate(lesson['prompt'])==after
release=json.loads((root/'artifacts/pilot-release.json').read_text(encoding='utf-8'))
release['language']={'checkpoint':destination.relative_to(root).as_posix(),'experimental':True,'quality_gate_passed':False,
 'parent_development_exact':.1875,'parent_test_exact':.1875,'greeting_explicitly_taught':True,'heldout_after_lesson':'Not evaluated; no generalization claim',
 'lesson_run':run.relative_to(root).as_posix(),'vocabulary':len(model.vocabulary),'scope':'Eight authored families plus direct greeting correction; can blend/repeat or answer wrongly.'}
write_json(root/'artifacts/pilot-release.json',release)
result={'before':before,'after':after,'weights_changed':True,'restart_equal':True,'training_point_only':True,'quality_gate_passed':False,'checkpoint':destination.relative_to(root).as_posix()}
write_json(run/'result.json',result)
(run/'report.md').write_text('# Explicit greeting product lesson\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nThis is a directly trained example, not an unseen-prompt/general-chat result.\n',encoding='utf-8')
print(json.dumps({'checkpoint':str(destination),'before':before['text'],'after':after['text'],'quality_gate_passed':False},ensure_ascii=True,indent=2))
