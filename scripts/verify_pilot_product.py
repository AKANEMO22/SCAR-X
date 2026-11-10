"""Measure opt-in labs on an isolated data directory, not user's active state."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json
import uuid
from scarx.app.state import Workbench
from scarx.app.openai_backend import chat
from scarx.app.pilots import LANGUAGE,VISION
from scarx.evaluation.runs import new_run,write_json

root=Path(__file__).resolve().parents[1]
run=new_run(root,'v02-pilot-product-check',{'scope':'Experimental feature/persistence check, no research-gate relaxation'},
 data_description='Trained v2checkpoints, project-authored correction and eight exported UCI digit diagnostic fixtures')
app=Workbench(root/'local'/('pilot-product-'+uuid.uuid4().hex[:8]))
try:
    base=app._pack();pilot=app.pilots
    before=pilot.language('chào bạn');initial=pilot.checkpoints()[0]['id']
    taught=pilot.teach_language({'prompt':'chào bạn','answer':'chào bạn, tôi là scar-x.','confirmation':True,'repeats':16})
    after=pilot.language('chào bạn');state=pilot._load('language').dumps()
    assert base==app._pack()
    directory=app.directory;app.close();app=Workbench(directory)
    assert app.pilots._load('language').dumps()==state and app.pilots.language('chào bạn')['text']==after['text']
    app.pilots.restore(initial);assert app.pilots.language('chào bạn')['text']==before['text']
    examples=app.pilots.examples();images=[]
    for item in examples:
        result=app.pilots.vision({'image':item['image']})
        images.append({'index':item['index'],'expected':item['expected_digit'],'predicted':result['digit']})
    completion,_=chat(app,{'model':LANGUAGE,'messages':[{'role':'user','content':'chào bạn'}]})
    assert completion['choices'][0]['message']['content']==before['text']
    vision,_=chat(app,{'model':VISION,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':examples[0]['image']}}]}]})
    assert json.loads(vision['choices'][0]['message']['content'])['digit']==images[0]['predicted']
    result={'before':before,'manual_teach':taught,'after':after,'restart_equal':True,'restore_equal':True,'v01_model_unchanged':True,
            'image_fixtures':images,'neural_hermes_language_and_image_transport':True,'resources':app.resources(),
            'limitations':'Language quality gate failed; one trained correction is not heldout generalization. Digits-only vision; screenshot/visualQA not implemented.'}
    write_json(run/'result.json',result)
    (run/'report.md').write_text('# V0.2 experimental product check\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    write_json(root/'artifacts/pilot-product-check.json',{'run':run.relative_to(root).as_posix(),'experimental':True,'research_quality_gate':False})
    print(json.dumps({'run':str(run),'before':before['text'],'after':after['text'],'restart':True,'images':images},ensure_ascii=True,indent=2))
finally:app.close()
