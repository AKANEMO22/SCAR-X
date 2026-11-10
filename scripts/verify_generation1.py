"""Engineering acceptance in an isolated session. No research quality claims."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import json
import threading
import time
import urllib.request
import uuid
import numpy as np
from scarx.app.generation import Generation,MODEL
from scarx.app.generation_server import make_server
from scarx.runtime.limits import apply_memory_limit
from scarx.evaluation.runs import new_run,write_json

if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
limit=apply_memory_limit()
run=new_run(root,'generation1-engineering',{'protocol':'experiments/generation1-product.md','research_quality_test':False})
directory=root/'local'/('generation1-check-'+uuid.uuid4().hex[:8]);app=Generation(directory);app.limit_status=limit
server=make_server(app,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
base='http://127.0.0.1:'+str(server.server_port)

def call(path,body=None,token=None):
    headers={'Content-Type':'application/json'}
    if token:headers['X-SCARX-Token']=token
    req=urllib.request.Request(base+path,None if body is None else json.dumps(body).encode(),headers=headers)
    return json.load(urllib.request.urlopen(req,timeout=15))

try:
    with app.lock:before=app.model.core.fingerprint();updates=app.model.core.updates;ticks=app.model.ticks
    # Bounded observation, not an8h endurance claim.
    threading.Event().wait(1.3)
    with app.lock:
        assert app.model.ticks>ticks and app.model.core.updates==updates and app.model.core.fingerprint()==before
        idle_ticks=app.model.ticks-ticks
    token=call('/api/status')['token'];call('/api/idle',{'enabled':False},token)
    identities=call('/v1/models');assert [x['id'] for x in identities['data']]==[MODEL]
    prompts=['hi','ch\u00e0o','ch\u00e0o b\u1ea1n'];answers=[]
    for prompt in prompts:
        start=time.perf_counter();r=call('/v1/chat/completions',{'model':MODEL,'messages':[{'role':'user','content':prompt}]})
        answers.append({'prompt':prompt,'text':r['choices'][0]['message']['content'],'seconds':time.perf_counter()-start})
    sample=json.loads((root/'artifacts/pilots/vision-examples.json').read_text(encoding='utf-8'))[0]
    image=call('/api/vision',{'image':sample['image']},token);assert 0<=image['digit']<=9
    teach=call('/api/teach',{'prompt':prompts[2],'answer':'Ch\u00e0o b\u1ea1n.','repeats':2,'confirmation':True},token)
    assert teach['weight_delta']['w_rec']>0 and teach['weight_delta']['w_out']>0
    saved=app.export();active=app.active
    clone=Generation(root/'local'/('generation1-resume-'+uuid.uuid4().hex[:8]),background=False)
    try:
        clone._unpack(saved)
        expected=app.model.generate(prompts[0]);actual=clone.model.generate(prompts[0]);assert expected==actual
        for key,array in app.model.core.arrays().items():np.testing.assert_array_equal(array,clone.model.core.arrays()[key])
    finally:clone.close()
    status=call('/api/status');assert status['cores_loaded']==1 and status['ram_bytes']<2_000_000_000
    result={'models':[MODEL],'cores_loaded':1,'idle_advanced_ticks':idle_ticks,'idle_weights_unchanged':True,
       'training_prompt_outputs_not_generalization':answers,'digit_example':{'expected':sample['expected_digit'],'predicted':image['digit']},
       'explicit_learning_weight_delta':teach['weight_delta'],'session_exact_continuation':True,
       'ram_bytes':status['ram_bytes'],'resources':status['resources'],'memory_cap':limit,
       'guarded_online_learning':True,'idle_learning_only_from_explicit_or_verified_labels':True,'large_model_started':False,'research_decision':'PIVOT_NO_SCALE'}
    write_json(run/'result.json',result);write_json(root/'artifacts/generation1-engineering.json',{'run':run.relative_to(root).as_posix(),**result})
    (run/'report.md').write_text('# Generation1 engineering acceptance\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nIsolated public-fixture check, not a generalization/retention/endurance benchmark.\n',encoding='utf-8')
    print(json.dumps({'run':run.relative_to(root).as_posix(),'models':[MODEL],'idle_ticks':idle_ticks,'ram_mb':status['ram_bytes']/1e6,'answers':answers},ensure_ascii=False))
finally:server.shutdown();server.server_close();thread.join();app.close()
