"""Candidate encoder counterexamples, not a model competence benchmark."""
from pathlib import Path
import os,sys,json,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.input_events import InputEvents
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from audit_input_information import pairs as fresh_pairs

cap=apply_memory_limit();source=json.loads((root/'artifacts/p1-input-information-audit.json').read_text(encoding='utf-8'))
audit=root/source['run'];pairs=json.loads((audit/'text-pairs.json').read_text(encoding='utf-8'))
pairs=[{**r,'split':'regression_already_viewed'} for r in pairs]
pairs += [{**r,'split':'fresh_confirmation'} for r in fresh_pairs(29107,256)]
release=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'));model=UnifiedCore.loads((root/release['checkpoint']).read_bytes())
enc=InputEvents(len(model.vocabulary))
assert enc.dimensions==model.core.config.inputs
run=new_run(root,'p1-input-events-v2',{'protocol':'experiments/p1-input-events-v2.md','source_counterexamples':source['run'],'fresh_seed':29107,'training_updates':0},data_description='Previously viewed regression counterexamples plus256fresh pairs seed29107; no labels used for fitting')
with ResourceMonitor(interval=.1) as monitor:
    rows=[]
    for item in pairs:
        a,b=enc.text(item['left']),enc.text(item['right'])
        rows.append({'id':item['id'],'split':item['split'],'events':len(a),'sequence_equal':bool(np.array_equal(a,b)),
          'sequence_l2_difference':float(np.linalg.norm(a-b)),'sum_equal':bool(np.allclose(a.sum(0),b.sum(0),atol=1e-5)),
          'reserved_output_input_channels_zero':bool(np.all(a[:,:enc.word_channels]==0))})
    red=enc.image((audit/'red.png').read_bytes());green=enc.image((audit/'green.png').read_bytes())
    result={'run':run.relative_to(root).as_posix(),'pairs':len(rows),'ordered_collisions':sum(r['sequence_equal'] for r in rows),
      'pooled_collisions':sum(r['sum_equal'] for r in rows),'minimum_sequence_difference':min(r['sequence_l2_difference'] for r in rows),
      'fresh_confirmation_collisions':sum(r['sequence_equal'] for r in rows if r['split']=='fresh_confirmation'),
      'fresh_confirmation_pairs':sum(r['split']=='fresh_confirmation' for r in rows),
      'color_events_equal':bool(np.array_equal(red,green)),'color_l2_difference':float(np.linalg.norm(red-green)),
      'same_channel_layout_as_g1':True,'input_channels':enc.dimensions,'no_target_features':all(r['reserved_output_input_channels_zero'] for r in rows),
      'text_maximum_events':enc.maximum_events,'image_events':len(red),'resources':monitor.report(),'cap':cap,
      'deployed':False,'training_updates':0,'normal_chat_or_vision_proven':False,'interpretation':'Chronological input preserves the tested order/color distinctions. The learner still must use them; pooled decoding would discard order again.'}
    write_json(run/'text-results.json',rows);write_json(run/'result.json',result);write_json(root/'artifacts/p1-input-events-latest.json',result)
    (run/'report.md').write_text('# Ordered/RGB input events\n\n'+json.dumps(result,indent=2)+'\n\nThis fixes a tested information-loss precondition in an isolated frontend, not neural task learning. No live weights changed.\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
