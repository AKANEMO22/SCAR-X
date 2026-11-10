import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.evaluation.byte_dialogue import ByteDialogue
from scarx.neural.byte_codec import UTF8Cursor
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from audit_input_information import pairs

cap=apply_memory_limit();began=time.monotonic();source=json.loads((root/'artifacts/byte-dialogue-latest.json').read_text(encoding='utf-8'))
run=new_run(root,'decoder-state-retention-audit-v1',{'protocol':'experiments/decoder-state-retention-audit-v1.md','new_pair_seed':39107,'pairs':32,'new_training':False},data_description='Fresh authored last-write counterfactuals and identical forced ASCII stream, no teacher answers used')
models=[(f'fresh-{s}',ByteDialogue(s,'byte_raw')) for s in (8101,8111,8123)]
models += [(f'trained-{r["seed"]}',ByteDialogue.loads((root/r['checkpoint']).read_bytes())) for r in source['trials'] if r['condition']=='byte_raw']
rows=[]
with ResourceMonitor(interval=.1) as monitor:
 for label,model in models:
    # Untrained models need frozen neutral statistics; raw transform ignores them.
    if not model.head.frozen:
        model.head.observe(np.zeros(48));model.head.observe(np.ones(48));model.head.freeze()
    for item in pairs(39107,32):
        a=ByteDialogue.loads(model.dumps());b=ByteDialogue.loads(model.dumps())
        for m in (a,b):m.core.reset_state()
        ca=a._prepare([{'role':'user','text':item['left']}]);cb=b._prepare([{'role':'user','text':item['right']}])
        assert np.allclose(ca,cb,atol=1e-6)
        erased=ByteDialogue.loads(a.dumps());erased.core.reset_state()
        initial=float(np.linalg.norm(a.core.features-b.core.features));trajectory=[];previous=1;cursor=UTF8Cursor()
        for position in range(128):
            for m in (a,b,erased):m.backbone.byte_step(previous,ca,position)
            if position in (0,1,4,8,16,32,64,127):
                mask=cursor.mask(128-position);pa=a.head.probabilities(a.core.features,mask);pb=b.head.probabilities(b.core.features,mask)
                pe=erased.head.probabilities(erased.core.features,mask)
                trajectory.append({'byte_position':position,'feature_equal':bool(np.array_equal(a.core.features,b.core.features)),
                  'feature_l2':float(np.linalg.norm(a.core.features-b.core.features)),'adaptation_l2':float(np.linalg.norm(a.core.adaptation-b.core.adaptation)),
                  'probability_l1':float(np.abs(pa-pb).sum()),'erased_probability_l1':float(np.abs(pa-pe).sum())})
            previous=ord('ab .'[position%4])+2;cursor.accept(previous)
        rows.append({'model':label,'pair_id':item['id'],'encoding_end_feature_l2':initial,'trajectory':trajectory})
        monitor.check()
        if time.monotonic()-began>60:raise TimeoutError('Registered60s audit budget')
 summary={}
 for label,_ in models:
    sample=[r for r in rows if r['model']==label];summary[label]={'initial_mean_feature_l2':float(np.mean([r['encoding_end_feature_l2'] for r in sample])),'positions':{}}
    for position in (0,1,4,8,16,32,64,127):
        values=[t for r in sample for t in r['trajectory'] if t['byte_position']==position]
        summary[label]['positions'][str(position)]={'exact_collapse_fraction':sum(v['feature_equal'] for v in values)/len(values),
          'mean_feature_l2':float(np.mean([v['feature_l2'] for v in values])),'mean_probability_l1':float(np.mean([v['probability_l1'] for v in values])),
          'mean_erased_probability_l1':float(np.mean([v['erased_probability_l1'] for v in values]))}
 result={'run':run.relative_to(root).as_posix(),'summary':summary,'resources':monitor.report(),'cap':cap,'new_training':False,'deployed':False,'task_quality_proven':False}
 write_json(run/'result.json',result);write_json(run/'trajectories.json',rows);write_json(root/'artifacts/decoder-state-retention-audit.json',result)
 (run/'report.md').write_text('# Decoder state retention\n\nMeasured causal distinguishability under matched forced bytes. Activation persistence is not acquired knowledge or correct task performance.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'run':str(run),'summary':summary}),flush=True)
