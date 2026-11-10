"""Backprop controls live only in evaluation/controls.py. Never deploy artifacts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.runtime.limits import apply_memory_limit
cap=apply_memory_limit()
import numpy as np
import torch
from scarx.evaluation.byte_dialogue import ByteDialogue
from scarx.evaluation.controls import ByteBPTTControl
from scarx.runtime.resources import ResourceMonitor
from scarx.evaluation.runs import new_run,write_json

torch.set_num_threads(1)
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
data=root/'data/candidates/antigravity-events-curated-20261109T094142Z';records=[json.loads(l) for l in (data/'experiences.jsonl').read_text(encoding='utf-8').splitlines()]
train=[r for r in records if r['split']=='train'];dev=[r for r in records if r['split']=='development']
run=new_run(root,'byte-bptt-learnability-v1',{'protocol':'experiments/byte-bptt-learnability-control-v1.md','seeds':[7103,7121,7133],'passes':4,'device':'cpu','control_only':True},data_description='Same public96train/24devdialogue prefixes; isolated Adam/BPTT controls, no private data')
artifact=root/'artifacts/byte-bptt-control'/run.name;artifact.mkdir(parents=True);results=[];began=time.monotonic()
with ResourceMonitor(interval=.1) as monitor:
  for seed in (7103,7121,7133):
    base=ByteDialogue(seed,'byte_normalized_full');base.calibrate(train)
    for frozen in (True,False):
        condition='Adam_readout' if frozen else 'Adam_BPTT';model=ByteBPTTControl(base,hidden_frozen=frozen);rng=np.random.default_rng(seed);t0=time.monotonic()
        # Check the exact input/output seam before any optimizer updates.
        a=ByteDialogue.loads(base.dumps());context=a._prepare(train[0]['messages']);a.backbone.byte_step(1,context,0)
        with torch.no_grad():
            context2=model.prepare(train[0]['messages']);model.byte_step(1,context2,0)
            np.testing.assert_allclose(model.state[3].numpy(),a.core.features,atol=1e-6,rtol=1e-6)
        model=ByteBPTTControl(base,hidden_frozen=frozen)
        initial=model.score(dev);initial_train=model.score(train);stages=[]
        grouped={key:[r for r in train if r['dialogue_id']==key] for key in sorted({r['dialogue_id'] for r in train})}
        for epoch in range(1,5):
            losses=[]
            for key in rng.permutation(list(grouped)):
                for row in grouped[key]:
                    losses.append(model.learn(row['messages'],row['target']))
                    if time.monotonic()-t0>180:raise TimeoutError('Registered180s percontrol limit')
                    if time.monotonic()-began>1200:raise TimeoutError('Registered1200s round limit')
            monitor.check()
            if epoch in (2,4):
                score=model.score(dev);stage={'pass':epoch,'train_loss':float(np.mean(losses)),'development':score}
                stages.append(stage);write_json(run/f'{condition}-{seed}-pass-{epoch}.json',stage)
                print(json.dumps({'condition':condition,'seed':seed,'pass':epoch,'dev_exact':score['exact'],'forced_byte_accuracy':score['teacher_forced_byte_accuracy'],'seconds':time.monotonic()-t0}),flush=True)
        final_train=model.score(train);path=artifact/f'{condition}-{seed}.control.npz';model.export_arrays(path)
        final={'condition':condition,'seed':seed,'initial':initial,'initial_train':initial_train,'stages':stages,'final_train':final_train,'optimizer_steps':model.updates,
          'byte_steps_train':model.byte_steps,'seconds':time.monotonic()-t0,'forward_preflight_matches':True,'artifact':path.relative_to(root).as_posix(),
          'limited_train_learnability_evidence':final_train['teacher_forced_byte_accuracy']>=.65 and final_train['byte_nll']<initial_train['byte_nll']}
        results.append(final);write_json(run/'progress.json',results)
  result={'run':run.relative_to(root).as_posix(),'trials':results,'resources':monitor.report(),'cap':cap,'control_only':True,'cuda_allocated':False,
    'weights_imported_into_scarx':False,'main_local_learning_success':False,'normal_chat_or_vision_proven':False,
    'scope':'Different optimizer/all-synapse backward vs local per-byte updates, not matched compute/updates; negative result inconclusive under limited budget'}
  write_json(run/'result.json',result);write_json(root/'artifacts/byte-bptt-control-latest.json',result)
  (run/'report.md').write_text('# Isolated byte learnability controls\n\nThese weights cannot be deployed as SCAR-X local-learning evidence. Same forward model/input but different optimizer, temporal gradients and step aggregation.\n\n'+json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2)+'\n',encoding='utf-8')
  print('RESULT '+str(run),flush=True)
