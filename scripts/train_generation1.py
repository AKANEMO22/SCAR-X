"""Bounded joint training of ONE fixed48 ALIF core; protocol generation1-product."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'src'))
import json
import time
import hashlib
import numpy as np
import psutil
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scarx.neural.unified import UnifiedCore
from scarx.neural.language import tokenize
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.evaluation.runs import new_run,write_json

if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
if os.name=='nt': psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
limit=apply_memory_limit()
corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
train=[(p,f['answer']) for f in corpus['families'] for p in f['train']]
dev=[(p,f['answer']) for f in corpus['families'] for p in f['development']]
vocab=['<eos>','<bos>']+sorted({t for _,a in train for t in tokenize(a)})
digits=load_digits()
training,rest=train_test_split(np.arange(len(digits.target)),test_size=.4,random_state=2701,stratify=digits.target)
development,testing=train_test_split(rest,test_size=.5,random_state=2703,stratify=digits.target[rest])
run=new_run(root,'generation1-joint',{'protocol':'experiments/generation1-product.md','seeds':[17,29],'hidden':48,'epochs':100,'seconds_per_seed':600},
 data_description='Authored tiny dialogue and sklearn UCI digits; joint shared ALIF; no external model.',data_files=['data/dialogue/generation1.json'])
artifact=root/'artifacts/generation1'/run.name; artifact.mkdir(parents=True)
write_json(run/'vision-split.json',{'training':training.tolist(),'development':development.tolist(),'diagnostic_test':testing.tolist()})

def evaluate(model,pairs,indices):
    # Independent diagnostic copy; no interaction with live background state.
    clone=UnifiedCore.loads(model.dumps()); outputs=[]
    for prompt,answer in pairs:
        result=clone.generate(prompt)
        outputs.append({'prompt':prompt,'answer':answer,'generated':result['text'],'exact':tokenize(result['text'])==tokenize(answer)})
    accuracy=float(np.mean([clone.image(digits.data[i])['digit']==int(digits.target[i]) for i in indices]))
    return {'exact':float(np.mean([r['exact'] for r in outputs])),'digits':accuracy,'outputs':outputs}

results=[]
with ResourceMonitor(interval=.5) as monitor:
    for seed in (17,29):
        model=UnifiedCore(vocab,seed); rng=np.random.default_rng(seed); best=(-1.,-1.); chosen=None; began=time.monotonic()
        for epoch in range(1,101):
            losses=[]
            order=rng.permutation(training)
            # Interleave modalities, no independent core/checkpoint per modality.
            for step,i in enumerate(order):
                model.image(digits.data[i],int(digits.target[i]))
                if step%3==0:
                    prompt,answer=train[int(rng.integers(len(train)))]
                    losses.append(model.train_pair(prompt,answer))
                if step%128==0:
                    monitor.check()
                    if time.monotonic()-began>600: break
            score=evaluate(model,dev,development)
            rank=((score['exact']+score['digits'])/2,score['exact'])
            record={'seed':seed,'epoch':epoch,'dev_exact':score['exact'],'dev_digits':score['digits'],'train_text_loss':float(np.mean(losses)),'elapsed':time.monotonic()-began}
            with (run/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            if epoch==1 or epoch%5==0:print(json.dumps(record),flush=True)
            if rank>best:
                best=rank; chosen=artifact/f'seed-{seed}.checkpoint.zip'; chosen.write_bytes(model.dumps())
                write_json(artifact/f'seed-{seed}-selection.json',record)
            if time.monotonic()-began>600:break
        selected=UnifiedCore.loads(chosen.read_bytes())
        selection=json.loads((artifact/f'seed-{seed}-selection.json').read_text())
        # Acceptance opened once, after final checkpoint selection for this seed.
        pairs=[(p,f['answer']) for f in corpus['families'] for p in f['acceptance']]
        outcome=evaluate(selected,pairs,testing)
        greeting=float(np.mean([r['exact'] for r in outcome['outputs'][:6]]))
        results.append({'seed':seed,'selection':selection,'acceptance':outcome,'greeting_accuracy':greeting,'product_targets_pass':greeting>=.8 and outcome['digits']>=.75,'checkpoint':chosen.relative_to(root).as_posix()})
        write_json(run/f'seed-{seed}-result.json',results[-1])
        print(json.dumps({'seed':seed,'final_greeting':greeting,'final_text':outcome['exact'],'final_digits':outcome['digits']}),flush=True)
    # Deployment selection uses development only; no test-based seed selection.
    winner=max(results,key=lambda r:((r['selection']['dev_exact']+r['selection']['dev_digits'])/2,r['selection']['dev_exact']))
    release={'format':'scarx-generation1-release','model_id':'scarx-generation1','checkpoint':winner['checkpoint'],
      'sha256':hashlib.sha256((root/winner['checkpoint']).read_bytes()).hexdigest(),
      'run':run.relative_to(root).as_posix(),'seed':winner['seed'],'hidden':48,'shared_recurrent_core':True,
      'general_chat':False,'general_vision':False,'lifelong_gates_passed':False}
    write_json(root/'artifacts/generation1-release.json',release)
    write_json(run/'result.json',{'seeds':results,'release':release,'resources':monitor.report(),'windows_limit':limit})
    lines=['# Generation 1 joint training','','One48-ALIF core, shared input/recurrent/readout arrays. Two output slices; no model router.','',
      '| Seed | Dev text | Dev digits | Greeting acceptance | All text acceptance | Digits diagnostic |','|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['seed']} | {r['selection']['dev_exact']:.3f} | {r['selection']['dev_digits']:.3f} | {r['greeting_accuracy']:.3f} | {r['acceptance']['exact']:.3f} | {r['acceptance']['digits']:.3f} |" for r in results]
    lines += ['',f"Release seed{winner['seed']} selected by development only.",'','## Release acceptance outputs','']
    lines += [f"- {r['prompt']} → {r['generated']} (target: {r['answer']})" for r in winner['acceptance']['outputs']]
    lines += ['','Product diagnostic, tiny authored families; no representation/retention/general language proof. Previous research PIVOT_NO_SCALE remains.']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('RESULT '+str(run),flush=True)
