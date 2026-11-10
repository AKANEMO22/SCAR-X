"""Registered bounded comparison; keep deployed model unchanged until audited."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from sklearn.datasets import load_digits
from scarx.neural.unified import UnifiedCore
from scarx.neural.language import tokenize
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.evaluation.runs import new_run,write_json,cluster_interval
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
cap=apply_memory_limit()
corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
train=[(p,f['answer']) for f in corpus['families'] for p in f['train']]
dev=[(p,f['answer']) for f in corpus['families'] for p in f['development']]
vocabulary=['<eos>','<bos>']+sorted({t for _,a in train for t in tokenize(a)})
split=json.loads((root/'runs/20261109T000625Z-generation1-joint-bd6c92/vision-split.json').read_text(encoding='utf-8'))
digits=load_digits();conditions=('original_full','lexical_full','lexical_readout')
run=new_run(root,'generation1-improvement',{'protocol':'experiments/generation1-adaptive-improvement.md','seeds':[17,29,43],'epochs':60,'conditions':conditions},data_description='Known8dialoguefamilies; train-only lexical IDF; same UCI diagnostic split. No private data/pretrained model.',data_files=['data/dialogue/generation1.json','data/dialogue/generation1-confirmation-next.json'])
artifact=root/'artifacts/generation1-improvement'/run.name;artifact.mkdir(parents=True)
def score(model,pairs,indices):
    m=UnifiedCore.loads(model.dumps());outputs=[];loss=0.;count=0
    for prompt,answer in pairs:
        response=m.generate(prompt)
        outputs.append({'prompt':prompt,'answer':answer,'generated':response['text'],'exact':tokenize(response['text'])==tokenize(answer)})
        context=m.context(prompt);m.boundary();previous=1
        for position,token in enumerate([m.ids[t] for t in tokenize(answer)]+[0]):
            p=m.word_step(previous,context,position);loss-=np.log(max(float(p[token]),1e-12));count+=1;previous=token
    correct=sum(m.image(digits.data[i])['digit']==int(digits.target[i]) for i in indices)
    return {'exact':sum(r['exact'] for r in outputs)/len(outputs),'loss':float(loss/count),'digits':correct/len(indices),'outputs':outputs}
outcomes=[]
with ResourceMonitor(interval=.5) as monitor:
 for seed in (17,29,43):
  for condition in conditions:
    encoding=None if condition=='original_full' else UnifiedCore.lexical_encoding(p for p,_ in train)
    model=UnifiedCore(vocabulary,seed,encoding);rng=np.random.default_rng(seed);best=(-1.,-1.);selected=None;start=time.monotonic()
    mode='readout' if condition.endswith('readout') else 'full'
    for epoch in range(1,61):
        for index,i in enumerate(rng.permutation(split['training'])):
            model.image(digits.data[i],int(digits.target[i]),mode)
            if index%6==0:model.train_pair(*train[int(rng.integers(len(train)))],mode)
            if index%256==0:monitor.check()
        record=score(model,dev,split['development']);rank=((record['exact']+record['digits'])/2,-record['loss'])
        row={'seed':seed,'condition':condition,'epoch':epoch,'dev_exact':record['exact'],'dev_digits':record['digits'],'dev_loss':record['loss'],'seconds':time.monotonic()-start}
        with (run/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if rank>best:
            best=rank;selected={**row,'checkpoint':(artifact/f'{condition}-{seed}.zip').relative_to(root).as_posix()};(root/selected['checkpoint']).write_bytes(model.dumps())
        if epoch==1 or epoch%15==0:print(json.dumps(row),flush=True)
        if time.monotonic()-start>600:raise TimeoutError('Registered600s condition budget')
    outcomes.append(selected);write_json(run/'selection-progress.json',outcomes)
 # Freeze all selection decisions before opening confirmation.
 means={c:float(np.mean([(r['dev_exact']+r['dev_digits'])/2 for r in outcomes if r['condition']==c])) for c in conditions}
 winner=max(conditions,key=lambda c:means[c])
 release=max((r for r in outcomes if r['condition']==winner),key=lambda r:((r['dev_exact']+r['dev_digits'])/2,-r['dev_loss']))
 write_json(run/'selection-frozen.json',{'outcomes':outcomes,'condition_scores':means,'selected':release,'confirmation_opened':False})
 confirmation=json.loads((root/'data/dialogue/generation1-confirmation-next.json').read_text(encoding='utf-8'))['prompts']
 for row in outcomes:
    row['confirmation']=score(UnifiedCore.loads((root/row['checkpoint']).read_bytes()),confirmation,split['diagnostic_test'])
 gains=[next(r['confirmation']['exact'] for r in outcomes if r['seed']==s and r['condition']=='lexical_full')-next(r['confirmation']['exact'] for r in outcomes if r['seed']==s and r['condition']=='original_full') for s in (17,29,43)]
 hidden=[next(r['confirmation']['exact'] for r in outcomes if r['seed']==s and r['condition']=='lexical_full')-next(r['confirmation']['exact'] for r in outcomes if r['seed']==s and r['condition']=='lexical_readout') for s in (17,29,43)]
 result={'outcomes':outcomes,'selection':release,'lexical_vs_original':cluster_interval(gains),'full_vs_readout':cluster_interval(hidden),'resources':monitor.report(),'memory_cap':cap,'deployed':False,'scope':'knownfamilyparaphrases, not general chat/retention'}
 write_json(run/'result.json',result)
 lines=['# G1 local capability comparison','','| Seed | Condition | Dev text | Confirm text | Digits |','|---|---|---:|---:|---:|']
 lines += [f"| {r['seed']} | {r['condition']} | {r['dev_exact']:.3f} | {r['confirmation']['exact']:.3f} | {r['confirmation']['digits']:.3f} |" for r in outcomes]
 lines+=['',json.dumps({'lexical_vs_original':result['lexical_vs_original'],'full_vs_readout':result['full_vs_readout']},indent=2),'','Three independent seeds, exploratory uncertainty; familiar8families. No representation/retention/scaling success claim.']
 (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 write_json(root/'artifacts/generation1-improvement-summary.json',{'run':run.relative_to(root).as_posix(),'selected':release,'deployed':False})
 print('RESULT '+str(run),flush=True)
