"""Second bounded pilot, clipping diagnosis registered before training."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import json
import sys
import time
import numpy as np
import psutil
from scarx import Config,SpikingCore
from scarx.neural.language import WordLanguageCore,tokenize
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.evaluation.runs import new_run,write_json

if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
cap=apply_memory_limit(); root=Path(__file__).resolve().parents[1]
run=new_run(root,'language-vision-pilot-v2',{'hidden':48,'seed':17,'readout_lr':.015,'weight_clip':20.,'max_seconds':360,'conditions':['full','readout']},
 data_description='Self-authored Vietnamese dialogue; fresh v2test paraphrases; bundled UCI digits using v1split (reused diagnostic test). No pretrained model/private data.',
 data_files=['data/dialogue/pilot-v1.json','data/dialogue/pilot-v2-test.json'])
artifact=root/'artifacts/language-vision-pilot-v2'/run.name; artifact.mkdir(parents=True)
deadline=time.monotonic()+360


def log(kind,record):
    with (run/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'kind':kind,**record},ensure_ascii=False)+'\n')
    print(kind+' '+json.dumps(record,ensure_ascii=False),flush=True)
    monitor.check()
    if time.monotonic()>deadline:raise TimeoutError('360-second registered budget')


def language_score(model,pairs):
    loss=correct=count=0; outputs=[]
    for prompt,answer in pairs:
        l,c,n=model.teacher_score(prompt,answer); loss+=l; correct+=c; count+=n
        output=model.generate(prompt)
        outputs.append({'prompt':prompt,'target':answer,'generated':output['text'],'stopped':output['stopped'],'exact':tokenize(output['text'])==tokenize(answer)})
    return {'loss':loss/count,'token_accuracy':correct/count,'exact':sum(r['exact'] for r in outputs)/len(outputs),'outputs':outputs}


def language():
    corpus=json.loads((root/'data/dialogue/pilot-v1.json').read_text(encoding='utf-8'))
    tests=json.loads((root/'data/dialogue/pilot-v2-test.json').read_text(encoding='utf-8'))
    train=[(p,f['answer']) for f in corpus['families'] for p in f['train']]
    dev=[(p,f['answer']) for f in corpus['families'] for p in f['development']]
    test=[(p,f['answer']) for f in corpus['families'] for p in tests[f['intent']]]
    assert not {p for p,_ in train+dev}&{p for p,_ in test}
    outcomes=[]
    for mode in ('full','readout'):
        model=WordLanguageCore(WordLanguageCore.vocabulary_from(train),config_overrides={'readout_lr':.015,'weight_clip':20.})
        rng=np.random.default_rng(17); best=(-1.,-float('inf')); selected=None; stale=0
        for epoch in range(1,201):
            total=correct=count=0
            for i in rng.permutation(len(train)):
                loss,c,n=model.train_pair(*train[int(i)],mode=mode); total+=loss; correct+=c; count+=n
            score=language_score(model,dev)
            candidate=(score['exact'],-score['loss'])
            if candidate>best:
                best=candidate; stale=0; selected=artifact/f'language-{mode}'/f'epoch-{epoch:03d}'; model.save(selected)
            else:stale+=1
            if epoch==1 or epoch%5==0:log('language_epoch',{'mode':mode,'epoch':epoch,'train_loss':total/count,'dev_exact':score['exact'],'dev_loss':score['loss']})
            if epoch>=30 and stale>=30:break
        outcomes.append({'mode':mode,'development_exact':best[0],'development_loss':-best[1],'checkpoint':str(selected.relative_to(root))})
    chosen=max(outcomes,key=lambda r:(r['development_exact'],-r['development_loss']))
    write_json(run/'language-selection.json',{'chosen':chosen,'development_conditions':outcomes,'test_opened':False})
    model=WordLanguageCore.load(root/chosen['checkpoint']); final=language_score(model,test)
    greeting=model.generate('chào bạn'); restart=WordLanguageCore.load(root/chosen['checkpoint']).generate('chào bạn')==greeting
    qualifies=chosen['development_exact']>=.75 and final['exact']>=.5 and tokenize(greeting['text'])==tokenize(corpus['families'][0]['answer']) and all(r['stopped'] for r in final['outputs'])
    result={'development_conditions':outcomes,'selected':chosen,'test':final,'requested_greeting':greeting,'restart_equal':restart,'limited_chat_candidate_pass':bool(qualifies),
      'scope':'44-word vocabulary/8authored conversational families; autoregressive ALIF outputs, no answerbank; not general chat.'}
    write_json(run/'language-result.json',result)
    log('language_final',{'selected':chosen['mode'],'dev_exact':chosen['development_exact'],'test_exact':final['exact'],'greeting':greeting['text'],'limited_candidate_pass':bool(qualifies)})
    return result


def image_sequence(image):return np.repeat((image.astype(np.float32)/16-.3)[None,:],4,axis=0)


def vision_score(model,x,y):
    loss=0.; predictions=[]
    for image,label in zip(x,y):
        p=model.forward(image_sequence(image)); loss-=float(np.log(max(float(p[int(label)]),1e-12)));predictions.append(int(np.argmax(p)))
    return {'accuracy':float(np.mean(np.asarray(predictions)==y)),'loss':loss/len(y),'predictions':predictions}


def vision():
    from sklearn.datasets import load_digits
    digits=load_digits()
    old=json.loads((root/'runs/20261108T190007Z-language-vision-pilot-v1-c11f25/vision-data.json').read_text(encoding='utf-8'))
    train=np.asarray(old['train_indices']); dev=np.asarray(old['development_indices']); test=np.asarray(old['test_indices'])
    write_json(run/'vision-data.json',old)
    outcomes=[]
    for mode in ('full','readout'):
        model=SpikingCore(Config(inputs=64,hidden=48,outputs=10,fan_in=8,plastic_rows=12,seed=17,leak=.65,threshold=.5,feature_decay=.25,
                   readout_lr=.015,hidden_lr=.0003,anchor_strength=0,weight_clip=20.))
        model.w_in-=.15;model.stable_in[:]=model.w_in
        rng=np.random.default_rng(1707);best=(-1.,-float('inf'));selected=None
        for epoch in range(1,25):
            total=0.
            for i in rng.permutation(train):
                model.forward(image_sequence(digits.data[i]),track=True);total+=model.learn(int(digits.target[i]),mode=mode)
            score=vision_score(model,digits.data[dev],digits.target[dev]);candidate=(score['accuracy'],-score['loss'])
            if candidate>best:
                best=candidate;selected=artifact/f'vision-{mode}-epoch-{epoch:03d}.npz';model.save(selected)
            if epoch==1 or epoch%4==0:log('vision_epoch',{'mode':mode,'epoch':epoch,'train_loss':total/len(train),'dev_accuracy':score['accuracy']})
        outcomes.append({'mode':mode,'dev_accuracy':best[0],'dev_loss':-best[1],'checkpoint':str(selected.relative_to(root))})
    chosen=max(outcomes,key=lambda r:(r['dev_accuracy'],-r['dev_loss']))
    write_json(run/'vision-selection.json',{'chosen':chosen,'development_conditions':outcomes,'test_opened':False})
    model=SpikingCore.load(root/chosen['checkpoint']);final=vision_score(model,digits.data[test],digits.target[test])
    result={'development_conditions':outcomes,'selected':chosen,'test':final,'diagnostic_pass':final['accuracy']>=.75,
            'scope':'Reused digits diagnostic test, never trained on test. Not screenshot understanding/OCR/visual QA.'}
    write_json(run/'vision-result.json',result);log('vision_final',{'selected':chosen['mode'],'test_accuracy':final['accuracy'],'pass':result['diagnostic_pass']})
    return result


try:
    with ResourceMonitor(interval=.5) as monitor:
        text=language(); images=vision()
        result={'language':text,'vision':images,'resources':monitor.report(),'windows_limit':cap,'production_replaced':False,'general_vision_chat_achieved':False}
        write_json(run/'result.json',result)
        lines=['# Language/vision pilot v2','',f"Language selected={text['selected']['mode']}; devexact={text['selected']['development_exact']:.3f}; testexact={text['test']['exact']:.3f}; limitedcandidate={text['limited_chat_candidate_pass']}",'',
          '| Prompt | Generated by neural weights | Target |','|---|---|---|']
        lines+=['| '+r['prompt']+' | '+r['generated']+' | '+r['target']+' |' for r in text['test']['outputs']]
        lines+=['',f"Requested greeting: {text['requested_greeting']['text']}",'',f"Digits diagnostic test accuracy: {images['test']['accuracy']:.3f}",images['scope'],'',text['scope'],'',f"Resource report: {result['resources']}",'','Production not silently replaced. Native screenshot understanding and general chat remain unachieved.']
        (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write_json(root/'artifacts/language-vision-pilot-summary.json',{'run':run.relative_to(root).as_posix(),'artifact_directory':artifact.relative_to(root).as_posix(),'limited_chat_candidate_pass':text['limited_chat_candidate_pass'],'general_vision_chat':False})
        print('RESULT '+str(run),flush=True)
except BaseException as exc:
    write_json(run/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
    (run/'report.md').write_text(f'# Incomplete pilot v2\n\n{type(exc).__name__}: {exc}\n',encoding='utf-8')
    raise
