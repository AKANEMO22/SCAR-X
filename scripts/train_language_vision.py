"""Bounded local training; publishes no backend capability without evidence."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import hashlib
import json
import shutil
import sys
import time
import numpy as np
import psutil
from scarx.neural.language import WordLanguageCore,tokenize
from scarx import Config,SpikingCore
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.evaluation.runs import new_run,write_json

root=Path(__file__).resolve().parents[1]
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
if os.name=='nt': psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
cap=apply_memory_limit()
settings={'hidden':48,'fan_in':8,'plastic_rows':12,'seed':17,'language_max_epochs':80,'vision_epochs':12,'max_seconds':300,'threads':1,
          'scope':'limited dialogue generation and independent digits diagnostic; not general VLM/chat'}
run=new_run(root,'language-vision-pilot-v1',settings,
    data_description='Project-authored Vietnamese dialogue JSON; sklearn bundled UCI handwritten digits. No external model/private conversation/network dataset download.',
    data_files=['data/dialogue/pilot-v1.json'])
artifact=root/'artifacts/language-vision-pilot-v1'/run.name; artifact.mkdir(parents=True)
deadline=time.monotonic()+300


def log(kind,record):
    with (run/'training.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps({'kind':kind,**record},ensure_ascii=False,allow_nan=False)+'\n')
    print(kind+' '+json.dumps(record,ensure_ascii=False,allow_nan=False),flush=True)
    monitor.check()
    if time.monotonic()>deadline: raise TimeoutError('Registered 300-second budget exceeded')


def score_language(model,pairs):
    loss=correct=count=0; rows=[]
    for prompt,answer in pairs:
        l,c,n=model.teacher_score(prompt,answer); loss+=l; correct+=c; count+=n
        generated=model.generate(prompt)
        rows.append({'prompt':prompt,'target':answer,'generated':generated['text'],'stopped':generated['stopped'],
                     'exact':tokenize(generated['text'])==tokenize(answer)})
    return {'loss':loss/count,'token_accuracy':correct/count,'exact':sum(r['exact'] for r in rows)/len(rows),'outputs':rows}


def train_language():
    corpus=json.loads((root/'data/dialogue/pilot-v1.json').read_text(encoding='utf-8'))
    pairs={split:[(prompt,family['answer']) for family in corpus['families'] for prompt in family[split]] for split in ('train','development','test')}
    normalized=[set(p.strip().casefold() for p,_ in pairs[split]) for split in pairs]
    assert not normalized[0]&normalized[1] and not normalized[0]&normalized[2] and not normalized[1]&normalized[2]
    model=WordLanguageCore(WordLanguageCore.vocabulary_from(pairs['train']))
    rng=np.random.default_rng(17); best=(-1.,-float('inf')); selected=None; stale=0
    initial_dev=score_language(model,pairs['development'])
    log('language_initial',{'dev_loss':initial_dev['loss'],'dev_exact':initial_dev['exact'],'vocabulary':len(model.vocabulary)})
    started=time.monotonic()
    for epoch in range(1,81):
        losses=correct=words=0
        for index in rng.permutation(len(pairs['train'])):
            loss,c,n=model.train_pair(*pairs['train'][int(index)]); losses+=loss; correct+=c; words+=n
        development=score_language(model,pairs['development'])
        row={'epoch':epoch,'train_loss':losses/words,'train_token_accuracy':correct/words,'dev_loss':development['loss'],
             'dev_token_accuracy':development['token_accuracy'],'dev_exact':development['exact'],'seconds':time.monotonic()-started}
        log('language_epoch',row)
        candidate=(development['exact'],-development['loss'])
        if candidate>best:
            best=candidate; stale=0; selected=artifact/'language'/f'epoch-{epoch:03d}'; model.save(selected)
        else: stale+=1
        if epoch>=12 and stale>=12: break
    chosen=WordLanguageCore.load(selected); test=score_language(chosen,pairs['test']); dev=score_language(chosen,pairs['development'])
    greeting=next(r for r in test['outputs'] if r['prompt']=='chào bạn')
    qualifies=dev['exact']>=.75 and test['exact']>=.5 and greeting['exact'] and all(r['stopped'] for r in test['outputs'])
    before=chosen.core.fingerprint(); resumed=WordLanguageCore.load(selected)
    parity=all(chosen.generate(p)==resumed.generate(p) for p,_ in pairs['test'])
    assert chosen.core.fingerprint()==before and parity
    result={'train_examples':len(pairs['train']),'development_examples':len(pairs['development']),'test_examples':len(pairs['test']),
            'selected_checkpoint':str(selected.relative_to(root)),'selected_dev':dev,'test':test,'initial_dev':initial_dev,
            'limited_chat_candidate_pass':bool(qualifies),'restart_equal':parity,'seconds':time.monotonic()-started,
            'note':'Autoregressive word output from weights, no target lookup; finite vocabulary and eight authored families. Not normal open-ended chat.'}
    write_json(run/'language-result.json',result)
    log('language_final',{'pass':bool(qualifies),'test_exact':test['exact'],'greeting':greeting,'checkpoint':result['selected_checkpoint']})
    return result


def vision_features(image):
    return np.repeat((image.astype(np.float32)/16-.3)[None,:],4,axis=0)


def score_vision(model,x,y):
    predictions=[]; loss=0.
    for image,label in zip(x,y):
        p=model.forward(vision_features(image)); predictions.append(int(np.argmax(p))); loss-=float(np.log(max(float(p[int(label)]),1e-12)))
    return {'accuracy':float(np.mean(np.asarray(predictions)==y)),'loss':loss/len(y),'predictions':predictions}


def train_vision():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import sklearn
    digits=load_digits(); ids=np.arange(len(digits.target))
    train,hold=train_test_split(ids,test_size=.4,random_state=1701,stratify=digits.target)
    dev,test=train_test_split(hold,test_size=.5,random_state=1703,stratify=digits.target[hold])
    write_json(run/'vision-data.json',{'dataset':'sklearn bundled load_digits / UCI Optical Recognition of Handwritten Digits',
        'sklearn_version':sklearn.__version__,'sha256_pixels':hashlib.sha256(digits.data.tobytes()).hexdigest(),
        'sha256_labels':hashlib.sha256(digits.target.tobytes()).hexdigest(),'train_indices':train.tolist(),'development_indices':dev.tolist(),'test_indices':test.tolist(),
        'source':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html'})
    outcomes=[]
    for mode in ('full','readout'):
        c=Config(inputs=64,hidden=48,outputs=10,fan_in=8,plastic_rows=12,seed=17,leak=.65,threshold=.5,feature_decay=.25,
                 readout_lr=.15,hidden_lr=.0003,anchor_strength=0)
        model=SpikingCore(c); model.w_in-=.15; model.stable_in[:]=model.w_in
        initial=score_vision(model,digits.data[dev],digits.target[dev]); best=(-1.,-float('inf')); selected=None
        rng=np.random.default_rng(1707); start=time.monotonic()
        for epoch in range(1,13):
            total=0.
            for i in rng.permutation(train):
                model.forward(vision_features(digits.data[i]),track=True); total+=model.learn(int(digits.target[i]),mode=mode)
            evaluation=score_vision(model,digits.data[dev],digits.target[dev])
            log('vision_epoch',{'mode':mode,'epoch':epoch,'train_loss':total/len(train),'dev_accuracy':evaluation['accuracy'],'seconds':time.monotonic()-start})
            candidate=(evaluation['accuracy'],-evaluation['loss'])
            if candidate>best:
                best=candidate; selected=artifact/f'vision-{mode}-epoch-{epoch:03d}.npz'; model.save(selected)
        chosen=SpikingCore.load(selected); final=score_vision(chosen,digits.data[test],digits.target[test])
        predictions=np.asarray(final['predictions']); y=digits.target[test]
        parity=np.array_equal(predictions,score_vision(SpikingCore.load(selected),digits.data[test],y)['predictions'])
        outcome={'mode':mode,'initial_dev_accuracy':initial['accuracy'],'best_dev_accuracy':best[0],'test':final,
                 'per_class_accuracy':{str(k):float(np.mean(predictions[y==k]==k)) for k in range(10)},
                 'selected_checkpoint':str(selected.relative_to(root)),'diagnostic_pass':final['accuracy']>=.75,
                 'restart_equal':bool(parity),'seconds':time.monotonic()-start}
        outcomes.append(outcome); log('vision_final',{'mode':mode,'test_accuracy':final['accuracy'],'checkpoint':outcome['selected_checkpoint']})
    return {'outcomes':outcomes,'warning':'Digits classification diagnostic only. Does not establish screenshot/UI understanding, OCR, visual QA or general supports_vision.'}


try:
    with ResourceMonitor(interval=.5) as monitor:
        language=train_language(); vision=train_vision()
        result={'language':language,'vision':vision,'resources':monitor.report(),'windows_limit':cap,'production_replaced':False,
                'vision_and_normal_general_chat_goal_complete':False,'artifact_directory':str(artifact.relative_to(root))}
        write_json(run/'result.json',result)
        lines=['# Language/vision pilot v1','',f"Language exact test: {language['test']['exact']:.3f}; limited candidate pass: {language['limited_chat_candidate_pass']}",'',
               '| Prompt | Neural generation | Expected supervision |','|---|---|---|']
        lines += ['| '+r['prompt'].replace('|','/')+' | '+r['generated'].replace('|','/')+' | '+r['target'].replace('|','/')+' |' for r in language['test']['outputs']]
        lines += ['',vision['warning']]+[f"- Vision {r['mode']}: test accuracy {r['test']['accuracy']:.3f}" for r in vision['outcomes']]
        lines += ['',f"Resources: {result['resources']}",'','Production was not silently replaced. This is not a general chat/vision model.']
        (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write_json(root/'artifacts/language-vision-pilot-summary.json',{'run':run.relative_to(root).as_posix(),'artifact_directory':artifact.relative_to(root).as_posix(),'limited_chat_candidate_pass':language['limited_chat_candidate_pass'],'general_vision_chat':False})
        print('RESULT '+str(run),flush=True)
except BaseException as exc:
    write_json(run/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
    (run/'report.md').write_text(f'# Incomplete language/vision pilot\n\n{type(exc).__name__}: {exc}\n',encoding='utf-8')
    raise
