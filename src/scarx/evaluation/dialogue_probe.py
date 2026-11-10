"""Read-only prompt/teacher-forcing diagnostics; target-leaky view is explicit."""
import numpy as np
from ..neural.unified import UnifiedCore
from ..neural.language import tokenize
from ..app.workload import plan_workload


def canonical(model,prompt):
    model.core.reset_state();model.last_active[:]=model.ticks
    model.select_capacity(plan_workload(model,prompt)['requested_neurons'])


def prompt_features(payload,prompt):
    model=UnifiedCore.loads(payload);canonical(model,prompt);context=model.context(prompt)
    model.boundary();p=model.word_step(1,context,0)
    return context,model.core.features.copy(),p.copy()


def inspect_example(payload,prompt,target):
    context,first_features,first_probabilities=prompt_features(payload,prompt)
    model=UnifiedCore.loads(payload);canonical(model,prompt);model.boundary()
    tokens=tokenize(target)
    if not tokens or any(t not in model.ids for t in tokens):raise ValueError('Unrepresentable diagnostic target')
    targets=[model.ids[t] for t in tokens]+[0];previous=1;loss=0.;correct=0;teacher_features=[]
    for position,identifier in enumerate(targets):
        p=model.word_step(previous,context,position);loss-=np.log(max(float(p[identifier]),1e-12));correct+=int(p.argmax()==identifier)
        teacher_features.append(model.core.features.copy());previous=identifier
    free=UnifiedCore.loads(payload);canonical(free,prompt);reply=free.generate(prompt,24)
    actual=tokenize(reply['text']);same_prefix=0
    for a,b in zip(actual,tokens):
        if a!=b:break
        same_prefix+=1
    idle=UnifiedCore.loads(payload);canonical(idle,prompt);idle.boundary()
    row={'prompt':prompt,'target':target,'generated':reply['text'],'strict_style_exact':actual==tokens,
        'first_token_correct':int(first_probabilities.argmax())==targets[0],
        'teacher_forced_token_accuracy':correct/len(targets),'teacher_forced_nll':float(loss/len(targets)),
        'prefix_fraction':same_prefix/len(tokens),'stopped':reply['stopped'],'stop_reason':reply['stop_reason'],
        'target_tokens':len(tokens),'generated_tokens':len(actual),'vocabulary_expressible':True,
        'repeated_bigram':len(actual)>=4 and len(set(zip(actual,actual[1:])))<len(actual)-1}
    return row,{'context':context,'prompt_hidden':first_features,
        'gold_teacher_hidden_leaky':np.mean(teacher_features,axis=0),
        'idle':idle.core.features.copy()}


def ridge(train_x,train_y,test_x,test_y,classes=8):
    train_x=np.asarray(train_x,np.float64);test_x=np.asarray(test_x,np.float64)
    mean=train_x.mean(0);scale=train_x.std(0)+1e-4
    a=np.column_stack(((train_x-mean)/scale,np.ones(len(train_x))));b=np.column_stack(((test_x-mean)/scale,np.ones(len(test_x))))
    w=np.linalg.solve(a.T@a+.01*np.eye(a.shape[1]),a.T@np.eye(classes)[train_y])
    predictions=(b@w).argmax(1)
    return {'accuracy':float(np.mean(predictions==test_y)),'predictions':predictions.tolist()}


def local_probe(train_x,train_y,test_x,test_y,seed,classes=8):
    train_x=np.asarray(train_x,np.float32);test_x=np.asarray(test_x,np.float32)
    mean=train_x.mean(0);scale=np.maximum(train_x.std(0,ddof=1),.05)
    a=np.clip((train_x-mean)/scale,-5,5);b=np.clip((test_x-mean)/scale,-5,5)
    w=np.zeros((classes,a.shape[1]),np.float32);bias=np.zeros(classes,np.float32);rng=np.random.default_rng(seed);losses=[]
    for _ in range(20):
        for index in rng.permutation(len(a)):
            logits=w@a[index]+bias;p=np.exp(logits-logits.max());p/=p.sum();target=int(train_y[index])
            losses.append(-float(np.log(max(float(p[target]),1e-12))));error=-p;error[target]+=1
            w+=np.clip(.08*error[:,None]*a[index][None,:],-.025,.025);bias+=np.clip(.08*error,-.025,.025)
            np.clip(w,-3,3,out=w);np.clip(bias,-3,3,out=bias)
    predictions=(b@w.T+bias).argmax(1)
    return {'accuracy':float(np.mean(predictions==test_y)),'predictions':predictions.tolist(),'mean_loss':float(np.mean(losses)),
        'head_updates':20*len(a),'scope':'diagnostic8class output-local head only; never a template answer selector'}
