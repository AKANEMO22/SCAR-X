"""Dataset/oracle/probe/statistics for the registered matched R8 DEV study."""
import itertools,math
import numpy as np
from .r8_matched_credit import peer_module

TASKS=('strict_binding','xor','distractor_binding','order')
SEEDS=(140011,140017,140023)


def dataset(task,seed):
    generators=peer_module('r8_benchmark_integrity')
    names={'strict_binding':'generate_strict_binding','xor':'generate_xor_representation',
        'distractor_binding':'generate_distractor_binding','order':'generate_order_representation'}
    generator=getattr(generators,names[task]);out={}
    for split,offset,count,ood in [('train',200,1024,False),('refit',300,256,False),('id',400,512,False),('ood',500,512,True)]:
        x,y=generator(seed+offset,count,ood=ood)
        labels,summaries=event_oracle(task,x)
        if not np.array_equal(labels,y):raise ValueError('Independent event-label oracle mismatch: '+task+'/'+split)
        out[split]=(x,y,summaries)
    return out


def event_oracle(task,sequences):
    labels=[];summaries=[]
    for x in np.asarray(sequences):
        if task in ('strict_binding','distractor_binding'):
            scores=x[:,:2].max(1)+x[:,2:4].max(1);ticks=sorted(np.argsort(scores)[-2:].tolist())
            values={};writes=[]
            for t in ticks:
                key=int(np.argmax(x[t,:2]));value=int(np.argmax(x[t,2:4]));values[key]=value
                writes.append({'tick':t,'key':key,'value':value})
            if set(values)!={0,1} or ticks[-1]>=10:raise ValueError('Ambiguous binding event fixture')
            query=int(np.argmax(x[10,4:6]));label=values[query]
            summary={'writes':writes,'query':query,'query_tick':10}
        elif task=='xor':
            ta=int(np.argmax(x[:,:2].max(1)));tb=int(np.argmax(x[:,2:4].max(1)))
            a=int(np.argmax(x[ta,:2]));b=int(np.argmax(x[tb,2:4]));label=a^b
            summary={'first_bit':a,'first_tick':ta,'second_bit':b,'second_tick':tb,'rule':'xor'}
        else:
            ticks=sorted(np.argsort(x[:,:2].max(1))[-2:].tolist());first=int(np.argmax(x[ticks[0],:2]));second=int(np.argmax(x[ticks[1],:2]))
            if first==second:raise ValueError('Ambiguous order fixture')
            label=first;summary={'first_symbol':first,'first_tick':ticks[0],'second_symbol':second,'second_tick':ticks[1],'rule':'order'}
        labels.append(label);summaries.append(summary)
    return np.asarray(labels,np.int32),summaries


def refit_predict(train_f,train_y,test_f):
    mean=train_f.mean(0);scale=train_f.std(0)+1e-4
    a=np.column_stack(((train_f-mean)/scale,np.ones(len(train_f))))
    b=np.column_stack(((test_f-mean)/scale,np.ones(len(test_f))))
    weights=np.linalg.solve(a.T@a+.01*np.eye(a.shape[1]),a.T@np.eye(2)[train_y])
    return np.argmax(b@weights,axis=1).astype(np.int32)


def direct_predict(model,features):
    scale=np.maximum(np.sqrt(model.f_var),.05).astype(np.float32)
    f_hat=np.clip((features-model.f_mean)/scale,-5,5).astype(np.float32)
    return np.argmax(f_hat@model.core.w_out.T+model.core.bias,axis=1).astype(np.int32)


def paired_statistics(values):
    a=np.asarray(values,np.float64)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Exactly3finite seed effects required')
    means=np.asarray([np.mean(v) for v in itertools.product(a,repeat=3)])
    observed=abs(float(a.mean()))
    sign_means=[abs(float((a*np.array(s)).mean())) for s in itertools.product((-1,1),repeat=3)]
    return {'mean':float(a.mean()),'lower95':float(np.quantile(means,.025,method='inverted_cdf')),
        'upper95':float(np.quantile(means,.975,method='inverted_cdf')),'exact_two_sided_p':sum(v>=observed-1e-12 for v in sign_means)/8,
        'seed_clusters':3,'effects':a.tolist(),'scope':'Reused exploratory DEV seeds, not confirmation'}


def holm(values):
    ordered=sorted(values,key=values.get);previous=0.;out={}
    for i,key in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-i)*values[key]));out[key]=previous
    return out
