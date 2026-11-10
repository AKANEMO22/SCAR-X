"""P2 diagnostic generators/probes/interventions. No learner sees task metadata."""
import numpy as np
from .tasks21 import context_task


def mixed_stream(seed,per_window=192,contexts=20):
    pools=[context_task(seed+7300000+131*j,per_window*2,j,contexts) for j in range(contexts)]
    cursor=np.zeros(contexts,np.int32);rng=np.random.default_rng(seed+7400000);xs=[];ys=[];groups=[]
    for window in range(contexts):
        for offset in range(per_window):
            group=window+int(window<contexts-1 and rng.random()<(offset+.5)/per_window)
            x,y=pools[group];i=cursor[group];cursor[group]+=1
            xs.append(x[i]);ys.append(y[i]);groups.append(group)
    return np.stack(xs),np.array(ys,np.int32),np.array(groups,np.int32)


def ridge_multiclass(train_features,train_labels,test_features,test_labels,outputs=4):
    f=np.asarray(train_features,np.float64);x=np.asarray(test_features,np.float64)
    if f.ndim!=2 or x.ndim!=2 or f.shape[1]!=x.shape[1] or len(f)!=len(train_labels) or len(x)!=len(test_labels):raise ValueError('Invalid ridge diagnostic')
    mean=f.mean(0);scale=f.std(0)+1e-4
    a=np.column_stack(((f-mean)/scale,np.ones(len(f))));b=np.column_stack(((x-mean)/scale,np.ones(len(x))))
    w=np.linalg.solve(a.T@a+.01*np.eye(a.shape[1]),a.T@np.eye(outputs)[train_labels])
    predictions=(b@w).argmax(1)
    return float(np.mean(predictions==test_labels)),predictions


def head_accuracy(features,labels,weights,bias,mean,scale):
    predictions=(np.clip((features-mean)/scale,-5,5)@weights.T+bias).argmax(1)
    return float(np.mean(predictions==labels))


def component_intervention(old_features,new_features,labels,old_head,new_head,mean,scale):
    scores={}
    for hidden,f in (('old',old_features),('new',new_features)):
        for head,(w,b) in (('old',old_head),('new',new_head)):
            scores[hidden+'_hidden_'+head+'_head']=head_accuracy(f,labels,w,b,mean,scale)
    base=scores['old_hidden_old_head'];new=scores['new_hidden_new_head']
    head_drop=base-scores['old_hidden_new_head'];hidden_drop=base-scores['new_hidden_old_head']
    return {**scores,'head_only_drop':head_drop,'hidden_only_drop':hidden_drop,
        'interaction_drop':(base-new)-head_drop-hidden_drop}


def retention_metrics(matrix):
    a=np.asarray(matrix,float)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite(a).all():raise ValueError('Expected complete square accuracy matrix')
    diagonal=np.diag(a);final=a[-1];forgetting=np.array([a[j:,j].max()-final[j] for j in range(len(final))])
    acquisition=float(diagonal.mean());last=float(final.mean());mean=float(forgetting.mean());worst=float(forgetting.max())
    return {'acquisition':acquisition,'final':last,'mean_forgetting':mean,'worst_forgetting':worst,
        'per_context_forgetting':forgetting.tolist(),'diagonal':diagonal.tolist(),'signed_bwt':float(np.mean(final-diagonal)),
        'joint_gate':bool(acquisition>=.65 and last>=.65 and mean<=.03 and worst<=.05)}
