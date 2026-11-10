import numpy as np


def representation(task,seed,count,ood=False):
    """Twelve-step tasks; no label channel. Query/control tokens explicit."""
    rng=np.random.default_rng(seed)
    x=rng.normal(0,.08,(count,12,8)).astype(np.float32)
    y=np.arange(count)%2
    for i in range(count):
        p1=0 if ood else int(rng.integers(1,3))
        p2=8 if ood else int(rng.integers(4,7))
        if task=="xor":
            a=int(rng.integers(2)); b=a^int(y[i])
            x[i,p1,a]+=1.5; x[i,p2,2+b]+=1.5
        elif task=="order":
            a=int(y[i]); b=1-a
            x[i,p1,a]+=1.5; x[i,p2,b]+=1.5
            # Same counts, same final marker. Only their ordering distinguishes label.
        elif task=="binding":
            query=int(rng.integers(2)); values=rng.integers(2,size=2); values[query]=y[i]
            order=rng.permutation(2)
            for pos,key in zip((p1,p2),order):
                x[i,pos,key]+=1.5; x[i,pos,2+int(values[key])]+=1.5
            x[i,10,4+query]+=1.5
        else: raise ValueError(task)
        x[i,11,7]+=1
    order=rng.permutation(count)
    return x[order],y[order].astype(np.int32)


def context_task(seed,count,context,contexts=20):
    rng=np.random.default_rng(seed); x=rng.normal(0,.025,(count,12,4+contexts)).astype(np.float32)
    labels=np.random.default_rng(700+context).permutation(4)
    cue=np.arange(count)%4
    for i in range(count): x[i,:2,cue[i]]+=1
    x[:,:,4+context]+=.65
    order=rng.permutation(count)
    return x[order],labels[cue[order]].astype(np.int32)


def ridge_probe(train_f,train_y,test_f,test_y):
    mean=train_f.mean(0); scale=train_f.std(0)+1e-4
    a=np.column_stack(((train_f-mean)/scale,np.ones(len(train_f))))
    b=np.column_stack(((test_f-mean)/scale,np.ones(len(test_f))))
    target=np.eye(2)[train_y]
    w=np.linalg.solve(a.T@a+.01*np.eye(a.shape[1]),a.T@target)
    return float(np.mean(np.argmax(b@w,axis=1)==test_y))
