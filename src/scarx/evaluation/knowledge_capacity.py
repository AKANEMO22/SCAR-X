"""Closed-world load/retention diagnostics; no training policy changes."""
import numpy as np
from ..learning.verifiers import integer_answer
from ..neural.language import tokenize
from ..neural.unified import UnifiedCore


def canonical(a,op,b):
    return (min(a,b),op,max(a,b)) if op=='+' else (a,op,b)


def dataset(seed=8107,nonnegative_only=False):
    rng=np.random.default_rng(seed);seen=set();rows=[]
    while len(rows)<168:
        op='+' if len(rows)%2==0 else '-'
        a,b=map(int,rng.integers(0,20,2));key=canonical(a,op,b)
        if nonnegative_only and op=='-' and a<b:continue
        if key in seen:continue
        seen.add(key);index=len(rows)
        rows.append({'id':f'fact-{index:03d}','key':list(key),'a':a,'op':op,'b':b,
          'prompt':f'Tính {a} {op} {b}, chỉ ghi số.',
          'paraphrase':f'Kết quả của phép tính {a} {"cộng" if op=="+" else "trừ"} {b} là bao nhiêu?',
          'target':str(integer_answer((a,op,b))),
          'split':'curriculum' if index<128 else 'unseen' if index<160 else 'distractor',
          'source':'project-authored deterministic arithmetic; independently recomputed',
          'verified_by':'integer-arithmetic-v1'})
    return rows


def probe(model,rows,field='paraphrase',independent=True):
    m=UnifiedCore.loads(model.dumps());out=[]
    for row in rows:
        if independent:
            m.core.reset_state();m.last_active[:]=m.ticks
        text=m.generate(row[field])['text']
        out.append({'id':row['id'],'target':row['target'],'generated':text,
          'correct':tokenize(text)==tokenize(row['target'])})
    return out


def fraction(outputs):
    return sum(x['correct'] for x in outputs)/len(outputs) if outputs else None


def forgetting(previous_best,current):
    """Per-skill losses cannot be cancelled by improvement on other skills."""
    best=np.asarray(previous_best,dtype=float);now=np.asarray(current,dtype=float)
    if best.shape!=now.shape or best.size==0:raise ValueError('Matching nonempty skill vectors required')
    losses=np.maximum(0,best-now)
    return {'mean':float(losses.mean()),'worst':float(losses.max()),'losses':losses.tolist(),'next_best':np.maximum(best,now).tolist()}


def frozen_slot(model,prompt,target):
    """Observe teacher forcing without weight updates, matching encoded slots."""
    context=model.context(prompt);model.boundary()
    model.core.elig_in.fill(0);model.core.elig_rec.fill(0)
    model.core.pre.fill(0);model.core.post.fill(0)
    previous=1
    for position,word in enumerate(tokenize(target)+['<eos>']):
        token=model.ids[word];model.word_step(previous,context,position,track=True);previous=token


def retention(immediate,baseline,current):
    before={x['id']:x['correct'] for x in baseline}
    now={x['id']:x['correct'] for x in current}
    acquired=[x['id'] for x in immediate if x['correct'] and not before.get(x['id'],False)]
    count=sum(now.get(key,False) for key in acquired);n=len(acquired)
    if not n:return {'acquired':0,'retained':0,'rate':None,'wilson95':None,'status':'undefined: no acquired examples'}
    p=count/n;z=1.959963984540054;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return {'acquired':n,'retained':count,'rate':p,'wilson95':[max(0.,center-half),min(1.,center+half)],
      'status':'descriptive sample interval; examples share arithmetic structure'}
