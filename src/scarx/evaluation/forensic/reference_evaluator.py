"""Independent relational-task scorer. No agent evaluator/summary imports.

The oracle follows the public 8-channel event specification. Pair scores are
computed from labels/facts/query and group identity, never supplied flip masks
alone. Numerical tolerances concern float32 pulse subtraction, not accuracy.
"""
import ast
import hashlib
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np


class AuditViolation(ValueError):
    pass


def tensor_digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()


def binary(a,name):
    a=np.asarray(a)
    if a.ndim!=1 or not np.isin(a,[0,1]).all():raise AuditViolation(name+': binary vector required')
    return a.astype(np.int64)


def oracle_events(x):
    x=np.asarray(x)
    if x.ndim!=3 or x.shape[2]!=8 or not np.isfinite(x).all():raise AuditViolation('finite event tensor [N,T,8] required')
    values=[];queries=[];ticks=[]
    for row in x:
        key_marks=np.argwhere(row[:,:2]>.5)
        query_marks=np.argwhere(row[:,4:6]>.5)
        if len(key_marks)!=2 or len(query_marks)!=1:raise AuditViolation('exactly two writes and one query required')
        vals={}
        for tick,key in key_marks:
            marks=np.flatnonzero(row[tick,2:4]>.5)
            if len(marks)!=1 or int(key) in vals:raise AuditViolation('unambiguous independent key/value writes required')
            vals[int(key)]=int(marks[0])
        qt,q=query_marks[0]
        if set(vals)!={0,1} or qt<=int(key_marks[:,0].max()):raise AuditViolation('both keys must precede query')
        values.append([vals[0],vals[1]]);queries.append(int(q));ticks.append(int(qt))
    v=np.array(values,dtype=np.int32).reshape(-1,2);q=np.array(queries,np.int32)
    return {'vA':v[:,0],'vB':v[:,1],'q':q,'query_ticks':np.array(ticks,np.int32),
            'labels':np.where(q==0,v[:,0],v[:,1]).astype(np.int32)}


def pair_metrics(predictions,labels,vA,vB,q,pair_id,flips=None):
    arrays={k:binary(a,k) for k,a in [('prediction',predictions),('label',labels),('vA',vA),('vB',vB),('q',q)]}
    p,y,a,b,query=[arrays[k] for k in ('prediction','label','vA','vB','q')]
    ids=np.asarray(pair_id)
    n=len(p)
    if n==0 or n%2 or any(len(v)!=n for v in arrays.values()) or ids.shape!=(n,):raise AuditViolation('nonempty complete sibling pairs required')
    if not np.array_equal(ids[::2],ids[1::2]) or len(np.unique(ids))!=n//2:raise AuditViolation('pair IDs do not match adjacent sibling groups')
    if not np.all(query[::2]==0) or not np.all(query[1::2]==1):raise AuditViolation('ordered query-A/query-B siblings required')
    if not np.array_equal(a[::2],a[1::2]) or not np.array_equal(b[::2],b[1::2]):raise AuditViolation('siblings have different stored facts')
    if not np.array_equal(y,np.where(query==0,a,b)):raise AuditViolation('labels disagree with independent stored facts and query')
    category=a[::2]!=b[::2]
    if flips is not None and not np.array_equal(np.asarray(flips),category):raise AuditViolation('supplied flips disagree with stored facts')
    correct=p==y;both=correct[::2]&correct[1::2]
    rate=lambda mask:float(both[mask].mean()) if mask.any() else None
    need=(query==1)&(a!=b)
    return {'direct':float(correct.mean()),'both_A':rate(category),'both_B':rate(~category),
            'both_all':float(both.mean()),'needB':float(correct[need].mean()) if need.any() else None,
            'case_A_pairs':int(category.sum()),'case_B_pairs':int((~category).sum()),'pair_count':n//2}


def validate_pair_inputs(x,labels,vA,vB,q,pair_id,flips=None,expected_tick=None,common_noise=True):
    observed=oracle_events(x)
    for name,value in [('labels',labels),('vA',vA),('vB',vB),('q',q)]:
        if not np.array_equal(observed[name],value):raise AuditViolation('raw markers disagree with '+name)
    if expected_tick is not None and not np.all(observed['query_ticks']==expected_tick):raise AuditViolation('query tick contract violated')
    pair_metrics(labels,labels,vA,vB,q,pair_id,flips)
    base=np.asarray(x,np.float64).copy()
    base[np.arange(len(base)),observed['query_ticks'],4+observed['q']]-=1.5
    noise_difference=np.max(np.abs(base[::2]-base[1::2]),axis=(1,2))
    if common_noise and np.any(noise_difference>2e-7):raise AuditViolation('counterfactual siblings do not share non-query input/noise')
    return {'samples':len(base),'query_ticks':np.unique(observed['query_ticks']).tolist(),
            'non_query_mismatch_pairs':int(np.sum(noise_difference>2e-7)),
            'max_non_query_difference':float(noise_difference.max(initial=0)),'oracle_all_rows':True}


def seed_bootstrap(by_seed,resamples=10000,random_seed=2101):
    if not isinstance(by_seed,dict) or not by_seed:raise AuditViolation('distinct seed-keyed effects required')
    keys=sorted(by_seed);values=np.asarray([by_seed[k] for k in keys],np.float64)
    if not np.isfinite(values).all():raise AuditViolation('finite seed-level effects required')
    rng=np.random.default_rng(random_seed)
    draws=values[rng.integers(0,len(values),(resamples,len(values)))].mean(1)
    lo,hi=np.quantile(draws,[.025,.975])
    return {'mean':float(values.mean()),'lower95':float(lo),'upper95':float(hi),'worst':float(values.min()),
            'independent_seed_count':len(keys),'seed_ids':keys,'per_seed':values.tolist(),
            'resamples':resamples,'bootstrap_seed':random_seed}


def integrate_gates(required,gates):
    if len(required)!=len(set(required)):raise AuditViolation('duplicate required gate')
    unknown=[];failed=[]
    for name in required:
        value=gates.get(name)
        if type(value) is not bool:unknown.append(name)
        elif not value:failed.append(name)
    return {'all_gates_complete':not unknown,'missing_or_untyped':unknown,'failed':failed,
            'unexpected_keys':sorted(set(gates)-set(required)),
            'verdict':'FAIL' if failed else 'INCONCLUSIVE' if unknown or set(gates)!=set(required) else 'PASS'}


def checked_threshold(value,minimum,declared=None):
    if type(value) not in (int,float) or not math.isfinite(value):raise AuditViolation('gate measurement is absent or nonfinite')
    actual=value>=minimum
    if declared is not None and (type(declared) is not bool or declared!=actual):raise AuditViolation('declared gate disagrees with measured value')
    return actual


def ensure_joinable(left,right,keys):
    missing=[k for k in keys if k not in left or k not in right]
    changed=[k for k in keys if k in left and k in right and left[k]!=right[k]]
    if missing or changed:raise AuditViolation('NON_EQUIVALENT_CONTROL: missing='+str(missing)+' changed='+str(changed))


@dataclass(frozen=True)
class Features:
    values: np.ndarray
    scaler_digest: str
    raw_digest: str


def scaled_features(raw,mean,sd,scaler_digest,clip=None):
    raw=np.asarray(raw);mean=np.asarray(mean);sd=np.asarray(sd)
    if not scaler_digest or not np.isfinite(raw).all() or not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd<=0):raise AuditViolation('valid source scaler required')
    z=(raw-mean)/sd
    if clip is not None:z=np.clip(z,-clip,clip)
    return Features(z.astype(np.float32),scaler_digest,tensor_digest(raw))


def require_scaled(value,digest):
    if not isinstance(value,Features) or value.scaler_digest!=digest:raise AuditViolation('raw or differently scaled features rejected')
    return value.values


def require_active_shift(clean,shifted):
    if np.array_equal(clean,shifted):raise AuditViolation('NO_OP_SHIFT: hidden feature tensor did not change')


def unique_top_level_definitions(source):
    tree=ast.parse(source);names=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
    duplicates={k:v for k,v in Counter(names).items() if v>1}
    if duplicates:raise AuditViolation('shadowed top-level definitions: '+str(duplicates))
    return names
