"""Conservative audit gates, independent from peer verdicts. No model learning."""
from collections import defaultdict
import itertools,math
import numpy as np
from .reference_evaluator import AuditViolation

TASKS=('strict_binding','distractor_binding','xor','order')


def finite_fraction(value,name):
    if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=1:
        raise AuditViolation(name+': finite fraction in[0,1] required')
    return float(value)


def retention_metrics(matrix):
    a=np.asarray(matrix,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not 2<=len(a)<=100:
        raise AuditViolation('Square R-matrix for2..100stages required')
    observed=a[np.tril_indices(len(a))]
    if not np.isfinite(observed).all() or np.any((observed<0)|(observed>1)):
        raise AuditViolation('Observed R-matrix cells must be finite fractions')
    final=a[-1];forget=np.array([a[j:,j].max()-final[j] for j in range(len(a))])
    return {'acquisition':float(np.diag(a).mean()),'final':float(final.mean()),
        'mean_forgetting':float(forget.mean()),'worst_forgetting':float(forget.max()),
        'forgetting':forget.tolist(),'denominator':'All learned tasks, including last structural zero; Phase2.1 convention'}


def retention_gate(records,expected_seeds):
    if not isinstance(records,list) or not records or not isinstance(expected_seeds,(list,tuple)) or not expected_seeds:
        raise AuditViolation('Nonempty records and preregistered seed inventory required')
    if any(type(seed) is not int for seed in expected_seeds) or len(set(expected_seeds))!=len(expected_seeds):
        raise AuditViolation('Distinct integer seeds required')
    seeds=[r.get('seed') for r in records]
    if any(type(seed) is not int for seed in seeds) or len(set(seeds))!=len(seeds) or set(seeds)!=set(expected_seeds):
        raise AuditViolation('Missing/duplicate/unregistered seed')
    rows=[]
    for row in records:
        metrics=retention_metrics(row['matrix'])
        checks={'acquisition':metrics['acquisition']>=.65,'final':metrics['final']>=.65,
                'mean_forgetting':metrics['mean_forgetting']<=.03,'worst_forgetting':metrics['worst_forgetting']<=.05}
        rows.append({'seed':row['seed'],**metrics,'checks':checks,'numeric_gate_pass':all(checks.values())})
    return {'per_seed':rows,'seeds_passing':sum(r['numeric_gate_pass'] for r in rows),'seed_count':len(rows),
            'all_seed_numeric_gate_pass':all(r['numeric_gate_pass'] for r in rows),
            'release_approved':False,'scope':'Numerical retention gate only; provenance/causal/runtime/approval remain separate'}


def audit_p1_summaries(raw):
    """Reject known failures, but never certify a learner using summaries alone."""
    if not isinstance(raw,dict):raise AuditViolation('Raw result object required')
    results={}
    for task in TASKS:
        data=raw.get('p1_results',{}).get(task,{})
        learned=data.get('learned_mtsg',{})
        direct=learned.get('direct_acc_mean');lower=learned.get('probe_gain_lower95')
        errors=[];missing=[]
        if direct is None:missing.append('direct_acc_mean')
        else:
            finite_fraction(direct,'direct_acc_mean')
            if direct<.65:errors.append('direct_below_65')
        if lower is None:missing.append('probe_gain_lower95')
        elif type(lower) not in (int,float) or not math.isfinite(lower) or not -1<=lower<=1:
            raise AuditViolation('Finite signed gain interval required')
        elif lower<.05:errors.append('master_probe_gain_lower95_below_5pp')
        for control in ('frozen_hidden','readout_only','zero_signal','shuffled_signal'):
            if not data.get(control):missing.append(control)
        if not raw.get('per_seed_records') and not raw.get('trials'):missing.append('per_seed_raw_records')
        results[task]={'status':'FAIL' if errors else 'INSUFFICIENT_EVIDENCE','failures':errors,
            'missing':missing,'pass':False,
            'scope':'Master-gate compliance audit; does not rewrite historical branch criteria or certify from means'}
    return {'tasks':results,'overall_pass':False,'policy':'Master7.2, no ceiling exceptions or missing-data PASS'}


def require_injective_codebook(codebook):
    groups=defaultdict(list)
    if not isinstance(codebook,dict) or not codebook:raise AuditViolation('Nonempty explicit codebook required')
    for key,code in codebook.items():
        arr=np.asarray(code,dtype=float)
        if arr.ndim!=1 or not len(arr) or not np.isfinite(arr).all():raise AuditViolation('Finite code vector required')
        groups[tuple(arr.tolist())].append(key)
    collisions=[keys for keys in groups.values() if len(keys)>1]
    if collisions:raise AuditViolation('Ambiguous query codes: '+repr(collisions))
    return True


def binary_query_ambiguity_bound(codebook):
    """Uniform requested key, iid fair stored binary values, noiseless writes known.

    Exact Bayes ceiling due ONLY to duplicate query codes; not model accuracy.
    """
    keys=list(codebook)
    if not 1<=len(keys)<=12:raise AuditViolation('Finite1..12key enumeration required')
    groups=defaultdict(list)
    for i,key in enumerate(keys):groups[tuple(codebook[key])].append(i)
    score=0.
    for values in itertools.product((0,1),repeat=len(keys)):
        score+=sum(max(sum(values[i] for i in group),len(group)-sum(values[i] for i in group)) for group in groups.values())/len(keys)
    return {'noiseless_uniform_key_binary_value_ceiling':score/(2**len(keys)),
            'query_collision_groups':[[keys[i] for i in group] for group in groups.values() if len(group)>1],
            'scope':'Model-free conditional-label ambiguity; does not estimate learned hidden capacity'}
