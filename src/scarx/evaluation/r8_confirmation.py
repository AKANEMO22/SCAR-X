"""Frozen constants and fail-closed statistics for approved12seed confirmation."""
import itertools,math
import numpy as np
from .r8_matched_study import holm

SEEDS=tuple(2026111201+6*i for i in range(12))
PRIMARY_TASKS=('strict_binding','xor','distractor_binding')
TASKS=PRIMARY_TASKS+('order',)
ARMS=('stopped_correct','own_reset_correct','stopped_shuffle','stopped_zero','stopped_readout_only')
BUDGETS=('equal_updates','equal_cpu')


def arms_for(task):
    if task not in TASKS:raise ValueError('Unregistered confirmation task')
    return ARMS if task in PRIMARY_TASKS else ARMS[:2]


def expected_cells():return [(t,s,a,b) for t in TASKS for s in SEEDS for a in arms_for(t) for b in BUDGETS]


def cluster_statistics(values):
    a=np.asarray(values,np.float64)
    if a.shape!=(12,) or not np.isfinite(a).all() or np.any(np.abs(a)>1):raise ValueError('12finite signed fraction effects required')
    rng=np.random.default_rng(919);draws=a[rng.integers(0,12,size=(10000,12))].mean(1)
    signs=np.asarray(list(itertools.product((-1,1),repeat=12)),np.int8)
    observed=abs(float(a.mean()));null=np.abs((signs*a).mean(1))
    return {'mean':float(a.mean()),'lower95':float(np.quantile(draws,.025)),
        'upper95':float(np.quantile(draws,.975)),'exact_two_sided_p':float(np.mean(null>=observed-1e-12)),
        'seed_count':12,'seed_ids':list(SEEDS),'effects':a.tolist(),'bootstrap_resamples':10000,'bootstrap_rng':919,
        'method':'Paired seed-cluster bootstrap percentile CI; exact4096sign-flips, no sample-level pseudo-replication'}


def summarize_confirmation(rows):
    expected=set(expected_cells());index={}
    for row in rows:
        key=tuple(row.get(k) for k in ('task','seed','arm','budget'))
        if key not in expected or key in index:raise ValueError('Duplicate/unregistered confirmation cell')
        m=row.get('metrics',{})
        for field in ('id_direct','id_probe','ood_direct','ood_probe'):
            value=m.get(field)
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid accuracy metric')
        index[key]=row
    missing=sorted(expected-set(index))
    if missing:return {'complete':False,'missing_cells':missing,'master_p1_pass':False,'decision':'INCOMPLETE_NO_PASS'}
    statistics={};pvalues={};means={};controls_valid=True
    for budget in BUDGETS:
        by_task={}
        for task in TASKS:
            task_rows={(s,a):index[(task,s,a,budget)] for s in SEEDS for a in arms_for(task)}
            arm_means={a:{k:float(np.mean([task_rows[(s,a)]['metrics'][k] for s in SEEDS])) for k in ('id_direct','id_probe','ood_direct','ood_probe')} for a in arms_for(task)}
            def contrast(left,right,metric):
                return cluster_statistics([task_rows[(s,left)]['metrics'][metric]-task_rows[(s,right)]['metrics'][metric] for s in SEEDS])
            contrasts={'reset':contrast('stopped_correct','own_reset_correct','id_probe'),
                'own_direct':contrast('stopped_correct','own_reset_correct','id_direct'),
                'ood_direct_noninferiority':contrast('stopped_correct','own_reset_correct','ood_direct'),
                'ood_probe_noninferiority':contrast('stopped_correct','own_reset_correct','ood_probe')}
            if task in PRIMARY_TASKS:
                contrasts['hidden']=contrast('stopped_correct','stopped_readout_only','id_probe')
                contrasts['routing']=contrast('stopped_correct','stopped_shuffle','id_probe')
                contrasts['zero']=contrast('stopped_correct','stopped_zero','id_probe')
                for s in SEEDS:
                    zero=task_rows[(s,'stopped_zero')];readout=task_rows[(s,'stopped_readout_only')]
                    for row in (zero,readout):
                        if row.get('hidden_unchanged') is not True or row.get('initial_features_match') is not True:controls_valid=False
                    # Same-update controls must be exact. A fixed CPU ceiling
                    # can yield different episode counts/head fits; hidden
                    # invariance/raw-feature checks still apply in both budgets.
                    if budget=='equal_updates' and zero.get('predictions')!=readout.get('predictions'):controls_valid=False
                if budget=='equal_updates':
                    pvalues[task+'/hidden']=contrasts['hidden']['exact_two_sided_p']
                    pvalues[task+'/routing']=contrasts['routing']['exact_two_sided_p']
            if task=='xor' and budget=='equal_updates':pvalues['xor/reset']=contrasts['reset']['exact_two_sided_p']
            by_task[task]={'arm_means':arm_means,'contrasts':contrasts}
        statistics[budget]=by_task
    if len(pvalues)!=7:raise ValueError('Exactly7primary tests required')
    adjusted=holm(pvalues);gates={}
    for task in PRIMARY_TASKS:
        t=statistics['equal_updates'][task];c=t['contrasts']
        checks={'direct_gte_65':t['arm_means']['stopped_correct']['id_direct']>=.65,
            'hidden_lower95_gte_5pp':c['hidden']['lower95']>=.05,
            'positive_readout_contrast_holm':c['hidden']['mean']>0 and adjusted[task+'/hidden']<=.05,
            'routing_lower95_positive_holm':c['routing']['lower95']>0 and adjusted[task+'/routing']<=.05,
            'ood_direct_noninferior_3pp':c['ood_direct_noninferiority']['lower95']>=-.03,
            'ood_probe_noninferior_3pp':c['ood_probe_noninferiority']['lower95']>=-.03}
        gates[task]={'checks':checks,'counts_toward_master_p1':all(checks.values())}
    order=statistics['equal_updates']['order']
    protection={'direct_gte_65':order['arm_means']['stopped_correct']['id_direct']>=.65,
        'direct_lower95_noninferior_3pp':order['contrasts']['own_direct']['lower95']>=-.03}
    count=sum(g['counts_toward_master_p1'] for g in gates.values());protected=all(protection.values())
    passed=count>=2 and protected and controls_valid
    xor=statistics['equal_updates']['xor']['contrasts']['reset']
    return {'complete':True,'statistics':statistics,'primary_p':pvalues,'primary_holm_p':adjusted,
        'task_gates':gates,'qualifying_primary_tasks':count,'protected_order':protection,'controls_valid':controls_valid,
        'master_p1_pass':passed,'xor_reset_superiority':xor['lower95']>0 and adjusted['xor/reset']<=.05,
        'decision':'P1_NUMERIC_GATE_PASS_REQUIRES_INTEGRITY_AUDIT' if passed else 'P1_FAIL_RESEARCH_ONLY',
        'p2_tested':False,'production_ready':False}
