"""Scalar independent check of the approved12cluster statistics and P1 gates."""
import math
import numpy as np


def reference_statistics(values):
    values=[float(v) for v in values]
    if len(values)!=12 or not all(math.isfinite(v) and abs(v)<=1 for v in values):raise ValueError('12finite fraction effects required')
    observed=abs(sum(values)/12);count=0
    for mask in range(4096):
        total=sum(value if mask&(1<<i) else -value for i,value in enumerate(values))
        if abs(total/12)>=observed-1e-12:count+=1
    rng=np.random.default_rng(919);samples=rng.integers(0,12,size=(10000,12)).tolist()
    means=sorted(sum(values[i] for i in draw)/12 for draw in samples)
    def quantile(q):
        position=q*(len(means)-1);left=math.floor(position);right=math.ceil(position)
        return means[left]+(means[right]-means[left])*(position-left)
    return {'mean':sum(values)/12,'lower95':quantile(.025),'upper95':quantile(.975),'exact_two_sided_p':count/4096}


def check_statistics_and_gate(summary):
    if not summary.get('complete'):raise ValueError('Complete numeric result required')
    checked=0
    for budget,tasks in summary['statistics'].items():
        for task,data in tasks.items():
            for name,statistics in data['contrasts'].items():
                reference=reference_statistics(statistics['effects'])
                for field,value in reference.items():
                    if abs(value-statistics[field])>1e-12:raise ValueError('Scalar statistical crosscheck mismatch: '+task+'/'+name+'/'+field)
                checked+=1
    p=summary['primary_p'];ordered=sorted(p,key=p.get);adjusted={};previous=0.
    if len(p)!=7:raise ValueError('Seven locked primary tests required')
    for i,name in enumerate(ordered):
        previous=max(previous,min(1.,(7-i)*p[name]));adjusted[name]=previous
        if abs(previous-summary['primary_holm_p'][name])>1e-12:raise ValueError('Holm reference mismatch')
    qualifying=[]
    for task in ('strict_binding','xor','distractor_binding'):
        data=summary['statistics']['equal_updates'][task];c=data['contrasts']
        good=data['arm_means']['stopped_correct']['id_direct']>=.65 and c['hidden']['lower95']>=.05
        good=good and c['hidden']['mean']>0 and adjusted[task+'/hidden']<=.05
        good=good and c['routing']['lower95']>0 and adjusted[task+'/routing']<=.05
        good=good and c['ood_direct_noninferiority']['lower95']>=-.03 and c['ood_probe_noninferiority']['lower95']>=-.03
        if good:qualifying.append(task)
        if bool(good)!=summary['task_gates'][task]['counts_toward_master_p1']:raise ValueError('Task gate crosscheck mismatch')
    order=summary['statistics']['equal_updates']['order']
    protected=order['arm_means']['stopped_correct']['id_direct']>=.65 and order['contrasts']['own_direct']['lower95']>=-.03
    passed=len(qualifying)>=2 and protected and summary['controls_valid'] is True
    if bool(passed)!=summary['master_p1_pass']:raise ValueError('Independent overall gate mismatch')
    return {'status':'PASS','statistical_contrasts_crosschecked':checked,'method':'Independent scalar bootstrap/permutation/quantile/Holm/gate arithmetic',
            'qualifying_tasks':qualifying,'master_p1_numeric_pass':bool(passed)}
