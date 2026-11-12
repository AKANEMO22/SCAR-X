"""Finite exploratory ACR evaluation, outside candidate and with raw evidence."""
import copy,hashlib
import numpy as np
from .acr import AdaptiveCreditRouting as ACR
from .r8_matched_study import TASKS,SEEDS,dataset,refit_predict,direct_predict

ENVS=TASKS+('joint',)
METRICS=('id_direct','id_probe','ood_direct','ood_probe')

def sha(raw):return hashlib.sha256(raw).hexdigest()

def build_data(seed):
    out={t:dataset(t,seed) for t in TASKS};joint={}
    for split in ('train','refit','id','ood'):
        n=256 if split=='train' else 64 if split=='refit' else 512
        xs=[];ys=[];summaries=[];families=[]
        for j,t in enumerate(TASKS):
            x,y,s=out[t][split];xs.extend(x[:n]);ys.extend(y[:n]);summaries.extend(s[:n]);families.extend([j]*n)
        order=np.arange(len(xs))
        if split in ('train','refit'):np.random.default_rng(seed+900+(split=='refit')).shuffle(order)
        joint[split]=(np.asarray(xs,np.float32)[order],np.asarray(ys,np.int32)[order],[summaries[i] for i in order])
        joint[split+'_families']=np.asarray(families,np.int32)[order]
    out['joint']=joint
    return out

def evaluate(model,data):
    before=model.core.fingerprint();fitx,fity,_=data['refit'];fit=model.features_many(fitx)
    metrics={};predictions={};interventions={};family={}
    for split in ('id','ood'):
        x,y,_=data[split];f=model.features_many(x)
        direct=direct_predict(model,f);probe=refit_predict(fit,fity,f)
        metrics[split+'_direct']=float(np.mean(direct==y));metrics[split+'_probe']=float(np.mean(probe==y))
        predictions[split]={'direct':direct.tolist(),'probe':probe.tolist(),'target':y.tolist()}
        if split+'_families' in data:
            for j,t in enumerate(TASKS):
                mask=data[split+'_families']==j;family.setdefault(t,{})
                family[t][split+'_direct']=float(np.mean(direct[mask]==y[mask]))
                family[t][split+'_probe']=float(np.mean(probe[mask]==y[mask]))
    for name,rin,rrec in [('revert_in',True,False),('revert_rec',False,True),('revert_both',True,True)]:
        m=copy.deepcopy(model);m.revert_hidden_weights(rin,rrec)
        f=m.features_many(data['id'][0]);direct=direct_predict(m,f)
        probe=refit_predict(m.features_many(fitx),fity,f);y=data['id'][1]
        interventions[name]={'direct':direct.tolist(),'probe':probe.tolist(),'target':y.tolist()}
        metrics[name+'_direct']=float(np.mean(direct==y));metrics[name+'_probe']=float(np.mean(probe==y))
    if before!=model.core.fingerprint():raise ValueError('Evaluation mutated weights')
    return metrics,predictions,interventions,family

def means(rows,profiles):
    out={}
    for p in profiles:
        out[p]={}
        for t in ENVS:
            condition='shuffled_signal' if p=='state_signal_shuffle' else 'readout_only' if p=='readout_only' else 'learned_mtsg'
            cells=[r for r in rows if r['profile']==p and r['task']==t and r['budget']=='equal_updates' and r['condition']==condition]
            if len(cells)!=3:raise ValueError('Incomplete DEV cell group')
            out[p][t]={k:float(np.mean([r['metrics'][k] for r in cells])) for k in METRICS}
    return out

def nominate(rows,profiles):
    m=means(rows,profiles)
    def rank(p):
        diffs=[m[p][t][k]-max(m[e][t][k] for e in ('static_0','static_100'))
               for t in TASKS[:3] for k in ('id_direct','id_probe')]
        direct=[m[p][t]['id_direct']-max(m[e][t]['id_direct'] for e in ('static_0','static_100')) for t in TASKS[:3]]
        return (min(diffs),float(np.mean(direct)),p.startswith('static_'))
    # Baselines are reference profiles, not evidence of a new candidate.
    candidates=sorted(p for p in profiles if p not in ('static_0','static_100','readout_only') and 'shuffle' not in p and p!='random_alpha')
    return max(candidates,key=rank)

def broad_checks(rows,p):
    m=means(rows,('static_0','static_100','readout_only',p));out={};improved=0
    for t in TASKS[:3]:
        x=m[p][t];own=m['static_0'][t];stop=m['static_100'][t];ro=m['readout_only'][t]
        group=[r for r in rows if r['profile']==p and r['task']==t and r['budget']=='equal_updates' and r['condition']=='learned_mtsg']
        causal=float(np.mean([r['metrics']['id_direct']-r['metrics']['revert_both_direct'] for r in group]))
        gain=x['id_probe']-ro['id_probe']
        goodgain=any(x[k]-max(own[k],stop[k])>=.02 for k in ('id_direct','id_probe'))
        if goodgain and gain>=.05 and causal>=.03:improved+=1
        out[t]={'direct65':x['id_direct']>=.65,
            'endpoint_preserved':all(x[k]-max(own[k],stop[k])>=-.03 for k in ('id_direct','id_probe')),
            'ood_preserved':all(x[k]-own[k]>=-.03 for k in ('ood_direct','ood_probe')),
            'hidden_probe_gain':gain,'causal_direct_drop':causal,'improved':goodgain}
    out['order_preserved']=m[p]['order']['id_direct']>=.65 and m[p]['order']['id_direct']-max(m['static_0']['order']['id_direct'],m['static_100']['order']['id_direct'])>=-.03
    joint={}
    for t in TASKS:
        def jm(profile):
            cells=[r for r in rows if r['profile']==profile and r['task']=='joint' and r['budget']=='equal_updates' and r['condition']=='learned_mtsg']
            return float(np.mean([r['family_metrics'][t]['id_direct'] for r in cells]))
        joint[t]=jm(p)-max(jm('static_0'),jm('static_100'))>=-.03
    out['joint_preserved']=joint;out['families_improved_with_causality']=improved
    out['broad_numeric']=improved>=2 and out['order_preserved'] and all(joint.values()) and all(out[t][k] for t in TASKS[:3] for k in ('direct65','endpoint_preserved','ood_preserved'))
    return out

def stress_cases(task,data):
    x,y,s=data['id'];x=x[:128].copy();y=y[:128].copy();out={}
    # Noise moved/amplified before final marker, which is semantic channel7.
    a=x.copy();a[:,:-1,6:8]*=3;out['heavy_noise']=(a,y)
    a=x.copy();a[:,:-1,[6,7]]=a[:,:-1,[7,6]];out['moved_noise']=(a,y)
    a=x.copy()
    for i,summary in enumerate(s[:128]):
        ticks=[w['tick'] for w in summary['writes']] if 'writes' in summary else [summary['first_tick'],summary.get('second_tick')]
        for tick in ticks:
            if tick is not None:a[i,tick,:6]*=.7
    out['event_amplitude']=(a,y)
    # Insert12zero ticks before query (10) or end marker (11); preserve events.
    at=10 if task in ('strict_binding','distractor_binding') else 11
    out['delay12']=(np.concatenate((x[:,:at],np.zeros((len(x),12,8),np.float32),x[:,at:]),1),y)
    if task in ('strict_binding','distractor_binding'):
        a=x.copy();a[:,10,[4,5]]=a[:,10,[5,4]]
        labels=[]
        for summary in s[:128]:
            values={w['key']:w['value'] for w in summary['writes']};labels.append(values[1-summary['query']])
        out['query_flip']=(a,np.asarray(labels,np.int32))
    return out
