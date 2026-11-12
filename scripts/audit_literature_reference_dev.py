"""Read-only scalar/provenance audit. No training; independently reconstruct gates."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,math,os,statistics,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit,PEER
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit
from scarx.evaluation.literature_eprop import LiteratureEPropALIF
from scarx.evaluation.literature_braintrace import BrainTraceDRTRL
from scarx.evaluation.r8_matched_study import SEEDS,TASKS

METHODS=('own','stop','eprop_matched','eprop_aligned','braintrace','readout_only')
METRICS=('id_direct','id_probe','ood_direct','ood_probe')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(raw):return hashlib.sha256(raw).hexdigest()
def accuracy(pred,y):
    if len(pred)!=len(y) or not y or any(type(a)is not int or a not in (0,1) for a in pred+y):raise ValueError('Binary prediction schema')
    return sum(a==b for a,b in zip(pred,y))/len(y)
def load_model(method,raw):
    cls=MatchedLocalCredit if method in ('own','readout_only') else StoppedResetCredit if method=='stop' else LiteratureEPropALIF if method.startswith('eprop_') else BrainTraceDRTRL
    return cls.loads(raw)
def stats(effects):
    a=list(map(float,effects))
    if len(a)!=3 or not all(math.isfinite(x) for x in a):raise ValueError('3seed independent clusters required')
    mean=statistics.mean(a);boot=sorted(statistics.mean(xs) for xs in itertools.product(a,repeat=3))
    null=[abs(sum(x*s for x,s in zip(a,signs))/3) for signs in itertools.product((-1,1),repeat=3)]
    return {'effects':a,'mean':mean,'lower95':boot[math.ceil(.025*27)-1],'upper95':boot[math.ceil(.975*27)-1],
       'exact_p':sum(x>=abs(mean)-1e-12 for x in null)/8,'scope':'Exploratory paired3seed; no confirmation significance'}

def eigen_probe(fit,y,test):
    mean=fit.mean(0);scale=fit.std(0)+1e-4
    a=np.column_stack(((fit-mean)/scale,np.ones(len(fit))))
    b=np.column_stack(((test-mean)/scale,np.ones(len(test))))
    eig,v=np.linalg.eigh(a.T@a+.01*np.eye(a.shape[1]));rhs=np.zeros((a.shape[1],2))
    for row,target in zip(a,y):rhs[:,int(target)]+=row
    weights=v@((v.T@rhs)/eig[:,None]);return np.argmax(b@weights,axis=1).tolist()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--version',choices=('v1','v2'),default='v2');args=ap.parse_args();run=(ROOT/args.run).resolve()
    audit_name='audit-'+args.version+'.json';table_name='per-seed-results'+('-v2' if args.version=='v2' else '')+'.csv'
    if run.parent!=ROOT/'runs' or (run/audit_name).exists() or (run/table_name).exists():raise ValueError('Unique owned audit only')
    began=time.perf_counter();deadline=began+600;manifest=read(run/'manifest.json');result=read(run/'result.json')
    if result['error'] is not None or result['completed_jobs']!=132:raise ValueError('Incomplete worker cannot PASS')
    lock=manifest['source_lock']
    for path,h in lock['files'].items():
        if sha((ROOT/path).read_bytes())!=h:raise ValueError('Source changed: '+path)
    for path,h in lock['peer'].items():
        if sha((PEER/path).read_bytes())!=h:raise ValueError('Peer changed')
    for repo,entry in lock['external'].items():
        for path,h in entry['files'].items():
            if sha((ROOT.parent.parent/'references'/repo/path).read_bytes())!=h:raise ValueError('External source changed')
    if manifest['seeds']!=list(SEEDS) or manifest['fresh_confirmation']:raise ValueError('Seed scope changed')
    data={}
    for name,h in read(run/'dataset-hashes.json').items():
        path=run/'datasets'/name
        if sha(path.read_bytes())!=h:raise ValueError('Dataset hash changed')
        with np.load(path,allow_pickle=False) as z:data[name[:-4]]={k:z[k] for k in z.files}
    if len(data)!=12:raise ValueError('Dataset grid incomplete')
    nominee=read(run/'nominee-lock.json')['method'];expected=set()
    for method,task,seed in itertools.product(METHODS,TASKS,SEEDS):expected.add((method,task,seed,'equal_updates',False))
    for task,seed in itertools.product(TASKS,SEEDS):expected.add((nominee,task,seed,'equal_updates',True))
    for method,task,seed in itertools.product(('own','stop',nominee,'readout_only'),TASKS,SEEDS):expected.add((method,task,seed,'equal_cpu',False))
    rows={};scored=0;replayed=0;probe_rows=0;initials={};loaded=0
    for path in sorted((run/'jobs').glob('*/result.json')):
        if time.perf_counter()>deadline:raise TimeoutError('Audit10minute cap')
        r=read(path);key=tuple(r[k] for k in ('method','task','seed','budget','shuffled'))
        if key in rows or key not in expected:raise ValueError('Duplicate/unregistered cell')
        rows[key]=r;d=data[f"{r['task']}-{r['seed']}"]
        for split in ('id','ood'):
            p=r['predictions'][split];y=d[split+'_y'].tolist()
            if p['target']!=y:raise ValueError('Stored target differs')
            for metric in ('direct','probe'):
                if accuracy(p[metric],y)!=r['metrics'][split+'_'+metric]:raise ValueError('Scalar accuracy mismatch')
                scored+=len(y)
        for name,p in r['interventions'].items():
            if name not in ('revert_in','revert_rec','revert_both') or p['target']!=d['id_y'].tolist():raise ValueError('Intervention schema')
            for metric in ('direct','probe'):
                if accuracy(p[metric],p['target'])!=r['metrics'][name+'_'+metric]:raise ValueError('Causal metric mismatch')
                scored+=len(p['target'])
        raw=(path.parent/'model.zip').read_bytes();initial=(path.parent/'initial.zip').read_bytes()
        if sha(raw)!=r['checkpoint_sha256'] or sha(initial)!=r['initial_checkpoint_sha256']:raise ValueError('Model hash changed')
        m=load_model(r['method'],raw);base=load_model(r['method'],initial);loaded+=1
        if m.dumps()!=raw:raise ValueError('Checkpoint roundtrip mismatch')
        if m.signal_rng.bit_generator.state!=r['signal_rng']:raise ValueError('RNG receipt mismatch')
        if r['budget']=='equal_updates' and not r['shuffled']:
            prior=initials.setdefault((r['task'],r['seed']),base)
            for name in prior.core.arrays():np.testing.assert_array_equal(base.core.arrays()[name],prior.core.arrays()[name])
            for name in ('f_mean','f_var'):np.testing.assert_array_equal(getattr(base,name),getattr(prior,name))
        with np.load(path.parent/'learning.npz',allow_pickle=False) as z:history=z['history']
        n=r['training']['episodes']
        if history.shape!=(n,5) or history[:,0].tolist()!=list(range(1,n+1)) or not np.isfinite(history).all():raise ValueError('Learning log schema')
        if r['budget']=='equal_updates' and n!=1024:raise ValueError('Primary budget changed')
        if r['budget']=='equal_cpu' and (n>4096 or (history[-1,4]<.5 and n!=4096)):raise ValueError('CPU stopping changed')
        if history[-1,4]!=r['training']['train_cpu_seconds'] or abs(float(history[:,2].mean())-r['mean_physical_update_norm'])>1e-15:raise ValueError('Resource/norm receipt mismatch')
        if r['method']=='readout_only':
            if m.hidden_updates or np.any(history[:,2]) or not np.array_equal(m.core.w_in,base.core.w_in) or not np.array_equal(m.core.w_rec,base.core.w_rec):raise ValueError('Frozen hidden changed')
        for split in ('id','ood'):
            if m.predict_many(d[split+'_x'][:8]).tolist()!=r['predictions'][split]['direct'][:8]:raise ValueError('Forward replay mismatch')
            replayed+=8
        if r['budget']=='equal_updates' and not r['shuffled'] and r['method'] in ('own','stop',nominee,'readout_only'):
            fit=m.features_many(d['refit_x'])
            for split in ('id','ood'):
                if eigen_probe(fit,d['refit_y'],m.features_many(d[split+'_x'][:16]))!=r['predictions'][split]['probe'][:16]:raise ValueError('Independent eigenprobe mismatch')
                probe_rows+=16
    if set(rows)!=expected:raise ValueError('Missing registered cell')
    def row(method,t,s,budget='equal_updates',shuffle=False):return rows[(method,t,s,budget,shuffle)]
    means={m:{t:{k:statistics.mean(row(m,t,s)['metrics'][k] for s in SEEDS) for k in METRICS} for t in TASKS} for m in METHODS}
    def rank(m):
        ds=[means[m][t][k]-max(means[e][t][k] for e in ('own','stop')) for t in TASKS[:3] for k in ('id_direct','id_probe')]
        direct=[means[m][t]['id_direct']-max(means[e][t]['id_direct'] for e in ('own','stop')) for t in TASKS[:3]]
        return min(ds),statistics.mean(direct)
    if max(sorted(('eprop_matched','eprop_aligned','braintrace')),key=rank)!=nominee:raise ValueError('Nominee differs from fixed global ranking')
    for r in rows.values():
        if r['budget']=='equal_updates':
            ro=row('readout_only',r['task'],r['seed'])
            if r['interventions']['revert_both']['probe']!=ro['predictions']['id']['probe']:raise ValueError('Full revert does not recover frozen representation')
    contrasts={};checks={};good=0;shuffle_good=0
    for t in TASKS[:3]:
        x=means[nominee][t];own=means['own'][t];stop=means['stop'][t];ro=means['readout_only'][t]
        contrasts[t]={}
        for name,ref,metric in [('vs_own_direct','own','id_direct'),('vs_stop_direct','stop','id_direct'),('vs_own_probe','own','id_probe'),('vs_stop_probe','stop','id_probe'),('hidden','readout_only','id_probe'),('ood_direct','own','ood_direct'),('ood_probe','own','ood_probe')]:
            contrasts[t][name]=stats([row(nominee,t,s)['metrics'][metric]-row(ref,t,s)['metrics'][metric] for s in SEEDS])
        causal=statistics.mean(row(nominee,t,s)['metrics']['id_direct']-row(nominee,t,s)['metrics']['revert_both_direct'] for s in SEEDS)
        directgain=x['id_direct']-max(own['id_direct'],stop['id_direct'])
        gain=x['id_probe']-ro['id_probe']
        if directgain>=.02 and x['id_direct']>=.65 and gain>=.05 and causal>=.03:good+=1
        for k in ('id_direct','id_probe'):
            contrasts[t]['shuffle_'+k]=stats([row(nominee,t,s)['metrics'][k]-row(nominee,t,s,shuffle=True)['metrics'][k] for s in SEEDS])
        if contrasts[t]['shuffle_id_probe']['mean']>=.02:shuffle_good+=1
        checks[t]={'direct65':x['id_direct']>=.65,'direct_gain_vs_better_endpoint':directgain,'probe_gain_vs_readout':gain,
             'causal_direct_drop':causal,'endpoint_preserved':all(x[k]-max(own[k],stop[k])>=-.03 for k in ('id_direct','id_probe')),
             'ood_preserved':all(x[k]-own[k]>=-.03 for k in ('ood_direct','ood_probe')),
             'shuffle_direct_preserved':contrasts[t]['shuffle_id_direct']['mean']>=-.03}
    order=means[nominee]['order']['id_direct'];protected=order>=.65 and order-max(means[e]['order']['id_direct'] for e in ('own','stop'))>=-.03
    promising=good>=2 and shuffle_good>=2 and protected and all(checks[t][k] for t in TASKS[:3] for k in ('direct65','endpoint_preserved','ood_preserved','shuffle_direct_preserved'))
    decision='REFERENCE_DEV_PROMISING' if promising else 'REFERENCE_MIXED' if any(checks[t]['direct_gain_vs_better_endpoint']>=.02 for t in TASKS[:3]) else 'NO_CLEAR_GAIN'
    summary={'integrity':'PASS','decision':decision,'nominee':nominee,'means':means,'contrasts':contrasts,'checks':checks,
       'improving_families_with_causality':good,'shuffle_families':shuffle_good,'protected_order':protected,
       'models':len(rows),'predictions_rescored':scored,'checkpoints_roundtripped':loaded,'forward_replay_predictions':replayed,
       'independent_eigenprobe_predictions':probe_rows,'source_commit':manifest['source_commit'],'new_training_updates':0,
       'fresh_confirmation':False,'production_ready':False,'norm_independent_advantage_proven':False,'audit_wall_seconds':time.perf_counter()-began}
    with (run/audit_name).open('x',encoding='utf-8') as stream:json.dump(summary,stream,ensure_ascii=False,allow_nan=False,indent=2)
    with (run/table_name).open('x',encoding='utf-8',newline='') as stream:
        fields=['method','task','seed','budget','shuffled',*METRICS,'probe_gain','causal_direct_drop','mean_update_norm','train_cpu_seconds','episodes','array_bytes']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows.values():
            ro=row('readout_only',r['task'],r['seed'],budget=r['budget'])
            writer.writerow({k:r[k] for k in fields[:5]}|{k:r['metrics'][k] for k in METRICS}|
              {'probe_gain':r['metrics']['id_probe']-ro['metrics']['id_probe'],
               'causal_direct_drop':r['metrics']['id_direct']-r['metrics']['revert_both_direct'],
               'mean_update_norm':r['mean_physical_update_norm'],'train_cpu_seconds':r['training']['train_cpu_seconds'],
               'episodes':r['training']['episodes'],'array_bytes':r['array_bytes']})
    print(json.dumps({k:summary[k] for k in ('integrity','decision','nominee','models','predictions_rescored','audit_wall_seconds')}),flush=True)

if __name__=='__main__':main()
