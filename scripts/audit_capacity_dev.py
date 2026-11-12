"""Read-only independent scalar/grid/seed/budget audit of approved capacity DEV."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,math,os,statistics,subprocess,sys,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.capacity import CapacityOwn,CapacityACR
from scarx.evaluation.acr import AdaptiveCreditRouting
from scarx.evaluation.r8_matched_credit import PEER
from scarx.evaluation.r8_matched_study import TASKS,SEEDS,event_oracle

ARMS=('own','groups_up','readout_only','groups_shuffle');HEIGHTS=(48,64,128)
METRICS=('id_direct','id_probe','ood_direct','ood_probe')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(raw):return hashlib.sha256(raw).hexdigest()
def accuracy(p,y):
    if len(p)!=len(y) or not y or any(type(v)is not int or v not in (0,1) for v in p+y):raise ValueError('Binary prediction schema')
    return sum(a==b for a,b in zip(p,y))/len(y)
def statistics3(values):
    a=list(map(float,values))
    if len(a)!=3 or not all(math.isfinite(x) for x in a):raise ValueError('Three independent seed clusters required')
    mean=statistics.mean(a);boots=sorted(statistics.mean(x) for x in itertools.product(a,repeat=3))
    null=[abs(sum(x*s for x,s in zip(a,ss))/3) for ss in itertools.product((-1,1),repeat=3)]
    return {'effects':a,'mean':mean,'lower95':boots[math.ceil(.025*27)-1],'upper95':boots[math.ceil(.975*27)-1],
       'exact_p':sum(v>=abs(mean)-1e-12 for v in null)/8,'scope':'Exploratory3seed, descriptive CI; no confirmation significance'}
def eigen_probe(fit,y,test):
    mean=fit.mean(0);scale=fit.std(0)+1e-4
    a=np.column_stack(((fit-mean)/scale,np.ones(len(fit))))
    b=np.column_stack(((test-mean)/scale,np.ones(len(test))))
    values,vectors=np.linalg.eigh(a.T@a+.01*np.eye(a.shape[1]));rhs=np.zeros((a.shape[1],2))
    for v,target in zip(a,y):rhs[:,int(target)]+=v
    return np.argmax(b@(vectors@((vectors.T@rhs)/values[:,None])),axis=1).tolist()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run=(ROOT/args.run).resolve()
    if run.parent!=ROOT/'runs' or (run/'audit.json').exists():raise ValueError('Unique owned audit required')
    started=time.perf_counter();deadline=started+600;manifest=read(run/'manifest.json');result=read(run/'result.json')
    if result['error'] is not None or (result['new_jobs'],result['row_count'],result['reused_h48_cells'])!=(204,252,48):raise ValueError('Incomplete capacity grid')
    if result['wall_seconds']>manifest['worker_limit_seconds']:raise ValueError('Worker time cap exceeded')
    main_sha=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()
    if not result['main_unchanged'] or main_sha!=manifest['main_before']:raise ValueError('Main changed')
    resource=result['resources']
    if resource['measurement_errors'] or resource['observed_over_limit'] or max(resource['peak_rss_bytes'],resource['peak_commit_or_vms_bytes'],resource['os_process_peak_working_set_bytes'],resource['os_process_peak_pagefile_commit_bytes'])>2_000_000_000:raise ValueError('Resource contract failed')
    if manifest['seeds']!=list(SEEDS) or manifest['height_grid']!=list(HEIGHTS) or result['fresh_confirmation'] or result['p2'] or result['production_ready']:raise ValueError('Study scope changed')
    for path,h in manifest['source_lock']['files'].items():
        if sha((ROOT/path).read_bytes())!=h:raise ValueError('Locked source changed: '+path)
    for path,h in manifest['source_lock']['peer'].items():
        if sha((PEER/path).read_bytes())!=h:raise ValueError('Peer source changed')
    parent=ROOT.parent/'scarx-acr-20261112'/'runs'/manifest['parent_run']
    if sha((parent/'manifest.json').read_bytes())!=manifest['parent_manifest_sha256'] or sha((parent/'audit-v1.json').read_bytes())!=manifest['parent_audit_sha256']:raise ValueError('H48 source receipt changed')
    copied=read(run/'reused-h48-files.json')
    if len(copied)!=192:raise ValueError('H48 copied file grid incomplete')
    for destination,receipt in copied.items():
        dest=(run/destination).resolve();source=Path(receipt['source_path']).resolve()
        if not dest.is_relative_to(run/'reused-h48') or not source.is_relative_to(parent/'jobs'):raise ValueError('Unsafe H48 copy receipt')
        if sha(dest.read_bytes())!=receipt['sha256'] or sha(source.read_bytes())!=receipt['sha256']:raise ValueError('Copied H48 bytes changed')
    bank={};labels_checked=0
    for name,h in read(run/'dataset-hashes.json').items():
        path=run/'datasets'/name
        if sha(path.read_bytes())!=h:raise ValueError('Dataset changed')
        task=name.rsplit('-',1)[0]
        with np.load(path,allow_pickle=False) as z:bank[name[:-4]]={k:z[k] for k in z.files}
        d=bank[name[:-4]]
        for split in ('train','refit','id','ood'):
            oracle,_=event_oracle(task,d[split+'_x'])
            if not np.array_equal(oracle,d[split+'_y']):raise ValueError('Independent event-label oracle mismatch')
            labels_checked+=len(oracle)
    if len(bank)!=12:raise ValueError('Data grid changed')
    expected={(h,a,t,s,'equal_updates') for h,a,t,s in itertools.product(HEIGHTS,ARMS,TASKS,SEEDS)}
    expected|={(h,a,t,s,'equal_cpu') for h,a,t,s in itertools.product(HEIGHTS,ARMS[:3],TASKS,SEEDS)}
    rows={};models=0;scored=0;replayed=0;probes=0;history_count=0;initials={};within_h_pairs=0
    files=list((run/'jobs').glob('*/result.json'))+list((run/'reused-h48').glob('*/capacity-result.json'))
    for file in sorted(files):
        if time.perf_counter()>deadline:raise TimeoutError('Capacity audit10minute limit')
        r=read(file);key=tuple(r[k] for k in ('height','arm','task','seed','budget'))
        if key in rows or key not in expected:raise ValueError('Duplicate/unregistered capacity cell')
        rows[key]=r;d=bank[f"{r['task']}-{r['seed']}"]
        for split in ('id','ood'):
            p=r['predictions'][split];y=d[split+'_y'].tolist()
            if p['target']!=y:raise ValueError('Labels differ from frozen data')
            for metric in ('direct','probe'):
                if accuracy(p[metric],y)!=r['metrics'][split+'_'+metric]:raise ValueError('Scalar score mismatch')
                scored+=len(y)
        for name,p in r['interventions'].items():
            if name not in ('revert_in','revert_rec','revert_both') or p['target']!=d['id_y'].tolist():raise ValueError('Intervention schema')
            for metric in ('direct','probe'):
                if accuracy(p[metric],p['target'])!=r['metrics'][name+'_'+metric]:raise ValueError('Scalar causal metric mismatch')
                scored+=len(p['target'])
        folder=run/r['folder'];reused=r['reused'];model_file=folder/('model-0.zip' if reused else 'model.zip');initial_file=folder/('initial-0.zip' if reused else 'initial.zip')
        blob=model_file.read_bytes();initial_blob=initial_file.read_bytes()
        if sha(blob)!=r['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        cls=AdaptiveCreditRouting if reused else CapacityACR if r['arm'].startswith('groups') else CapacityOwn
        m=cls.loads(blob);initial=cls.loads(initial_blob);models+=1
        if m.dumps()!=blob:raise ValueError('Checkpoint roundtrip mismatch')
        if reused:
            oldrow=read(folder/'result-0.json')
            if sha(initial_blob)!=oldrow['initial_checkpoint_sha256'] or oldrow['metrics']!=r['metrics'] or oldrow['predictions']!=r['predictions'] or oldrow['interventions']!=r['interventions']:raise ValueError('H48 copied results differ')
            rng=oldrow['signal_rng_state']
        else:
            if sha(initial_blob)!=r['initial_sha256']:raise ValueError('Initial checkpoint changed')
            rng=r['signal_rng_state']
        if m.signal_rng.bit_generator.state!=rng:raise ValueError('RNG receipt mismatch')
        if m.config.hidden!=r['height']:raise ValueError('Actual neuron count differs')
        size=(944 if r['arm'].startswith('groups') else 596)*r['height']+864
        if r['array_bytes']!=size:raise ValueError('Capacity array budget differs')
        if not reused and m.memory_bytes()!=size:raise ValueError('Measured array budget mismatch')
        if r['budget']=='equal_updates':
            prior=initials.setdefault((r['height'],r['task'],r['seed']),initial)
            for name,value in prior.core.arrays().items():np.testing.assert_array_equal(value,initial.core.arrays()[name])
            for name in ('f_mean','f_var','syn_trace','syn_energy'):np.testing.assert_array_equal(getattr(prior,name),getattr(initial,name))
            within_h_pairs+=1
        with np.load(folder/('learning-0.npz' if reused else 'learning.npz'),allow_pickle=False) as z:history=z['history']
        n=r['training']['episodes'];history_count+=n
        expected_hidden=0 if r['arm']=='readout_only' else n
        if (m.episodes_learned,m.hidden_updates,m.core.updates,m.total_ticks)!=(n,expected_hidden,n,n*12):
            raise ValueError('Actual model update/tick counters differ from receipt')
        if history.shape!=(n,4 if reused else 5) or not np.isfinite(history).all() or history[:,0].tolist()!=list(range(1,n+1)):raise ValueError('Learning history schema/count')
        if r['budget']=='equal_updates' and n!=1024:raise ValueError('Primary updates changed')
        if r['budget']=='equal_cpu' and (n>4096 or (history[-1,-1]<.5 and n!=4096)):raise ValueError('CPU cutoff changed')
        if history[-1,-1]!=r['training']['train_cpu_seconds']:raise ValueError('CPU receipt mismatch')
        if np.any(np.diff(history[:,-1])<0):raise ValueError('CPU history moved backwards')
        if r['arm']=='readout_only' and (m.hidden_updates or np.any(history[:,2]) or not np.array_equal(m.core.w_in,initial.core.w_in) or not np.array_equal(m.core.w_rec,initial.core.w_rec)):raise ValueError('Frozen hidden moved')
        for split in ('id','ood'):
            if m.predict_many(d[split+'_x'][:8]).tolist()!=r['predictions'][split]['direct'][:8]:raise ValueError('Hard ALIF direct replay mismatch')
            replayed+=8
        if r['budget']=='equal_updates' and r['arm'] in ('own','groups_up','readout_only'):
            fit=m.features_many(d['refit_x'])
            for split in ('id','ood'):
                if eigen_probe(fit,d['refit_y'],m.features_many(d[split+'_x'][:8]))!=r['predictions'][split]['probe'][:8]:raise ValueError('Independent eigenprobe differs')
                probes+=8
    if set(rows)!=expected:raise ValueError('Missing capacity cells')
    def row(h,arm,task,seed,budget='equal_updates'):return rows[(h,arm,task,seed,budget)]
    for r in rows.values():
        if r['budget']=='equal_updates':
            ro=row(r['height'],'readout_only',r['task'],r['seed'])
            if r['interventions']['revert_both']['probe']!=ro['predictions']['id']['probe']:raise ValueError('Full revert failed frozen representation recovery')
    means={str(h):{a:{t:{k:statistics.mean(row(h,a,t,s)['metrics'][k] for s in SEEDS) for k in METRICS} for t in TASKS} for a in ARMS} for h in HEIGHTS}
    cpu_means={str(h):{a:{t:{k:statistics.mean(row(h,a,t,s,'equal_cpu')['metrics'][k] for s in SEEDS) for k in METRICS} for t in TASKS} for a in ARMS[:3]} for h in HEIGHTS}
    contrasts={};gates={};shuffle_contrasts={}
    for h in HEIGHTS:
        shuffle_contrasts[str(h)]={}
        for t in TASKS:
            shuffle_contrasts[str(h)][t]={k:statistics3([row(h,'groups_up',t,s)['metrics'][k]-row(h,'groups_shuffle',t,s)['metrics'][k] for s in SEEDS]) for k in METRICS}
    for h in (64,128):
        gates[str(h)]={};contrasts[str(h)]={}
        for arm in ('own','groups_up','readout_only'):
            bytask={};contrasts[str(h)][arm]={};direct_families=hidden_families=0;preserved=True;causal_improving=True
            for t in TASKS:
                effects={k:[row(h,arm,t,s)['metrics'][k]-row(48,arm,t,s)['metrics'][k] for s in SEEDS] for k in METRICS}
                contrasts[str(h)][arm][t]={k:statistics3(v) for k,v in effects.items()}
                c=contrasts[str(h)][arm][t];m=means[str(h)][arm][t]
                hidden=[(row(h,arm,t,s)['metrics']['id_probe']-row(h,'readout_only',t,s)['metrics']['id_probe'])-
                        (row(48,arm,t,s)['metrics']['id_probe']-row(48,'readout_only',t,s)['metrics']['id_probe']) for s in SEEDS]
                c['hidden_advantage_growth']=statistics3(hidden)
                causal=statistics.mean(row(h,arm,t,s)['metrics']['id_direct']-row(h,arm,t,s)['metrics']['revert_both_direct'] for s in SEEDS)
                if t!='order':
                    direct_gain=c['id_direct']['mean']>=.02
                    if direct_gain:direct_families+=1;causal_improving &= causal>=.03
                    if c['hidden_advantage_growth']['mean']>=.02:hidden_families+=1
                    preserved &= all(c[k]['mean']>=-.03 for k in METRICS)
                bytask[t]={'direct_gain':c['id_direct']['mean'],'hidden_advantage_growth':c['hidden_advantage_growth']['mean'],'causal_direct_drop':causal}
            protected=means[str(h)][arm]['order']['id_direct']>=.65 and contrasts[str(h)][arm]['order']['id_direct']['mean']>=-.03
            signal=arm!='readout_only' and direct_families>=2 and hidden_families>=2 and preserved and protected and causal_improving
            gates[str(h)][arm]={'by_task':bytask,'direct_improved_families':direct_families,'hidden_advantage_improved_families':hidden_families,
                'all_primary_preserved':bool(preserved),'protected_order':bool(protected),'causality_on_improving_families':bool(causal_improving),
                'capacity_dev_signal':bool(signal)}
    neural=any(gates[str(h)][a]['capacity_dev_signal'] for h in (64,128) for a in ('own','groups_up'))
    frozen=any(gates[str(h)]['readout_only']['direct_improved_families']>=2 and gates[str(h)]['readout_only']['all_primary_preserved'] and gates[str(h)]['readout_only']['protected_order'] for h in (64,128))
    raw=any(gates[str(h)][a]['direct_improved_families']>0 for h in (64,128) for a in ('own','groups_up'))
    decision='CAPACITY_DEV_SIGNAL' if neural else 'CAPACITY_ONLY_SIGNAL' if frozen else 'MIXED' if raw else 'NO_CLEAR_CAPACITY_GAIN'
    summary={'integrity':'PASS','decision':decision,'means':means,'equal_cpu_means':cpu_means,'contrasts':contrasts,'gates':gates,
       'groups_vs_signal_shuffle':shuffle_contrasts,
       'record_count':len(rows),'new_train_jobs':204,'reused_h48_cells':48,'predictions_rescored':scored,'checkpoints_roundtripped':models,
       'forward_replay_predictions':replayed,'independent_eigenprobe_predictions':probes,'initial_pair_checks':within_h_pairs,
       'labels_oracle_checked':labels_checked,'accepted_history_updates':history_count,'new_training_updates_in_audit':0,
       'fresh_confirmation':False,'p2_tested':False,'production_ready':False,'main_unchanged':True,'resource_contract_verified':True,
       'audit_wall_seconds':time.perf_counter()-started,'source_commit':manifest['source_commit'],
       'limitations':'Exploratory3seed; fixedfan8 changesdensity; gammaresampled; initacrossHpairedseed/distribution notsameweights; no lifelong/knowledge/language/vision claim'}
    with (run/'audit.json').open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,allow_nan=False,indent=2)
    with (run/'per-seed-results.csv').open('x',encoding='utf-8',newline='') as f:
        fields=['height','arm','task','seed','budget','reused',*METRICS,'probe_gain_vs_frozen','causal_direct_drop','episodes','cpu_seconds','array_bytes','folder']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows.values():
            ro=row(r['height'],'readout_only',r['task'],r['seed'],r['budget'])
            w.writerow({k:r[k] for k in fields[:6]}|{k:r['metrics'][k] for k in METRICS}|
             {'probe_gain_vs_frozen':r['metrics']['id_probe']-ro['metrics']['id_probe'],
              'causal_direct_drop':r['metrics']['id_direct']-r['metrics']['revert_both_direct'],'episodes':r['training']['episodes'],
              'cpu_seconds':r['training']['train_cpu_seconds'],'array_bytes':r['array_bytes'],'folder':r['folder']})
    print(json.dumps({k:summary[k] for k in ('integrity','decision','record_count','predictions_rescored','audit_wall_seconds')}),flush=True)
if __name__=='__main__':main()
