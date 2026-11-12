"""Independent read-only capacity audit. No learn calls, no candidate selection.

Literal ALIF replay, threshold-based event oracle, scalar metric/statistics and
SVD probe checks are separate from the registered runner's helpers.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,math,os,statistics,subprocess,sys,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.capacity import CapacityOwn,CapacityACR
from scarx.evaluation.acr import AdaptiveCreditRouting

SEEDS=(140011,140017,140023)
TASKS=('strict_binding','xor','distractor_binding','order')
ARMS=('own','groups_up','readout_only','groups_shuffle')
METRICS=('id_direct','id_probe','ood_direct','ood_probe')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def scalar_accuracy(a,b):
    if len(a)!=len(b) or not a or any(type(x)is not int or x not in (0,1) for x in a+b):raise ValueError('Binary scalar prediction schema')
    return sum(x==y for x,y in zip(a,b))/len(a)
def stat(a):
    if len(a)!=3 or any(not math.isfinite(x) for x in a):raise ValueError('3finite seed effects')
    boot=sorted(statistics.mean(v) for v in itertools.product(a,repeat=3));mean=statistics.mean(a)
    null=[abs(sum(x*s for x,s in zip(a,z))/3) for z in itertools.product((-1,1),repeat=3)]
    return {'effects':a,'mean':mean,'lower95':boot[math.ceil(.025*27)-1],'upper95':boot[math.ceil(.975*27)-1],
       'exact_two_sided_p':sum(x>=abs(mean)-1e-12 for x in null)/8,'seed_clusters':3}

def independent_labels(task,xs):
    out=[]
    for x in xs:
        if task in ('strict_binding','distractor_binding'):
            writes=np.flatnonzero((np.max(x[:10,:2],axis=1)>.7)&(np.max(x[:10,2:4],axis=1)>.7))
            if len(writes)!=2:raise ValueError('Ambiguous threshold oracle writes')
            values={int(np.argmax(x[t,:2])):int(np.argmax(x[t,2:4])) for t in writes}
            if set(values)!={0,1}:raise ValueError('Missing key in threshold oracle')
            out.append(values[int(np.argmax(x[10,4:6]))])
        elif task=='xor':
            first=np.flatnonzero(np.max(x[:,:2],axis=1)>.7);second=np.flatnonzero(np.max(x[:,2:4],axis=1)>.7)
            if len(first)!=1 or len(second)!=1:raise ValueError('Ambiguous threshold XOR oracle')
            out.append(int(np.argmax(x[first[0],:2]))^int(np.argmax(x[second[0],2:4])))
        else:
            ticks=np.flatnonzero(np.max(x[:,:2],axis=1)>.7)
            if len(ticks)!=2:raise ValueError('Ambiguous threshold order oracle')
            a=int(np.argmax(x[ticks[0],:2]));b=int(np.argmax(x[ticks[1],:2]))
            if a==b:raise ValueError('Order oracle repeated symbol')
            out.append(a)
    return out

def literal_features(model,xs):
    c=model.config;features=[]
    for x in xs:
        voltage=np.zeros(c.hidden,np.float32);adaptation=voltage.copy();z=voltage.copy();f=voltage.copy()
        for row in x:
            previous=z.copy();current=model.core.w_in@row+np.sum(model.core.w_rec*previous[model.core.indices],axis=1)
            voltage=c.leak*voltage+current;limit=c.threshold+c.adaptation_strength*adaptation
            z=(voltage>=limit).astype(np.float32);voltage-=z*limit
            adaptation=c.adaptation_decay*adaptation+z;f=c.feature_decay*f+(1-c.feature_decay)*z
        features.append(f)
    return np.asarray(features,np.float32)
def literal_predictions(model,features):
    scale=np.maximum(np.sqrt(model.f_var),.05).astype(np.float32)
    fhat=np.clip((features-model.f_mean)/scale,-5,5).astype(np.float32)
    return np.argmax(fhat@model.core.w_out.T+model.core.bias,axis=1).tolist()
def svd_probe(train,y,test):
    mean=train.mean(0);scale=train.std(0)+1e-4
    a=np.column_stack(((train-mean)/scale,np.ones(len(train))))
    b=np.column_stack(((test-mean)/scale,np.ones(len(test))))
    # Solve the same ridge objective through SVD, not runner's normal equations.
    u,s,vt=np.linalg.svd(a,full_matrices=False)
    weights=vt.T@((s/(s*s+.01))[:,None]*(u.T@np.eye(2)[y]))
    return np.argmax(b@weights,axis=1).tolist()
def load(r,folder,initial=False):
    if r['reused']:
        return AdaptiveCreditRouting.loads((folder/('initial-0.zip' if initial else 'model-0.zip')).read_bytes())
    cls=CapacityACR if r['arm'].startswith('groups') else CapacityOwn
    return cls.loads((folder/('initial.zip' if initial else 'model.zip')).read_bytes())

def audit(run,complete=False):
    start=time.perf_counter();m=read(run/'manifest.json');source=read(run/'result.json') if (run/'result.json').exists() else None
    if m['seeds']!=list(SEEDS) or m['height_grid']!=[48,64,128] or m['arms']!=list(ARMS) or m['fresh_confirmation'] or m['p2']:raise ValueError('Protocol/seed scope mismatch')
    for p,h in m['source_lock']['files'].items():
        if sha((ROOT/p).read_bytes())!=h:raise ValueError('Locked source changed '+p)
    peer=ROOT.parent/'review-gemini-r8-607ffca'
    if subprocess.check_output(['git','-C',str(peer),'rev-parse','HEAD'],text=True).strip()!=m['peer_sha']:raise ValueError('Peer moved')
    if subprocess.check_output(['git','-C',str(peer),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Peer dirty')
    for p,h in m['source_lock']['peer'].items():
        if sha((peer/p).read_bytes())!=h:raise ValueError('Peer body changed')
    parent=ROOT.parent/'scarx-acr-20261112'/'runs'/m['parent_run']
    if sha((parent/'manifest.json').read_bytes())!=m['parent_manifest_sha256'] or sha((parent/'audit-v1.json').read_bytes())!=m['parent_audit_sha256']:raise ValueError('Parent provenance mismatch')
    reuse=read(run/'reused-h48-files.json');folders=set()
    if len(reuse)!=192:raise ValueError('48reused cells ×4files expected')
    for p,receipt in reuse.items():
        path=Path(p);source_path=Path(receipt['source_path']).resolve()
        if path.is_absolute() or '..' in path.parts or not source_path.is_relative_to(parent/'jobs'):raise ValueError('Unsafe reused path')
        if sha((run/path).read_bytes())!=receipt['sha256'] or sha(source_path.read_bytes())!=receipt['sha256']:raise ValueError('Reused evidence changed')
        folders.add(path.parent)
    if len(folders)!=48:raise ValueError('Reuse folder count mismatch')
    data={};oracle_count=0
    for name,h in read(run/'dataset-hashes.json').items():
        file=run/'datasets'/name
        if sha(file.read_bytes())!=h or file.read_bytes()!=(parent/'datasets'/name).read_bytes():raise ValueError('Frozen data changed')
        with np.load(file,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        task=name.rsplit('-',1)[0]
        for split in ('train','refit','id','ood'):
            y=d[split+'_y'].tolist()
            if independent_labels(task,d[split+'_x'])!=y:raise ValueError('Independent threshold oracle mismatch')
            oracle_count+=len(y)
        sets=[{sha(x.tobytes()) for x in d[s+'_x']} for s in ('train','refit','id','ood')]
        if any(x&y for i,x in enumerate(sets) for y in sets[i+1:]):raise ValueError('Split overlap')
        data[name[:-4]]=d
    expected={(h,a,t,s,'equal_updates') for h,a,t,s in itertools.product((48,64,128),ARMS,TASKS,SEEDS)}
    expected|={(h,a,t,s,'equal_cpu') for h,a,t,s in itertools.product((48,64,128),ARMS[:3],TASKS,SEEDS)}
    rows={};initials={};scores=manual=probe=loads=histories=init_pairs=router_rng_checks=0
    files=sorted((run/'reused-h48').glob('*/capacity-result.json'))+sorted((run/'jobs').glob('*/result.json'))
    for path in files:
        r=read(path);key=tuple(r[k] for k in ('height','arm','task','seed','budget'))
        if key in rows or key not in expected:raise ValueError('Duplicate/unregistered result')
        rows[key]=r;d=data[f"{r['task']}-{r['seed']}"];folder=run/r['folder']
        if folder.resolve()!=path.parent.resolve():raise ValueError('Result path differs from containing folder')
        for split in ('id','ood'):
            p=r['predictions'][split];y=d[split+'_y'].tolist()
            if p['target']!=y:raise ValueError('Stored target changed')
            for metric in ('direct','probe'):
                if scalar_accuracy(p[metric],y)!=r['metrics'][split+'_'+metric]:raise ValueError('Scalar metric mismatch')
                scores+=len(y)
        for name,p in r['interventions'].items():
            if name not in ('revert_in','revert_rec','revert_both') or p['target']!=d['id_y'].tolist():raise ValueError('Intervention schema')
            for metric in ('direct','probe'):
                if scalar_accuracy(p[metric],p['target'])!=r['metrics'][name+'_'+metric]:raise ValueError('Intervention metric mismatch')
                scores+=len(p['target'])
        model_file=folder/('model-0.zip' if r['reused'] else 'model.zip');blob=model_file.read_bytes()
        if sha(blob)!=r['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        model=load(r,folder);base=load(r,folder,True);loads+=1
        if model.dumps()!=blob:raise ValueError('Checkpoint roundtrip mismatch')
        if model.config.hidden!=r['height'] or (model.config.inputs,model.config.fan_in,model.config.outputs)!=(8,8,2) or model.config.neuron!='alif':raise ValueError('Topology mismatch')
        expected_condition='readout_only' if r['arm']=='readout_only' else 'shuffled_signal' if r['arm']=='groups_shuffle' else 'learned_mtsg'
        if model.condition!=expected_condition or base.condition!=expected_condition:raise ValueError('Arm condition mismatch')
        if r['arm'].startswith('groups'):
            if model.profile!='groups_up' or model.router_rng.bit_generator.state!=base.router_rng.bit_generator.state:raise ValueError('Fixed groups router policy/RNG changed')
            router_rng_checks+=1
        if r['reused'] and model.profile!={'own':'static_0','groups_up':'groups_up','readout_only':'readout_only','groups_shuffle':'groups_up'}[r['arm']]:raise ValueError('Reused profile differs from mapped arm')
        if r['reused']:
            original=read(folder/'result-0.json')
            for k in ('metrics','predictions','interventions','training','checkpoint_sha256'):
                if r[k]!=original[k]:raise ValueError('Reuse transformed quality evidence')
            if sha((folder/'result-0.json').read_bytes())!=r['source_result_sha256']:raise ValueError('Reuse source result changed')
            if model.signal_rng.bit_generator.state!=original['signal_rng_state']:raise ValueError('Reused RNG receipt mismatch')
        else:
            if sha((folder/'initial.zip').read_bytes())!=r['initial_sha256']:raise ValueError('Initial hash mismatch')
            if model.signal_rng.bit_generator.state!=r['signal_rng_state']:raise ValueError('RNG receipt mismatch')
            if r['wall_seconds']>120:raise ValueError('Per-cell wall limit exceeded')
        expected_bytes=(944 if r['arm'].startswith('groups') else 596)*r['height']+864
        if r['array_bytes']!=expected_bytes or (not r['reused'] and model.memory_bytes()!=expected_bytes):
            raise ValueError('Capacity array accounting mismatch')
        prior=initials.setdefault((r['height'],r['task'],r['seed']),base)
        if prior is not base:
            for k,v in prior.core.arrays().items():np.testing.assert_array_equal(v,base.core.arrays()[k])
            for k in ('f_mean','f_var','syn_trace','syn_energy'):np.testing.assert_array_equal(getattr(prior,k),getattr(base,k))
            assert prior.signal_rng.bit_generator.state==base.signal_rng.bit_generator.state
            init_pairs+=1
        with np.load(folder/('learning-0.npz' if r['reused'] else 'learning.npz'),allow_pickle=False) as z:history=z['history']
        n=r['training']['episodes'];expected_columns=4 if r['reused'] else 5
        if history.shape!=(n,expected_columns) or not np.isfinite(history).all() or history[:,0].tolist()!=list(range(1,n+1)):raise ValueError('History schema/count mismatch')
        if r['budget']=='equal_updates' and n!=1024:raise ValueError('Primary update budget changed')
        if r['budget']=='equal_cpu' and (n>4096 or history[-1,-1]<.5 and n!=4096):raise ValueError('CPU stopping rule changed')
        if history[-1,-1]!=r['training']['train_cpu_seconds']:raise ValueError('CPU receipt mismatch')
        if model.episodes_learned!=n or model.core.updates!=n or model.total_ticks!=12*n:raise ValueError('Episode/update/tick counter mismatch')
        if model.hidden_updates!=(0 if r['arm']=='readout_only' else n):raise ValueError('Hidden counter mismatch')
        histories+=n
        if r['arm']=='readout_only':
            if np.any(history[:,2]) or not np.array_equal(model.core.w_in,base.core.w_in) or not np.array_equal(model.core.w_rec,base.core.w_rec):raise ValueError('Readout-only hidden moved')
        for split in ('id','ood'):
            f=literal_features(model,d[split+'_x'][:8])
            if literal_predictions(model,f)!=r['predictions'][split]['direct'][:8]:raise ValueError('Manual ALIF replay mismatch')
            manual+=8
        # Independent SVD probe on first seed for each size/arm; readout and
        # refit standardization remain fixed to refit data, never ID/OOD labels.
        if r['budget']=='equal_updates' and r['seed']==SEEDS[0]:
            fit=literal_features(model,d['refit_x'])
            for split in ('id','ood'):
                f=literal_features(model,d[split+'_x'][:8])
                if svd_probe(fit,d['refit_y'],f)!=r['predictions'][split]['probe'][:8]:raise ValueError('Independent SVD probe mismatch')
                probe+=8
    missing=sorted(expected-set(rows))
    if complete and (missing or not source or source['error'] is not None or source['new_jobs']!=204 or source['row_count']!=252 or source['reused_h48_cells']!=48):raise ValueError('Complete audit requested on incomplete grid')
    frozen_recoveries=0
    for key,r in rows.items():
        h,a,t,s,b=key
        if b!='equal_updates' or (h,'readout_only',t,s,b) not in rows:continue
        ro=rows[(h,'readout_only',t,s,b)]
        if r['interventions']['revert_both']['probe']!=ro['predictions']['id']['probe']:raise ValueError('Full revert does not recover frozen representation')
        frozen_recoveries+=1
    out={'integrity':'PASS' if not missing else 'PARTIAL_VALID_SO_FAR','records':len(rows),'missing_records':len(missing),
       'reused_files_verified':len(reuse),'independent_oracle_labels':oracle_count,'prediction_entries_rescored':scores,
       'checkpoint_roundtrips':loads,'manual_ALIF_predictions':manual,'independent_SVD_probe_predictions':probe,
       'initial_state_pairs':init_pairs,'history_update_calls':histories,'full_revert_frozen_recoveries':frozen_recoveries,
       'fixed_router_rng_checks':router_rng_checks,'arm_policy_checks':len(rows),
       'new_learning_updates_in_audit':0,'source_commit':m['source_commit'],'audit_wall_seconds':time.perf_counter()-start}
    if not missing:
        def row(h,a,t,s,b='equal_updates'):return rows[(h,a,t,s,b)]
        means={str(h):{a:{t:{k:statistics.mean(row(h,a,t,s)['metrics'][k] for s in SEEDS) for k in METRICS} for t in TASKS} for a in ARMS} for h in (48,64,128)}
        checks={};contrasts={}
        for h in (64,128):
            checks[str(h)]={};contrasts[str(h)]={}
            for a in ('own','groups_up'):
                families={};direct_good=hidden_good=0
                contrasts[str(h)][a]={}
                for t in TASKS[:3]:
                    x=means[str(h)][a][t];base=means['48'][a][t]
                    dg=x['id_direct']-base['id_direct']
                    hg=(x['id_probe']-means[str(h)]['readout_only'][t]['id_probe'])-(base['id_probe']-means['48']['readout_only'][t]['id_probe'])
                    causal=statistics.mean(row(h,a,t,s)['metrics']['id_direct']-row(h,a,t,s)['metrics']['revert_both_direct'] for s in SEEDS)
                    if dg>=.02:direct_good+=1
                    if hg>=.02:hidden_good+=1
                    families[t]={'direct_gain':dg,'hidden_gain_improvement':hg,'causal_direct_drop':causal,
                         'id_ood_preserved':all(x[k]-base[k]>=-.03 for k in METRICS),
                         'causal_for_improving_family':dg<.02 or causal>=.03}
                    contrasts[str(h)][a][t]={k:stat([row(h,a,t,s)['metrics'][k]-row(48,a,t,s)['metrics'][k] for s in SEEDS]) for k in METRICS}
                order=means[str(h)][a]['order']['id_direct']
                protected=order>=.65 and order-means['48'][a]['order']['id_direct']>=-.03
                signal=direct_good>=2 and hidden_good>=2 and protected and all(f['id_ood_preserved'] and f['causal_for_improving_family'] for f in families.values())
                checks[str(h)][a]={'families':families,'direct_improved_families':direct_good,'hidden_gain_improved_families':hidden_good,
                    'order_protected':protected,'capacity_dev_signal':signal}
        current_main=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()
        resource=source['resources']
        if source['main_unchanged'] is not True or current_main!=m['main_before']:raise ValueError('Main changed')
        if resource['measurement_errors'] or resource['observed_over_limit'] or max(resource['peak_rss_bytes'],resource['peak_commit_or_vms_bytes'],resource['os_process_peak_working_set_bytes'],resource['os_process_peak_pagefile_commit_bytes'])>2_000_000_000:raise ValueError('Resource contract violated')
        if source['wall_seconds']>3600 or source['fresh_confirmation'] or source['p2'] or source['production_ready']:raise ValueError('Time/scope contract violated')
        out.update(means=means,contrasts=contrasts,capacity_checks=checks,main_unchanged=True,resource_contract_verified=True,
            capacity_signal_pairs=[f'H{h}/{a}' for h,v in checks.items() for a,c in v.items() if c['capacity_dev_signal']],
            physical_ram_peak=resource['peak_rss_bytes'],committed_peak=resource['peak_commit_or_vms_bytes'])
        full_contrasts={};shuffle={}
        for h in (64,128):
            full_contrasts[str(h)]={}
            for arm in ('own','groups_up','readout_only'):
                full_contrasts[str(h)][arm]={}
                for task in TASKS:
                    values={k:stat([row(h,arm,task,s)['metrics'][k]-row(48,arm,task,s)['metrics'][k] for s in SEEDS]) for k in METRICS}
                    values['hidden_advantage_growth']=stat([
                      (row(h,arm,task,s)['metrics']['id_probe']-row(h,'readout_only',task,s)['metrics']['id_probe'])-
                      (row(48,arm,task,s)['metrics']['id_probe']-row(48,'readout_only',task,s)['metrics']['id_probe']) for s in SEEDS])
                    full_contrasts[str(h)][arm][task]=values
        for h in (48,64,128):
            shuffle[str(h)]={task:{k:stat([row(h,'groups_up',task,s)['metrics'][k]-row(h,'groups_shuffle',task,s)['metrics'][k] for s in SEEDS]) for k in METRICS} for task in TASKS}
        neural=bool(out['capacity_signal_pairs']);frozen=False
        for h in (64,128):
            count=sum(means[str(h)]['readout_only'][t]['id_direct']-means['48']['readout_only'][t]['id_direct']>=.02 for t in TASKS[:3])
            preserved=all(means[str(h)]['readout_only'][t][k]-means['48']['readout_only'][t][k]>=-.03 for t in TASKS[:3] for k in METRICS)
            order=means[str(h)]['readout_only']['order']['id_direct']
            frozen |= count>=2 and preserved and order>=.65 and order-means['48']['readout_only']['order']['id_direct']>=-.03
        raw=any(c['direct_improved_families']>0 for v in checks.values() for c in v.values())
        out.update(full_size_contrasts=full_contrasts,groups_vs_signal_shuffle=shuffle,
          decision='CAPACITY_DEV_SIGNAL' if neural else 'CAPACITY_ONLY_SIGNAL' if frozen else 'MIXED' if raw else 'NO_CLEAR_CAPACITY_GAIN')
        if (run/'audit.json').exists():
            parent_audit=read(run/'audit.json');compared=0
            if parent_audit['means']!=means or parent_audit['decision']!=out['decision']:raise ValueError('Parent means/decision differs from independent reconstruction')
            for own_set,parent_set in ((full_contrasts,parent_audit['contrasts']),(shuffle,parent_audit['groups_vs_signal_shuffle'])):
                # Both maps have height/task or height/arm/task nesting. Flatten
                # only metric dictionaries with a seed-effects field.
                def compare(a,b):
                    nonlocal compared
                    if 'effects' in a:
                        for k in ('effects','mean','lower95','upper95'):
                            if a[k]!=b[k]:raise ValueError('Parent contrast differs: '+k)
                        if a['exact_two_sided_p']!=b['exact_p']:raise ValueError('Parent sign-test differs')
                        compared+=1;return
                    if set(a)!=set(b):raise ValueError('Parent contrast schema differs')
                    for k in a:compare(a[k],b[k])
                compare(own_set,parent_set)
            out['parent_contrasts_crosschecked']=compared
        out['audit_wall_seconds']=time.perf_counter()-start
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--require-complete',action='store_true');p.add_argument('--output');a=p.parse_args()
    run=(ROOT/a.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned run required')
    out=audit(run,a.require_complete)
    if a.output:
        dest=(ROOT/a.output).resolve()
        if not dest.is_relative_to(ROOT):raise ValueError('Owned output required')
        with dest.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k not in ('means','contrasts','capacity_checks','full_size_contrasts','groups_vs_signal_shuffle')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
