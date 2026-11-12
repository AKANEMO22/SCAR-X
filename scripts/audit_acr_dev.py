"""Read-only raw validation and independent scalar summaries. ZERO training.

Audit output is a new file: old model/prediction/data evidence is never rewritten.
This script was prepared during DEV; it cannot change the locked learner/protocol.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,math,os,statistics,subprocess,sys,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.acr import AdaptiveCreditRouting as ACR,AB_PROFILES,C_PROFILES
from scarx.evaluation.r8_matched_study import SEEDS,TASKS

ENVS=TASKS+('joint',);KEYS=('id_direct','id_probe','ood_direct','ood_probe')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(raw):return hashlib.sha256(raw).hexdigest()
def acc(p,y):
    if len(p)!=len(y) or not p or any(type(v) is not int or v not in (0,1) for v in p+y):raise ValueError('Binary prediction schema')
    return sum(a==b for a,b in zip(p,y))/len(y)
def scalar_stats(effects):
    a=list(map(float,effects))
    if len(a)!=3 or any(not math.isfinite(x) for x in a):raise ValueError('3seed cluster required')
    boot=sorted(sum(t)/3 for t in itertools.product(a,repeat=3))
    mean=sum(a)/3
    null=[abs(sum(x*s for x,s in zip(a,ss))/3) for ss in itertools.product((-1,1),repeat=3)]
    return {'effects':a,'mean':mean,'lower95':boot[math.ceil(.025*27)-1],'upper95':boot[math.ceil(.975*27)-1],
        'exact_two_sided_p':sum(x>=abs(mean)-1e-12 for x in null)/8,'scope':'Exploratory3seed; no significance or confirmation claim'}

def independent_probe(fit,y,test):
    # Independent ridge implementation using a symmetric eigensystem rather than
    # candidate solve(); only refit data determine centering/scale/coefficients.
    mean=np.mean(fit,axis=0);scale=np.std(fit,axis=0)+1e-4
    a=np.column_stack(((fit-mean)/scale,np.ones(len(fit))))
    b=np.column_stack(((test-mean)/scale,np.ones(len(test))))
    gram=a.T@a+.01*np.eye(a.shape[1]);values,vectors=np.linalg.eigh(gram)
    rhs=np.zeros((a.shape[1],2),float)
    for i,target in enumerate(y):rhs[:,int(target)]+=a[i]
    w=vectors@((vectors.T@rhs)/values[:,None])
    return np.argmax(b@w,axis=1).astype(int).tolist()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',default='audit-v1.json');args=ap.parse_args()
    run=(ROOT/args.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned run only')
    destination=run/args.output
    if destination.exists():raise ValueError('Never overwrite audit evidence')
    start=time.perf_counter();manifest=read(run/'manifest.json');result=read(run/'result.json')
    if result['error'] is not None:raise ValueError('Incomplete DEV cannot receive complete audit')
    main_now=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()
    resource=result['resources']
    if result.get('main_unchanged') is not True or main_now!=manifest['main_before']:
        raise ValueError('Main ref changed during isolated research')
    if resource.get('measurement_errors') or resource.get('observed_over_limit') or max(resource['peak_rss_bytes'],resource['peak_commit_or_vms_bytes'],resource['os_process_peak_working_set_bytes'],resource['os_process_peak_pagefile_commit_bytes'])>2_000_000_000:
        raise ValueError('Resource contract failed or unmeasured')
    if result.get('new_confirmation') is not False or result.get('production_ready') is not False:
        raise ValueError('Research-only scope metadata mismatch')
    reused_count=0;reuse_chain=[]
    def verify_reuse(current,meta,depth=0):
        if depth>8:raise ValueError('Resume provenance depth exceeded')
        if 'parent_run' not in meta:return
        parent=(ROOT/'runs'/meta['parent_run']).resolve()
        if parent.parent!=ROOT/'runs' or parent==current or sha((parent/'manifest.json').read_bytes())!=meta['parent_manifest_sha256']:
            raise ValueError('Resume parent manifest mismatch')
        reused=meta['reused_files_sha256'];count=meta['reused_completed_cells']
        if type(count)is not int or count not in (55,203) or len(reused)!=4*count:raise ValueError('Reused evidence grid mismatch')
        folders={}
        for relative,h in reused.items():
            name=Path(relative)
            if name.is_absolute() or '..' in name.parts or len(name.parts)!=3 or name.parts[0]!='jobs':raise ValueError('Unsafe reused evidence path')
            if sha((parent/name).read_bytes())!=h or sha((current/name).read_bytes())!=h:raise ValueError('Reused evidence differs from audited parent bytes')
            folders.setdefault(name.parts[1],set()).add(name.name)
        if len(folders)!=count or any(names!={'initial-0.zip','learning-0.npz','model-0.zip','result-0.json'} for names in folders.values()):
            raise ValueError('Reused cell files incomplete')
        if 'replayed_incomplete_cell' in meta:
            original=parent/'jobs'/meta['replayed_incomplete_cell']/'initial-0.zip'
            repeated=current/'jobs'/meta['replayed_incomplete_cell']/'initial-0.zip'
            if sha(original.read_bytes())!=meta['incomplete_initial_sha256'] or sha(repeated.read_bytes())!=meta['incomplete_initial_sha256']:
                raise ValueError('Repeated interrupted-cell initial state changed')
            if (meta['uncheckpointed_attempt_updates_known_min'],meta['uncheckpointed_attempt_updates_reserved_max'])!=(128,1024):raise ValueError('Lost-attempt accounting changed')
        reuse_chain.append({'run':current.name,'parent':parent.name,'cells':count})
        verify_reuse(parent,read(parent/'manifest.json'),depth+1)
    verify_reuse(run,manifest)
    reused_count=manifest.get('reused_completed_cells',0)
    for path,h in manifest['source_sha256'].items():
        if sha((ROOT/path).read_bytes())!=h:raise ValueError('Locked source changed: '+path)
    from scarx.evaluation.r8_matched_credit import PEER
    for path,h in manifest['peer_sha256'].items():
        if sha((PEER/path).read_bytes())!=h:raise ValueError('Locked peer changed')
    if manifest['seeds']!=list(SEEDS) or manifest['confirmation'] is not False:raise ValueError('Wrong seed scope')
    data_hashes=read(run/'dataset-hashes.json');data={}
    for path,h in data_hashes.items():
        raw=(run/'datasets'/path).read_bytes()
        if sha(raw)!=h:raise ValueError('Dataset changed')
        with np.load(run/'datasets'/path,allow_pickle=False) as archive:data[path[:-4]]={k:archive[k] for k in archive.files}
    rows=[];count=0;checkpoints=0;direct_replay=0;probe_replay=0;norm_steps=0;zero_norm_steps=0;max_norm_relative_error=0.;seen=set()
    nominee=read(run/'nominee-lock.json')['profile'];ab=read(run/'stage-ab.json')
    raw_profiles=AB_PROFILES+('readout_only',)+(C_PROFILES if ab['open_state'] else ())
    expected=set()
    def expect(task,seed,p,budget='equal_updates',shuffled=False,paired=False):
        label=p+('-signal_shuffle' if shuffled else '')
        folder=f'{task}-{seed}-{label}-{budget}'
        for i in range(2 if paired else 1):expected.add((folder,i))
    for p in raw_profiles:
        for task,seed in itertools.product(ENVS,SEEDS):expect(task,seed,p)
    if nominee!='state':
        for task,seed in itertools.product(ENVS,SEEDS):expect(task,seed,nominee,shuffled=True)
    for p in dict.fromkeys(('static_50',nominee)):
        for task,seed in itertools.product(ENVS,SEEDS):expect(task,seed,p,'norm_matched',paired=True)
    for p,sh in (('static_0',False),('static_100',False),(nominee,False),(nominee,True),('readout_only',False)):
        for task,seed in itertools.product(ENVS,SEEDS):expect(task,seed,p,'equal_cpu',shuffled=sh)

    invalid=list((run/'jobs').glob('*/invalid-control.json'))
    for path in sorted((run/'jobs').glob('*/result-*.json')):
        r=read(path);folder=path.parent;index=r['model_index'];key=(folder.name,index)
        if key in seen or key not in expected:raise ValueError('Unexpected/duplicate result')
        seen.add(key);rows.append(r)
        if r['task'] not in ENVS or r['seed'] not in SEEDS:raise ValueError('Unregistered task/seed')
        d=data[f"{r['task']}-{r['seed']}"]
        for split in ('id','ood'):
            stored=r['predictions'][split];y=d[split+'_y'].tolist()
            if stored['target']!=y:raise ValueError('Target differs from frozen bank')
            for metric in ('direct','probe'):
                if acc(stored[metric],y)!=r['metrics'][split+'_'+metric]:raise ValueError('Scalar metric mismatch')
                count+=len(y)
            if r['task']=='joint':
                for j,t in enumerate(TASKS):
                    idx=np.flatnonzero(d[split+'_families']==j).tolist()
                    for metric in ('direct','probe'):
                        value=acc([stored[metric][i] for i in idx],[y[i] for i in idx])
                        if value!=r['family_metrics'][t][split+'_'+metric]:raise ValueError('Joint family metric mismatch')
        for name,intervention in r['interventions'].items():
            y=d['id_y'].tolist()
            if intervention['target']!=y:raise ValueError('Revert target differs')
            for metric in ('direct','probe'):
                if acc(intervention[metric],y)!=r['metrics'][name+'_'+metric]:raise ValueError('Scalar intervention mismatch')
                count+=len(y)
        raw=(folder/f'model-{index}.zip').read_bytes()
        if sha(raw)!=r['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        initial=(folder/f'initial-{index}.zip').read_bytes()
        if sha(initial)!=r['initial_checkpoint_sha256']:raise ValueError('Initial checkpoint mismatch')
        m=ACR.loads(raw);base=ACR.loads(initial);checkpoints+=1
        if m.dumps()!=raw:raise ValueError('Checkpoint roundtrip mismatch')
        if m.signal_rng.bit_generator.state!=r['signal_rng_state'] or m.router_rng.bit_generator.state!=r['router_rng_state']:
            raise ValueError('RNG receipt mismatch')
        with np.load(folder/f'learning-{index}.npz',allow_pickle=False) as z:history=z['history']
        n=r['training']['episodes']
        if history.shape!=(n,4) or not np.isfinite(history).all() or history[:,0].tolist()!=list(range(1,n+1)):raise ValueError('History schema/count')
        if r['budget'] in ('equal_updates','norm_matched') and n!=1024:raise ValueError('Update count changed')
        if r['budget']=='equal_cpu' and (n>4096 or (history[-1,3]<.5 and n!=4096)):raise ValueError('CPU stopping rule changed')
        if not np.isclose(history[-1,3],r['training']['train_cpu_seconds'],rtol=0,atol=0):raise ValueError('CPU receipt mismatch')
        if abs(float(np.mean(history[:,2]))-r['training']['mean_update_norm'])>1e-15:raise ValueError('Mean physical norm mismatch')
        if r['extra_array_bytes']!=16704:raise ValueError('State overhead mismatch')
        if r['condition']=='readout_only':
            if m.hidden_updates or not np.array_equal(m.core.w_in,base.core.w_in) or not np.array_equal(m.core.w_rec,base.core.w_rec):raise ValueError('Frozen hidden moved')
            if np.any(history[:,2]):raise ValueError('Frozen norm nonzero')
        # Every cell: inherited hard-forward replay, without any learning.
        for split in ('id','ood'):
            prediction=m.predict_many(d[split+'_x'][:8]).tolist()
            if prediction!=r['predictions'][split]['direct'][:8]:raise ValueError('Direct replay mismatch')
            direct_replay+=8
        # All nominal/endpoints/readout fixed-update checkpoints: independent ridge.
        if r['budget']=='equal_updates' and r['profile'] in ('static_0','static_100','readout_only',nominee) and r['condition']!='shuffled_signal':
            fit=m.features_many(d['refit_x'])
            for split in ('id','ood'):
                prediction=independent_probe(fit,d['refit_y'],m.features_many(d[split+'_x'][:16]))
                if prediction!=r['predictions'][split]['probe'][:16]:raise ValueError('Independent ridge replay mismatch')
                probe_replay+=16
    missing=expected-seen
    # Only the two trajectories of an explicitly failed norm-pair may be absent.
    # An unrelated missing raw/equal-CPU cell cannot hide behind another failure.
    explained={(path.parent.name,i) for path in invalid for i in (0,1)}
    if missing!=explained:raise ValueError('Missing cells are not exactly explained by preserved invalid controls')
    expected_jobs=len({folder for folder,i in expected})
    if result['rows']!=len(rows) or result['completed_jobs']!=expected_jobs:
        raise ValueError('Worker result/cell/job counters differ')
    if result['wall_seconds']>manifest['limit_seconds']:
        raise ValueError('Registered remaining wall budget exceeded')
    for folder in sorted((run/'jobs').glob('*-norm_matched')):
        if (folder/'invalid-control.json').exists():continue
        with np.load(folder/'norm-pairs.npz',allow_pickle=False) as z:norms=z['radius_raw']
        with np.load(folder/'learning-0.npz',allow_pickle=False) as z:a=z['history'][:,2]
        with np.load(folder/'learning-1.npz',allow_pickle=False) as z:b=z['history'][:,2]
        if norms.shape!=(1024,3):raise ValueError('Norm-pair budget changed')
        if not np.allclose(norms[:,0],.5*np.minimum(norms[:,1],norms[:,2]),rtol=0,atol=0):raise ValueError('Wrong norm radius')
        for radius,aa,bb in zip(norms[:,0],a,b):
            if radius==0:
                if aa!=0 or bb!=0:raise ValueError('Zero radius changed weights')
                zero_norm_steps+=1
            else:
                err=max(abs(aa-radius),abs(bb-radius))/radius;max_norm_relative_error=max(max_norm_relative_error,float(err))
                if err>1e-5:raise ValueError('Applied norm mismatch')
            norm_steps+=1
    # Full-revert refit must recover the frozen representation on same data.
    for r in rows:
        if r['budget']!='equal_updates':continue
        ro=next(x for x in rows if x['task']==r['task'] and x['seed']==r['seed'] and x['profile']=='readout_only' and x['budget']=='equal_updates')
        if r['interventions']['revert_both']['probe']!=ro['predictions']['id']['probe']:
            raise ValueError('Revert-both failed to recover frozen representation')

    def normal(p,t,seed,condition=None,budget='equal_updates'):
        condition=condition or ('readout_only' if p=='readout_only' else 'shuffled_signal' if p=='state_signal_shuffle' else 'learned_mtsg')
        candidates=[r for r in rows if r['profile']==p and r['task']==t and r['seed']==seed and r['condition']==condition and r['budget']==budget]
        if len(candidates)!=1:raise ValueError('Ambiguous nominal cell')
        return candidates[0]
    means={p:{t:{k:statistics.mean(normal(p,t,s)['metrics'][k] for s in SEEDS) for k in KEYS} for t in ENVS} for p in raw_profiles}
    # Independently reconstruct selection, including alphabetical deterministic tie.
    def rank(p):
        diffs=[means[p][t][k]-max(means[e][t][k] for e in ('static_0','static_100')) for t in TASKS[:3] for k in ('id_direct','id_probe')]
        ds=[means[p][t]['id_direct']-max(means[e][t]['id_direct'] for e in ('static_0','static_100')) for t in TASKS[:3]]
        return min(diffs),statistics.mean(ds),p.startswith('static_')
    candidates=sorted(p for p in raw_profiles if p not in ('static_0','static_100','readout_only') and 'shuffle' not in p and p!='random_alpha')
    if max(candidates,key=rank)!=nominee:raise ValueError('Nomination differs from locked global rule')
    contrasts={};shuffle_pass=0;norm_pass=0;regression=False;raw_improved=[];probe_only=[]
    for t in TASKS[:3]:
        x=means[nominee][t];ref=means['readout_only'][t]
        contrasts[t]={}
        for label,p,metric in [('hidden','readout_only','id_probe'),('own_direct','static_0','id_direct'),('stop_direct','static_100','id_direct'),
                               ('own_probe','static_0','id_probe'),('stop_probe','static_100','id_probe'),
                               ('own_ood_direct','static_0','ood_direct'),('own_ood_probe','static_0','ood_probe')]:
            contrasts[t][label]=scalar_stats([normal(nominee,t,s)['metrics'][metric]-normal(p,t,s)['metrics'][metric] for s in SEEDS])
        shuffle='state_signal_shuffle' if nominee=='state' else nominee
        shuffled=[normal(shuffle,t,s,'shuffled_signal') for s in SEEDS]
        for k in ('id_direct','id_probe'):
            effects=[normal(nominee,t,s)['metrics'][k]-r['metrics'][k] for s,r in zip(SEEDS,shuffled)]
            contrasts[t]['shuffle_'+k]=scalar_stats(effects)
            if statistics.mean(effects)<-.03:regression=True
        if contrasts[t]['shuffle_id_probe']['mean']>=.02:shuffle_pass+=1
        pairs=[]
        for seed in SEEDS:
            cells=[r for r in rows if r['task']==t and r['seed']==seed and r['profile']==nominee and r['budget']=='norm_matched' and r['model_index']==0]
            if len(cells)!=1:continue
            left=cells[0];right=next(r for r in rows if r['path']==left['path'] and r['model_index']==1);pairs.append((left,right))
        if len(pairs)==3:
            for k in ('id_direct','id_probe'):
                effects=[a['metrics'][k]-b['metrics'][k] for a,b in pairs];contrasts[t]['norm_'+k]=scalar_stats(effects)
                if statistics.mean(effects)<-.03:regression=True
            if contrasts[t]['norm_id_probe']['mean']>=.01:norm_pass+=1
        gain_direct=x['id_direct']-max(means[e][t]['id_direct'] for e in ('static_0','static_100'))
        gain_probe=x['id_probe']-max(means[e][t]['id_probe'] for e in ('static_0','static_100'))
        if max(gain_direct,gain_probe)>=.02:raw_improved.append(t)
        if gain_probe>=.02 and gain_direct<.02:probe_only.append(t)
    valid=not invalid and not missing and result['norm_controls_valid']
    # Numeric broad_checks was locked before controls; reconstruct all direct,
    # representation, OOD, protected and joint predicates independently here.
    broad={};improved_with_cause=0
    for t in TASKS[:3]:
        x=means[nominee][t];own=means['static_0'][t];stop=means['static_100'][t]
        gain=x['id_probe']-means['readout_only'][t]['id_probe']
        causal=statistics.mean(normal(nominee,t,s)['metrics']['id_direct']-normal(nominee,t,s)['metrics']['revert_both_direct'] for s in SEEDS)
        improves=t in raw_improved
        if improves and gain>=.05 and causal>=.03:improved_with_cause+=1
        broad[t]={'direct65':x['id_direct']>=.65,
            'endpoint_preserved':all(x[k]-max(own[k],stop[k])>=-.03 for k in ('id_direct','id_probe')),
            'ood_preserved':all(x[k]-own[k]>=-.03 for k in ('ood_direct','ood_probe')),
            'hidden_probe_gain':gain,'causal_direct_drop':causal,'improved':improves}
    broad['order_preserved']=means[nominee]['order']['id_direct']>=.65 and means[nominee]['order']['id_direct']-max(means[e]['order']['id_direct'] for e in ('static_0','static_100'))>=-.03
    def joint_score(p,t):return statistics.mean(normal(p,'joint',s)['family_metrics'][t]['id_direct'] for s in SEEDS)
    broad['joint_preserved']={t:joint_score(nominee,t)-max(joint_score(e,t) for e in ('static_0','static_100'))>=-.03 for t in TASKS}
    broad['families_improved_with_causality']=improved_with_cause
    broad['broad_numeric']=improved_with_cause>=2 and broad['order_preserved'] and all(broad['joint_preserved'].values()) and all(broad[t][k] for t in TASKS[:3] for k in ('direct65','endpoint_preserved','ood_preserved'))
    if broad!=result['checks']:raise ValueError('Independent broad numeric checks disagree')
    # Required mechanism/OOD artifacts are distinct from quality metrics. They
    # must exist and pass their own zero-update analysis before full delivery.
    mechanism_file=run/'mechanism-audit-v1.json'
    if not mechanism_file.exists():raise ValueError('Required independent mechanism/stress analysis missing')
    mechanism=read(mechanism_file)
    if mechanism['status']!='PASS' or mechanism['diagnostic_files']!=144 or mechanism['stress_files']!=36 or mechanism['new_learning_updates']!=0:
        raise ValueError('Incomplete or invalid mechanism/stress artifacts')
    if len(mechanism['artifact_sha256'])!=180:raise ValueError('Mechanism hash grid incomplete')
    for relative,h in mechanism['artifact_sha256'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or len(path.parts)!=2 or path.parts[0]!='diagnostics':
            raise ValueError('Unsafe mechanism artifact path')
        if sha((run/path).read_bytes())!=h:raise ValueError('Mechanism/stress artifact changed after audit')
    router=read(run/'router-analysis-v1.json')
    if router['status']!='PASS' or router['error'] is not None or router['rollouts']!=192 or router['checkpoint_files']!=48 or router['new_learning_updates']!=0 or router['labels_read'] is not False:
        raise ValueError('State-router analysis incomplete')
    reg=router['registration']
    if sha(reg['protocol_document_before_compute'].encode('utf-8'))!=reg['protocol_doc_sha256'] or reg['parent_manifest_sha256']!=sha((run/'manifest.json').read_bytes()):
        raise ValueError('Router analysis registration/provenance mismatch')
    if sha((ROOT/'scripts/audit_acr_router_state.py').read_bytes())!=reg['script_sha256'] or len(router['raw_receipts'])!=48:
        raise ValueError('Router analysis source/raw grid mismatch')
    for filename,receipt in router['raw_receipts'].items():
        name=Path(filename)
        if len(name.parts)!=1 or name.suffix!='.npz' or receipt['original_unchanged'] is not True:
            raise ValueError('Router analysis file/original-state schema')
        if sha((run/'router-analysis-v1-raw'/name).read_bytes())!=receipt['sha256']:raise ValueError('Router trace changed')
    if not valid:decision='BLOCKED_INVALID_CONTROLS'
    elif broad['broad_numeric'] and shuffle_pass>=2 and norm_pass>=2 and not regression:decision='ACR_DEV_ADVANCE'
    elif len(probe_only)>=2:decision='ACR_REPRESENTATION_ONLY'
    elif len(raw_improved)==1:decision='ACR_SPECIALIZED_GAIN'
    else:decision='ACR_NO_GAIN'
    summary={'integrity':'PASS' if valid else 'INVALID_CONTROLS_PRESERVED','decision':decision,'nominee':nominee,
        'profile_means':means,'contrasts':contrasts,'broad_checks':broad,'shuffle_families_pass':shuffle_pass,
        'norm_families_pass':norm_pass,'control_regression':regression,'raw_improved_families':raw_improved,
        'probe_only_improved_families':probe_only,'models':len(rows),'expected_models':len(expected),'missing':sorted(missing),
        'invalid_cells':[str(x.relative_to(run)) for x in invalid],'prediction_rows_rescored':count,'checkpoints_roundtripped':checkpoints,
        'direct_replay_predictions':direct_replay,'independent_eigen_ridge_predictions':probe_replay,
        'norm_steps_checked':norm_steps,'zero_norm_steps':zero_norm_steps,'max_norm_relative_error':max_norm_relative_error,
        'new_training_updates_in_audit':0,'reused_cells_verified':reused_count,'fresh_confirmation':False,'production_ready':False,
        'reuse_chain':reuse_chain,
        'main_unchanged_verified':True,'resource_contract_verified':True,'mechanism_stress_verified':True,
        'state_router_analysis_verified':True,
        'decision_scope':'Registered broad/control-qualified gain; non-advancing raw tradeoffs remain reported explicitly',
        'audit_wall_seconds':time.perf_counter()-start,'source_commit':manifest['source_commit']}
    with destination.open('x',encoding='utf-8') as stream:json.dump(summary,stream,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps({k:summary[k] for k in ('integrity','decision','nominee','models','prediction_rows_rescored','norm_steps_checked','audit_wall_seconds')}),flush=True)

if __name__=='__main__':main()
