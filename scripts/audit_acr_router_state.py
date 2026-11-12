"""Zero-update state-router diagnostics on existing completed DEV checkpoints.

Original snapshots stay byte-identical. Only clone neural/trace/router-RNG
state advances; no learn_episode calls, labels, optimizer or stats updates.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,sys,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.acr import AdaptiveCreditRouting as ACR
from scarx.evaluation.confirmation_resources import ConfirmationResources
from scarx.runtime.limits import apply_memory_limit

PROFILES=('state','state_shuffle','random_alpha','state_signal_shuffle')
TASKS=('strict_binding','xor','distractor_binding','order')
SEEDS=(140011,140017,140023)
INDICES=(0,63,511,1023)
DOC='docs/research/ACR_ROUTER_ANALYSIS_20261112.md'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write_once(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)
def sigmoid(x):return (1/(1+np.exp(-np.clip(x,-30,30)))).astype(np.float32)
def correlation(a,b):
    x=np.asarray(a,np.float64).ravel();y=np.asarray(b,np.float64).ravel()
    return float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None
def cosine(a,b):
    x=np.asarray(a,np.float64).ravel();y=np.asarray(b,np.float64).ravel();norm=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/norm) if norm else None

def learning_hash(model):
    h=hashlib.sha256()
    for name in model.core.weight_names:
        value=getattr(model.core,name);h.update(name.encode()+value.dtype.str.encode()+repr(value.shape).encode()+value.tobytes())
    for name in ('syn_trace','syn_energy','f_mean','f_var','w_in_0','w_rec_0','gamma_e_vec'):
        value=getattr(model,name);h.update(name.encode()+value.dtype.str.encode()+repr(value.shape).encode()+value.tobytes())
    settings={'condition':model.condition,'profile':model.profile,'episodes_learned':model.episodes_learned,
      'hidden_updates':model.hidden_updates,'core_updates':model.core.updates,'core_reward_mean':model.core.reward_mean,
      'calibrated':model.calibrated,'signal_rng':copy.deepcopy(model.signal_rng.bit_generator.state)}
    h.update(json.dumps(settings,sort_keys=True,allow_nan=False).encode());return h.hexdigest()

def check_router_interface():
    tree=ast.parse((ROOT/'src/scarx/evaluation/acr.py').read_text(encoding='utf-8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='AdaptiveCreditRouting')
    route=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='route')
    if [a.arg for a in route.args.args]!=['self','u','theta','z','a']:raise ValueError('Router interface changed')
    forbidden={'target','label','task','accuracy','loss','future','test'}
    if forbidden&{x.id for x in ast.walk(route) if isinstance(x,ast.Name)}:raise ValueError('Forbidden router input/state path')

def analyze(run):
    start=time.perf_counter();deadline=start+600;cap=apply_memory_limit()
    if run.parent!=ROOT/'runs' or (run/'router-analysis-v1.json').exists():raise ValueError('Unique owned run analysis required')
    # Require the full C grid first, even though joint checkpoints are not
    # used in descriptive correlations below. No cherry-picking early C cells.
    cells={}
    for path in (run/'jobs').glob('*/result-0.json'):
        row=read(path)
        if row['profile'] in PROFILES and row['budget']=='equal_updates':
            key=(row['profile'],row['task'],row['seed'])
            if key in cells:raise ValueError('Duplicate C cell')
            cells[key]=(row,path)
    expected={(p,t,s) for p in PROFILES for t in TASKS+('joint',) for s in SEEDS}
    if set(cells)!=expected:raise ValueError('Wait for all60registered C checkpoints before analysis')
    for row,path in cells.values():
        want='shuffled_signal' if row['profile']=='state_signal_shuffle' else 'learned_mtsg'
        if row['condition']!=want or row['training']['episodes']!=1024:raise ValueError('Unmatched C policy/budget')
    manifest=read(run/'manifest.json');source_lock=manifest['source_sha256']
    for name,h in source_lock.items():
        if sha((ROOT/name).read_bytes())!=h:raise ValueError('Frozen source changed: '+name)
    check_router_interface()
    rawdir=run/'router-analysis-v1-raw';rawdir.mkdir(exist_ok=False)
    registration={'scope':'Supplemental zero-update diagnostics, no candidate selection',
      'profiles':PROFILES,'tasks':TASKS,'seeds':SEEDS,'train_indices':INDICES,'rollouts':192,
      'guard_completed_C_cells':60,'source_commit':manifest['source_commit'],
      'parent_manifest_sha256':sha((run/'manifest.json').read_bytes()),
      'script_sha256':sha(Path(__file__).read_bytes()),'protocol_doc_sha256':sha((ROOT/DOC).read_bytes()),
      'protocol_document_before_compute':(ROOT/DOC).read_text(encoding='utf-8'),
      'max_seconds':600,'learning_updates_allowed':0,'source_sha256':source_lock,
      'input_labels_read':False,'main_or_Hermes_change':False}
    write_once(rawdir/'registration.json',registration)
    monitor=ConfirmationResources(interval=.05);rows=[];receipts={};error=None;fields=('alpha','spike','adaptation','q','distance','own_stop_cosine','own_stop_cosine_defined')
    def check():
        monitor.check()
        if time.perf_counter()>deadline:raise TimeoutError('Router diagnostic600second limit')
    with monitor:
      try:
        bank={};data_hashes=read(run/'dataset-hashes.json')
        for task in TASKS:
          for seed in SEEDS:
            path=run/'datasets'/f'{task}-{seed}.npz'
            if sha(path.read_bytes())!=data_hashes[path.name]:raise ValueError('Frozen TRAIN data changed')
            with np.load(path,allow_pickle=False) as z:bank[(task,seed)]=z['train_x'].copy()
            check()
        for profile in PROFILES:
          for task in TASKS:
            for seed in SEEDS:
                cell,path=cells[(profile,task,seed)];blob=(path.parent/'model-0.zip').read_bytes()
                if sha(blob)!=cell['checkpoint_sha256']:raise ValueError('Checkpoint hash changed')
                original=ACR.loads(blob);original_blob=original.dumps()
                if original_blob!=blob:raise ValueError('Checkpoint roundtrip mismatch')
                fixed=learning_hash(original);tapes={}
                for index in INDICES:
                    m=ACR.loads(blob);m.reset_episode();fixed_clone=learning_hash(m);ticks=m.total_ticks
                    records={name:[] for name in fields};same_state_random_distribution=True;state_shuffle_distinct=0;alpha_order_changes=0
                    for input_row in bank[(task,seed)][index]:
                        previous_q=m.router_q.copy();router_rng=copy.deepcopy(m.router_rng)
                        m.step(input_row,track=True)
                        own=np.linalg.norm(m.e_f_own,axis=1);stop=np.linalg.norm(m.e_f_stop,axis=1)
                        q=.9*previous_q+.1*(stop-own)/(stop+own+1e-6)
                        np.testing.assert_array_equal(m.router_q,q)
                        distance=np.clip(np.abs(m.last_u-m.last_theta)/(m.last_theta+1e-6),0,4)
                        f=np.column_stack((distance,m.core.spikes,m.last_a/(1+m.last_a),q))
                        perm=router_rng.permutation(m.config.hidden);coefficient=np.asarray([-.5,1.5,1,1],np.float32)
                        same_state=sigmoid(m.router_bias-1+f@coefficient)
                        state_shuffled=sigmoid(m.router_bias-1+f[perm]@coefficient)
                        random=same_state[perm]
                        np.testing.assert_array_equal(np.sort(random),np.sort(same_state))
                        state_shuffle_distinct+=int(not np.array_equal(state_shuffled,random))
                        alpha_order_changes+=int(not np.array_equal(random,same_state))
                        expected_alpha=state_shuffled if profile=='state_shuffle' else random if profile=='random_alpha' else same_state
                        np.testing.assert_array_equal(m.alpha,expected_alpha)
                        if np.any((m.alpha<0)|(m.alpha>1)):raise ValueError('Alpha range')
                        np.testing.assert_array_equal(m.e_f,(1-m.alpha[:,None])*m.e_f_own+m.alpha[:,None]*m.e_f_stop)
                        if learning_hash(m)!=fixed_clone:raise ValueError('Clone forward modified learning state')
                        trace_cosine=cosine(m.e_f_own,m.e_f_stop)
                        for name,value in {'alpha':m.alpha,'spike':m.core.spikes,'adaptation':m.last_a,'q':q,'distance':distance,
                          'own_stop_cosine':np.asarray(0. if trace_cosine is None else trace_cosine),
                          'own_stop_cosine_defined':np.asarray(trace_cosine is not None)}.items():records[name].append(np.array(value,copy=True))
                    if m.total_ticks!=ticks+len(bank[(task,seed)][index]):raise ValueError('Tick count mismatch')
                    records={name:np.asarray(value) for name,value in records.items()};alpha=records['alpha']
                    rows.append({'profile':profile,'task':task,'seed':seed,'train_index':index,
                      'alpha_mean':float(alpha.mean()),'alpha_min':float(alpha.min()),'alpha_max':float(alpha.max()),
                      'alpha_near_endpoint_fraction':float(np.mean((alpha<.05)|(alpha>.95))),
                      'mean_within_neuron_temporal_std':float(alpha.std(axis=0).mean()),
                      'mean_within_tick_neuron_std':float(alpha.std(axis=1).mean()),
                      'alpha_spike_correlation':correlation(alpha,records['spike']),
                      'alpha_adaptation_correlation':correlation(alpha,records['adaptation']),
                      'alpha_q_correlation':correlation(alpha,records['q']),
                      'alpha_distance_correlation':correlation(alpha,records['distance']),
                      'mean_trace_cosine':float(records['own_stop_cosine'][records['own_stop_cosine_defined']].mean()) if records['own_stop_cosine_defined'].any() else None,
                      'alpha_per_tick_mean':alpha.mean(axis=1).tolist(),'alpha_per_neuron_mean':alpha.mean(axis=0).tolist(),
                      'same_state_random_distribution_exact':same_state_random_distribution,
                      'same_state_shuffle_distinct_ticks':state_shuffle_distinct,'same_state_random_reordered_ticks':alpha_order_changes,
                      'ticks':len(alpha),'learning_updates':0})
                    for name,array in records.items():tapes.setdefault(name,[]).append(array)
                    check()
                if learning_hash(original)!=fixed or original.dumps()!=original_blob:raise ValueError('Original model changed')
                outfile=rawdir/f'{profile}-{task}-{seed}.npz'
                np.savez_compressed(outfile,train_indices=np.asarray(INDICES,np.int32),**{name:np.asarray(values) for name,values in tapes.items()})
                receipts[outfile.name]={'sha256':sha(outfile.read_bytes()),'checkpoint_sha256':sha(blob),'original_unchanged':True}
                check()
        if len(rows)!=192 or len(receipts)!=48:raise ValueError('Router analysis incomplete')
        for name,h in source_lock.items():
            if sha((ROOT/name).read_bytes())!=h:raise ValueError('Source changed during analysis')
      except Exception as exc:error=type(exc).__name__+': '+str(exc)
    summary={}
    for profile in PROFILES:
        summary[profile]={}
        for task in TASKS:
            group=[r for r in rows if r['profile']==profile and r['task']==task]
            if len(group)!=12:continue
            keys=('alpha_mean','alpha_near_endpoint_fraction','mean_within_neuron_temporal_std','mean_within_tick_neuron_std',
              'alpha_spike_correlation','alpha_adaptation_correlation','alpha_q_correlation','alpha_distance_correlation','mean_trace_cosine')
            summary[profile][task]={key:float(np.mean([r[key] for r in group if r[key] is not None])) if any(r[key] is not None for r in group) else None for key in keys}
            summary[profile][task]['alpha_per_tick_mean']=np.mean([r['alpha_per_tick_mean'] for r in group],axis=0).tolist()
            summary[profile][task]['alpha_per_neuron_mean']=np.mean([r['alpha_per_neuron_mean'] for r in group],axis=0).tolist()
    result={'status':'PASS' if error is None else 'STOPPED_PRESERVE_PARTIAL','error':error,'rollouts':len(rows),
       'checkpoint_files':len(receipts),'rows':rows,'summary':summary,'raw_receipts':receipts,'registration':registration,
       'new_learning_updates':0,'labels_read':False,'candidate_selection_performed':False,'fresh_confirmation':False,
       'cap':cap,'resources':monitor.report(),'wall_seconds':time.perf_counter()-start,
       'limitations':'Fixed heuristic router, final-checkpoint replay not original training trajectory. Alpha before endpoint does not directly change weights. Correlations are descriptive; models differ after training.'}
    write_once(run/'router-analysis-v1.json',result)
    print(json.dumps({k:result[k] for k in ('status','error','rollouts','checkpoint_files','new_learning_updates','wall_seconds')}))
    return int(error is not None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();return analyze((ROOT/args.run).resolve())
if __name__=='__main__':raise SystemExit(main())
