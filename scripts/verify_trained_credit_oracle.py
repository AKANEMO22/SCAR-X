"""Registered frozen-trained-gradient CONTROL: zero learning/selection.

180fixed TRAIN-episode rows, <=180seconds, <=2GB; never calls learn_episode.
The loss normalizer/readout are frozen, unlike the actual learner's updates.
"""
from pathlib import Path
import copy,hashlib,json,os,platform,subprocess,sys,time,uuid
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit
from scarx.evaluation.literature_eprop import LiteratureEPropALIF
from scarx.evaluation.literature_braintrace import BrainTraceDRTRL,full_surrogate_sensitivity,direction_metrics
from scarx.evaluation.confirmation_resources import ConfirmationResources
from scarx.runtime.limits import apply_memory_limit

PARENT='20261112T080533Z-literature-reference-dev-v1-839e07'
SOURCE='a3b79ea66cb6676f8b03a22b0cb907d158e5b2a2'
METHODS=('own','stop','eprop_matched','eprop_aligned','braintrace')
TASKS=('strict_binding','xor','distractor_binding');SEEDS=(140011,140017,140023)
INDICES=(0,63,511,1023)
PROTOCOL='experiments/trained-credit-oracle-v1.md'
SCRIPT='scripts/verify_trained_credit_oracle.py'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)

def learning_hash(model):
    """Ignore only deliberately advanced neural/eligibility state and tickcount."""
    h=hashlib.sha256()
    for name in model.core.weight_names:
        value=getattr(model.core,name);h.update(name.encode()+value.dtype.str.encode()+repr(value.shape).encode()+value.tobytes())
    for name in ('syn_trace','syn_energy','f_mean','f_var','w_in_0','w_rec_0','gamma_e_vec'):
        value=getattr(model,name);h.update(name.encode()+value.dtype.str.encode()+repr(value.shape).encode()+value.tobytes())
    control={'signal_rng':copy.deepcopy(model.signal_rng.bit_generator.state),
        'episodes_learned':model.episodes_learned,'hidden_updates':model.hidden_updates,
        'core_updates':model.core.updates,'core_reward_mean':model.core.reward_mean,
        'calibrated':model.calibrated,'condition':model.condition}
    h.update(json.dumps(control,sort_keys=True,allow_nan=False).encode())
    return h.hexdigest()

def model_class(method):
    return MatchedLocalCredit if method=='own' else StoppedResetCredit if method=='stop' else BrainTraceDRTRL if method=='braintrace' else LiteratureEPropALIF

def summarize(rows):
    out={}
    for method in METHODS:
        out[method]={}
        for task in TASKS+('all',):
            group=[r for r in rows if r['method']==method and (task=='all' or r['task']==task)]
            if not group:continue
            defined=[r for r in group if r['cosine'] is not None]
            out[method][task]={'rows':len(group),'defined_cosine_rows':len(defined),
                'mean_cosine':float(np.mean([r['cosine'] for r in defined])) if defined else None,
                'min_cosine':min((r['cosine'] for r in defined),default=None),
                'max_cosine':max((r['cosine'] for r in defined),default=None),
                'mean_sign_agreement':float(np.mean([r['sign_agreement'] for r in group])),
                'mean_norm_ratio':float(np.mean([r['norm_ratio'] for r in group if r['norm_ratio'] is not None])) if any(r['norm_ratio'] is not None for r in group) else None,
                'zero_local_norm_rows':sum(r['estimate_norm']==0 for r in group),
                'zero_full_norm_rows':sum(r['reference_norm']==0 for r in group)}
    return out

def main():
    start=time.perf_counter();deadline=start+180;cap=apply_memory_limit()
    reserve=ROOT/'local/trained-credit-oracle-v1-reservation';reserve.parent.mkdir(exist_ok=True);reserve.mkdir(exist_ok=False)
    run=ROOT/'runs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-trained-credit-oracle-v1-'+uuid.uuid4().hex[:6])
    run.mkdir();(run/'raw').mkdir();(reserve/'run.txt').write_text(str(run),encoding='utf-8')
    parent=ROOT/'runs'/PARENT;manifest=read(parent/'manifest.json');parent_result=read(parent/'result.json')
    if manifest['source_commit']!=SOURCE or parent_result['error'] is not None or parent_result['completed_jobs']!=132:raise ValueError('Completed pinned reference required')
    hashes=dict(manifest['source_lock']['files']);hashes[PROTOCOL]=sha((ROOT/PROTOCOL).read_bytes());hashes[SCRIPT]=sha((ROOT/SCRIPT).read_bytes())
    def check_source():
        for name,h in hashes.items():
            if sha((ROOT/name).read_bytes())!=h:raise ValueError('Locked source/protocol changed: '+name)
    check_source()
    lock={'scope':'Frozen trained-credit diagnostic, not quality/DEV selection',
        'protocol':PROTOCOL,'source_sha256':hashes,'reference_source_commit':SOURCE,
        'current_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'parent_run':PARENT,'parent_manifest_sha256':sha((parent/'manifest.json').read_bytes()),
        'methods':METHODS,'tasks':TASKS,'seeds':SEEDS,'train_indices':INDICES,'max_rows':180,'max_wall_seconds':180,
        'frozen_normalizer':True,'learning_updates_allowed':0,'fresh_confirmation':False,'pid':os.getpid(),
        'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'threads':1}}
    write(run/'manifest.json',lock);bank_hashes=read(parent/'dataset-hashes.json');rows=[];data={};receipts={};error=None
    monitor=ConfirmationResources(interval=.05)
    def check():
        monitor.check()
        if time.perf_counter()>deadline:raise TimeoutError('Registered180second diagnostic limit')
    with monitor:
      try:
        for task in TASKS:
          for seed in SEEDS:
            file=parent/'datasets'/f'{task}-{seed}.npz'
            if sha(file.read_bytes())!=bank_hashes[file.name]:raise ValueError('Frozen reference dataset changed')
            with np.load(file,allow_pickle=False) as z:
                # No ID/OOD labels or inputs are accessed by the diagnostic.
                data[(task,seed)]=(z['train_x'].copy(),z['train_y'].copy())
            check()
        for method in METHODS:
          for task in TASKS:
            for seed in SEEDS:
              check_source();check();folder=parent/'jobs'/f'{method}-{task}-{seed}-equal_updates'
              row=read(folder/'result.json');blob=(folder/'model.zip').read_bytes()
              if (row['method'],row['task'],row['seed'],row['budget'],row['shuffled'])!=(method,task,seed,'equal_updates',False):raise ValueError('Unexpected checkpoint cell')
              if sha(blob)!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed')
              cls=model_class(method);original=cls.loads(blob);original_blob=original.dumps()
              if original_blob!=blob:raise ValueError('Checkpoint roundtrip mismatch')
              frozen=learning_hash(original);payload={};xs,ys=data[(task,seed)]
              for index in INDICES:
                check();m=cls.loads(blob);before=learning_hash(m);ticks=m.total_ticks
                x=xs[index];target=int(ys[index]);m.reset_episode()
                for v in x:m.step(v,track=True)
                if learning_hash(m)!=before or m.total_ticks!=ticks+len(x):raise ValueError('Local rollout changed learning state')
                f=m.core.features.copy();fhat,raw,scale=m.standardized_features(f)
                logits=m.core.w_out@fhat+m.core.bias;logits-=np.max(logits);p=np.exp(logits);p/=np.sum(p)
                err=-p.copy();err[target]+=1
                signal=(m.core.w_out.T@err)*(np.abs(raw)<5)/scale
                local=(signal[:,None]*m.e_f).astype(np.float64).ravel()
                cpu=time.process_time();sensitivity=full_surrogate_sensitivity(original.core,x);oracle_cpu=time.process_time()-cpu
                full=signal.astype(np.float64)@sensitivity
                metrics=direction_metrics(local,full)
                loss=-float(np.log(max(float(p[target]),1e-12)))
                if not np.isfinite(loss) or learning_hash(m)!=before or learning_hash(original)!=frozen or original.dumps()!=original_blob:
                    raise ValueError('Oracle changed frozen learning state')
                rows.append({'method':method,'task':task,'seed':seed,'train_index':index,**metrics,
                    'frozen_loss':loss,'target':target,'direct_prediction':int(np.argmax(p)),
                    'clipped_feature_fraction':float(np.mean(np.abs(raw)>=5)),
                    'oracle_cpu_seconds':oracle_cpu,'checkpoint_sha256':row['checkpoint_sha256'],
                    'learning_state_sha256':frozen,'new_learning_updates':0,
                    'scope':'Within-checkpoint frozen normalizer/readout loss direction; before optimizer'})
                for name,value in {'local_direction':local,'full_direction':full,'signal':signal,
                   'eligibility':m.e_f.copy(),'features':f,'target':np.asarray(target),'loss':np.asarray(loss)}.items():payload.setdefault(name,[]).append(value)
                check()
              key=f'{method}-{task}-{seed}';rawfile=run/'raw'/f'{key}.npz'
              np.savez_compressed(rawfile,train_indices=np.asarray(INDICES,np.int32),**{k:np.asarray(v) for k,v in payload.items()})
              receipts[key]={'source_checkpoint_sha256':sha(blob),'diagnostic_raw_sha256':sha(rawfile.read_bytes()),'original_model_unchanged':original.dumps()==original_blob}
              print(json.dumps({'rows':len(rows),'max_rows':180,'method':method,'task':task,'seed':seed}),flush=True)
        if len(rows)!=180:raise ValueError('Incomplete fixed diagnostic grid')
        check_source();check()
      except Exception as exc:error=type(exc).__name__+': '+str(exc)
    result={'status':'COMPLETE_CONTROL_ONLY' if error is None else 'STOPPED_PRESERVE_PARTIAL','error':error,
        'rows':rows,'summary':summarize(rows),'checkpoint_receipts':receipts,'row_count':len(rows),
        'new_learning_updates':0,'candidate_selection_performed':False,'fresh_confirmation':False,'p2':False,
        'source_commit':SOURCE,'protocol_sha256':hashes[PROTOCOL],'cap':cap,'resources':monitor.report(),
        'wall_seconds':time.perf_counter()-start,
        'limitations':'Within-checkpoint estimator vs full surrogate; frozen stats/readout; not physical optimizer delta, not a cross-method matched-weights causal or quality comparison'}
    write(run/'result.json',result)
    print(json.dumps({'run':str(run),'status':result['status'],'rows':len(rows),'error':error,'wall_seconds':result['wall_seconds']}),flush=True)
    return int(error is not None)

if __name__=='__main__':raise SystemExit(main())
