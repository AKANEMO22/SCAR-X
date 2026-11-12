"""Validate real diagnostic clones and stress outputs, with ZERO learn calls.

Performs forward/trace replay and independently reconstructs the single logged
optimizer update as array arithmetic. Does not feed anything into candidates.
"""
from pathlib import Path
import argparse,json,os,sys,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.acr import AdaptiveCreditRouting as ACR,hidden_matrix
from scarx.evaluation.acr_study import stress_cases,sha
from scarx.evaluation.r8_matched_study import TASKS,SEEDS,direct_predict,refit_predict

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def cosine(a,b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel();norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/norm) if norm else None
def corr(a,b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel()
    if a.shape!=b.shape:raise ValueError('Correlation shape mismatch')
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None
def scalar_accuracy(pred,y):return sum(int(a)==int(b) for a,b in zip(pred,y))/len(y)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();run=(ROOT/args.run).resolve()
    if run.parent!=ROOT/'runs' or (run/'mechanism-audit-v1.json').exists():raise ValueError('Unique owned audit required')
    start=time.perf_counter();deadline=start+600;result=read(run/'result.json')
    if result['error'] is not None:raise ValueError('Worker incomplete')
    nominee=read(run/'nominee-lock.json')['profile'];profiles=list(dict.fromkeys(('static_0','static_100',nominee)))
    if len(profiles)!=3:raise ValueError('Nominee unexpectedly endpoint')
    groups={};hashes={};diags=stress_count=prediction_count=direct_replay=0;worst_mix_error=0.;max_delta_error=0.
    stress_summary={};cells={};raw_rows=[]
    for file in (run/'jobs').glob('*/result-0.json'):
        r=read(file)
        if r['budget']=='equal_updates' and r['condition']=='learned_mtsg':cells[(r['task'],r['seed'],r['profile'])]=r
    for profile in profiles:
      for task in TASKS:
        for seed in SEEDS:
            if time.perf_counter()>deadline:raise TimeoutError('Mechanism audit10minute limit')
            row=cells[(task,seed,profile)];blob=(run/row['path']/'model-0.zip').read_bytes()
            if sha(blob)!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            model=ACR.loads(blob);frozen=hidden_matrix(model);dfile=run/'datasets'/f'{task}-{seed}.npz'
            with np.load(dfile,allow_pickle=False) as z:data={s:(z[s+'_x'],z[s+'_y'],[]) for s in ('train','refit','id','ood')}
            # Source-independent summaries were already checked in primary raw
            # audit; restore event summaries for the fixed stress transformer.
            from scarx.evaluation.r8_matched_study import event_oracle
            labels,summaries=event_oracle(task,data['id'][0])
            if not np.array_equal(labels,data['id'][1]):raise ValueError('Stress input oracle mismatch')
            data['id']=(data['id'][0],data['id'][1],summaries)
            fit=model.features_many(data['refit'][0]);stress_file=run/'diagnostics'/f'{task}-{seed}-{profile}-stress.json'
            stress=read(stress_file);expected_stress=stress_cases(task,data)
            if set(stress)!=set(expected_stress):raise ValueError('Stress cases missing/unregistered')
            for name,(x,y) in expected_stress.items():
                r=stress[name]
                if r['target']!=y.tolist():raise ValueError('Stress labels differ')
                for metric in ('direct','probe'):
                    if len(r[metric])!=128 or not np.isin(r[metric],[0,1]).all():raise ValueError('Stress predictions invalid')
                    if scalar_accuracy(r[metric],r['target'])!=r[metric+'_accuracy']:raise ValueError('Stress scalar metric mismatch')
                    prediction_count+=128
                # Recompute every direct stress prediction and an independent
                # ridge-head implementation from existing refit data only.
                features=model.features_many(x)
                if direct_predict(model,features).tolist()!=r['direct']:raise ValueError('Stress hard-forward replay mismatch')
                if refit_predict(fit,data['refit'][1],features).tolist()!=r['probe']:raise ValueError('Stress probe replay mismatch')
                direct_replay+=len(x)
                if name=='query_flip':
                    value=float(np.mean(np.asarray(r['direct'])==1-model.predict_many(data['id'][0][:128])))
                    if value!=r['paired_direct_flip_fraction']:raise ValueError('Paired query flip metric mismatch')
                group=stress_summary.setdefault(profile,{}).setdefault(task,{}).setdefault(name,[])
                group.append({'seed':seed,'direct':r['direct_accuracy'],'probe':r['probe_accuracy'],
                   'flip_consistency':r.get('paired_direct_flip_fraction')})
            hashes[str(stress_file.relative_to(run))]=sha(stress_file.read_bytes());stress_count+=1
            for index in (0,63,511,1023):
                file=run/'diagnostics'/f'{task}-{seed}-{profile}-{index}.npz'
                with np.load(file,allow_pickle=False) as z:tape={name:z[name] for name in z.files}
                expected={'u','theta','z','adaptation','own','stop','mixed','alpha','q','signal','hidden_delta','loss','target','direct_prediction','scope'}
                if set(tape)!=expected or any(not np.isfinite(v).all() for k,v in tape.items() if k!='scope'):
                    raise ValueError('Diagnostic fields/nonfinite')
                x=data['train'][0][index];target=int(data['train'][1][index]);m=ACR.loads(blob);m.reset_episode()
                before_weights=hidden_matrix(m);before_rng=m.signal_rng.bit_generator.state.copy()
                for t,input_row in enumerate(x):
                    m.step(input_row)
                    fields={'u':m.last_u,'theta':m.last_theta,'z':m.core.spikes,'adaptation':m.last_a,
                       'own':m.e_f_own,'stop':m.e_f_stop,'mixed':m.e_f,'alpha':m.alpha,'q':m.router_q}
                    for name,value in fields.items():np.testing.assert_array_equal(tape[name][t],value)
                if not np.array_equal(hidden_matrix(m),before_weights) or m.signal_rng.bit_generator.state!=before_rng:
                    raise ValueError('Forward/trace audit mutated learning weights/RNG')
                a=tape['alpha'][:,:,None];mixed=(1-a)*tape['own']+a*tape['stop']
                # Endpoints explicitly copy; numerical expression may differ only
                # in signed zero. Any finite numerical discrepancy is rejected.
                err=float(np.max(np.abs(mixed-tape['mixed'])));worst_mix_error=max(worst_mix_error,err)
                if err!=0:raise ValueError('Mix-after-filter arithmetic differs')
                if np.any((tape['alpha']<0)|(tape['alpha']>1)):raise ValueError('Alpha outside bounds')
                f=m.core.features;mean=.99*m.f_mean+.01*f
                var=.99*m.f_var+.01*(f-mean)**2
                scale=np.maximum(np.sqrt(var),.05).astype(np.float32);raw=(f-mean)/scale
                fhat=np.clip(raw,-5,5).astype(np.float32)
                logits=m.core.w_out@fhat+m.core.bias;logits-=np.max(logits);prob=np.exp(logits);prob/=prob.sum()
                loss=-float(np.log(max(float(prob[target]),1e-12)));error=-prob.copy();error[target]+=1
                signal=(m.core.w_out.T@error)*(np.abs(raw)<5)/scale
                np.testing.assert_array_equal(tape['signal'],signal)
                if int(tape['target'])!=target or float(tape['loss'])!=loss:raise ValueError('Diagnostic loss/target mismatch')
                normal=np.maximum(np.sqrt(m.f_var),.05).astype(np.float32)
                pred=int(np.argmax(m.core.w_out@np.clip((f-m.f_mean)/normal,-5,5)+m.core.bias))
                if int(tape['direct_prediction'])!=pred:raise ValueError('Diagnostic direct output differs')
                direction=signal[:,None]*tape['mixed'][-1]
                moment=.90*m.syn_trace+.10*direction;energy=.99*m.syn_energy+.01*(direction*direction)
                step=np.clip(.008*(moment/(np.sqrt(energy)+1e-4)),-.010,.010)
                after=before_weights.copy();after+=step;np.clip(after,-3,3,out=after)
                delta=(after.astype(float)-before_weights.astype(float)).astype(np.float32)
                deltaerr=float(np.max(np.abs(delta-tape['hidden_delta'])));max_delta_error=max(max_delta_error,deltaerr)
                if deltaerr!=0:raise ValueError('Logged physical update not source optimizer')
                own=signal[:,None]*tape['own'][-1];stop=signal[:,None]*tape['stop'][-1]
                item={'profile':profile,'task':task,'seed':seed,'train_index':index,
                   'alpha_mean':float(tape['alpha'].mean()),'alpha_std':float(tape['alpha'].std()),
                   'alpha_spike_correlation':corr(tape['alpha'],tape['z']),
                   'alpha_adaptation_correlation':corr(tape['alpha'],tape['adaptation']),
                   'trace_cosine':cosine(tape['own'],tape['stop']),
                   'signal_own_stop_cosine':cosine(own,stop),'raw_own_norm':float(np.linalg.norm(own.astype(float))),
                   'raw_stop_norm':float(np.linalg.norm(stop.astype(float))),
                   'actual_update_norm':float(np.linalg.norm(delta.astype(float))),
                   'actual_update_raw_direction_cosine':cosine(delta,direction),'loss':loss,
                   'alpha_per_neuron_mean':tape['alpha'].mean(0).tolist(),'alpha_per_tick_mean':tape['alpha'].mean(1).tolist()}
                raw_rows.append(item);groups.setdefault(profile,{}).setdefault(task,[]).append(item)
                hashes[str(file.relative_to(run))]=sha(file.read_bytes());diags+=1
            if not np.array_equal(hidden_matrix(model),frozen):raise ValueError('Stress audit changed weights')
    if diags!=144 or stress_count!=36:raise ValueError('Required diagnostic grid incomplete')
    summary={}
    for profile,tasks in groups.items():
        summary[profile]={}
        for task,items in tasks.items():
            fields=['alpha_mean','alpha_std','alpha_spike_correlation','alpha_adaptation_correlation','trace_cosine','signal_own_stop_cosine',
                 'raw_own_norm','raw_stop_norm','actual_update_norm','actual_update_raw_direction_cosine','loss']
            summary[profile][task]={key:float(np.mean([v[key] for v in items if v[key] is not None])) if any(v[key] is not None for v in items) else None for key in fields}
            summary[profile][task]['alpha_per_neuron_mean']=np.mean([v['alpha_per_neuron_mean'] for v in items],axis=0).tolist()
            summary[profile][task]['alpha_per_tick_mean']=np.mean([v['alpha_per_tick_mean'] for v in items],axis=0).tolist()
    report={'status':'PASS','diagnostic_files':diags,'stress_files':stress_count,'stress_prediction_entries_rescored':prediction_count,
       'stress_direct_predictions_replayed':direct_replay,'worst_mixture_error':worst_mix_error,'max_physical_delta_error':max_delta_error,
       'mechanism_summary':summary,'stress_summary':stress_summary,'diagnostic_rows':raw_rows,'artifact_sha256':hashes,
       'new_learning_updates':0,'audit_wall_seconds':time.perf_counter()-start,
       'scope':'Final-checkpoint diagnostic clones, not original training trajectory; current learning signal/stat update and source optimizer reconstructed without learn calls'}
    with (run/'mechanism-audit-v1.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps({k:report[k] for k in ('status','diagnostic_files','stress_files','audit_wall_seconds')}))
if __name__=='__main__':main()
