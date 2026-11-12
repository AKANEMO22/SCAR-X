"""One supervised execution of the explicitly approved12seed confirmation."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.r8_confirmation import SEEDS,PRIMARY_TASKS,TASKS,BUDGETS,arms_for,summarize_confirmation
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit
from scarx.evaluation.r8_matched_study import dataset,refit_predict,direct_predict
from scarx.evaluation.runs import write_json
from scarx.neural.core import Config
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.work_trace import append,activity
from run_r8_matched_dev import atomic_json,train

POLICIES={'stopped_correct':(StoppedResetCredit,'multi_gated'),
    'own_reset_correct':(MatchedLocalCredit,'multi_gated'),
    'stopped_shuffle':(StoppedResetCredit,'multi_gated_shuffle'),
    'stopped_zero':(StoppedResetCredit,'multi_gated_zero'),
    'stopped_readout_only':(StoppedResetCredit,'multi_gated_readout_only')}


def hash_bytes(raw):return hashlib.sha256(raw).hexdigest()
def array_digest(array):return hash_bytes(str(array.dtype).encode()+repr(array.shape).encode()+np.ascontiguousarray(array).tobytes())


def evaluate_cell(model,data,initial_blob,control):
    weights_before=model.core.fingerprint();fitx,fity,_=data['refit'];fit=model.features_many(fitx)
    metrics={};predictions={};feature_hashes={'refit':array_digest(fit)}
    initial=type(model).loads(initial_blob) if control else None;initial_hashes={}
    hidden_unchanged=bool(np.array_equal(model.core.w_in,model.w_in_0) and np.array_equal(model.core.w_rec,model.w_rec_0))
    initial_features_match=True
    if control:
        original=initial.features_many(fitx);initial_hashes['refit']=array_digest(original)
        initial_features_match=bool(np.array_equal(fit,original))
    for split in ('id','ood'):
        x,y,_=data[split];features=model.features_many(x);direct=direct_predict(model,features);probe=refit_predict(fit,fity,features)
        feature_hashes[split]=array_digest(features)
        if control:
            original=initial.features_many(x);initial_hashes[split]=array_digest(original)
            initial_features_match=initial_features_match and bool(np.array_equal(features,original))
        metrics[split+'_direct']=float(np.mean(direct==y));metrics[split+'_probe']=float(np.mean(probe==y))
        predictions[split]={'direct':direct.tolist(),'probe':probe.tolist(),'target':y.tolist()}
    blob=model.dumps();interventions={}
    for name,rin,rrec in [('revert_in',True,False),('revert_rec',False,True),('revert_both',True,True)]:
        clone=type(model).loads(blob);clone.revert_hidden_weights(rin,rrec)
        features=clone.features_many(data['id'][0]);probe=refit_predict(clone.features_many(fitx),fity,features)
        direct=direct_predict(clone,features);target=data['id'][1]
        interventions[name]={'direct':direct.tolist(),'probe':probe.tolist(),'target':target.tolist(),
            'direct_accuracy':float(np.mean(direct==target)),'probe_accuracy':float(np.mean(probe==target))}
        metrics[name+'_probe']=interventions[name]['probe_accuracy'];metrics[name+'_direct']=interventions[name]['direct_accuracy']
    if weights_before!=model.core.fingerprint():raise ValueError('Evaluation changed neural weights')
    if control and (not hidden_unchanged or not initial_features_match):raise ValueError('Zero/readout control moved hidden weights/features')
    return metrics,predictions,interventions,hidden_unchanged,initial_features_match,feature_hashes,initial_hashes


def assert_source_lock(lock):
    for filename,expected in lock['source_sha256'].items():
        if hash_bytes((ROOT/filename).read_bytes())!=expected:raise ValueError('Frozen source changed: '+filename)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--seconds',type=float,required=True);args=parser.parse_args()
    run=Path(args.run).resolve()
    if run.parent!=ROOT/'runs' or not (run/'approval-and-lock.json').is_file():raise ValueError('Registered approved run required')
    if (run/'worker-result.json').exists() or (run/'datasets').exists():raise ValueError('Bank already opened; no rerun')
    lock=json.loads((run/'approval-and-lock.json').read_text(encoding='utf-8'));assert_source_lock(lock)
    with (run/'worker-started.json').open('x',encoding='utf-8') as marker:
        marker.write(json.dumps({'pid':os.getpid(),'fresh_bank_opening':True,'execution':'Once only, explicitly approved'}))
    cap=apply_memory_limit();started=time.perf_counter();deadline=started+args.seconds;rows=[];error=None
    status={'run':run.name,'status':'CONFIRMATION_DATA_VALIDATION','completed_jobs':0,'total_jobs':204,
        'seed_count':12,'fresh_confirmation':True,'approval':'Duyệt confirmation 12 seed','main_training':False,
        'model':'StoppedResetCredit, H48 ALIF, matched own/shuffle/zero/readout controls'}
    events={'learning':[],'answers':[]}
    def publish():atomic_json(run/'progress.json',status);atomic_json(run/'events.json',events)
    publish();atomic_json(ROOT/'local/research-observer-active.json',{'run':str(run)})
    activity(run,'Bắt đầu confirmation đã duyệt','12seed mới, protocol đã khóa; không tune, không P2/main/Hermes.',status='started',evidence='approval-and-lock.json')
    with ResourceMonitor(interval=.05) as monitor:
        try:
            (run/'datasets').mkdir();(run/'jobs').mkdir();all_data={};dataset_hashes={}
            for task in TASKS:
                for seed in SEEDS:
                    if time.perf_counter()>deadline:raise TimeoutError('Confirmation global wall limit')
                    d=dataset(task,seed);all_data[(task,seed)]=d
                    fields={name+'_'+suffix:value for name,(x,y,_) in d.items() for suffix,value in [('x',x),('y',y)]}
                    path=run/'datasets'/f'{task}-{seed}.npz';np.savez(path,**fields);dataset_hashes[path.name]=hash_bytes(path.read_bytes())
                    sets=[{hash_bytes(x.tobytes()) for x in d[split][0]} for split in d]
                    if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]):raise ValueError('Raw split overlap')
                    status.update(validated_datasets=len(all_data));publish();monitor.check()
            write_json(run/'dataset-hashes.json',dataset_hashes)
            for task in TASKS:
                for seed in SEEDS:
                    data=all_data[(task,seed)]
                    for arm in arms_for(task):
                        assert_source_lock(lock);job_deadline=min(deadline,time.perf_counter()+120)
                        job=run/'jobs'/f'{task}-{seed}-{arm}';job.mkdir();status.update(status='CONFIRMATION_TRAINING',task=task,seed=seed,arm=arm)
                        for budget in BUDGETS:
                            cls,policy=POLICIES[arm];model=cls(Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed),policy)
                            model.calibrate(data['train'][0][:32]);initial_blob=model.dumps()
                            (job/(budget+'-initial.zip')).write_bytes(initial_blob);status['budget']=budget
                            with (job/(budget+'-progress.jsonl')).open('w',encoding='utf-8') as stream:
                                def progress(measurement):
                                    event={'task':task,'seed':seed,'arm':arm,'budget':budget,**measurement}
                                    stream.write(json.dumps(event,allow_nan=False)+'\n');stream.flush()
                                    events['learning'].append(event);events['learning']=events['learning'][-40:]
                                    status.update(latest_learning=event,worker_resources=monitor.report());publish();monitor.check()
                                training,history=train(model,*data['train'][:2],budget,job_deadline,progress)
                            np.savez(job/(budget+'-learning.npz'),history=history,columns=np.array(['step','loss','hidden_norm','readout_norm','train_cpu_seconds']))
                            if budget=='equal_updates' and training['episodes']!=1024:raise ValueError('Equal-update budget mismatch')
                            control=arm in ('stopped_zero','stopped_readout_only')
                            measured=evaluate_cell(model,data,initial_blob,control)
                            metrics,predictions,interventions,unchanged,feature_match,feature_hashes,initial_feature_hashes=measured
                            blob=model.dumps();checkpoint=job/(budget+'-model.zip');checkpoint.write_bytes(blob)
                            row={'task':task,'seed':seed,'arm':arm,'budget':budget,'training':training,'metrics':metrics,
                                'predictions':predictions,'interventions':interventions,'hidden_unchanged':unchanged,
                                'initial_features_match':feature_match if control else None,'feature_sha256':feature_hashes,
                                'initial_feature_sha256':initial_feature_hashes,'checkpoint_sha256':hash_bytes(blob),
                                'initial_checkpoint_sha256':hash_bytes(initial_blob),
                                'signal_rng_state':model.signal_rng.bit_generator.state}
                            write_json(job/(budget+'-result.json'),row);rows.append(row)
                            for i in range(8):events['answers'].append({'task':task,'seed':seed,'arm':arm,'budget':budget,'input':data['id'][2][i],
                                'prediction':predictions['id']['direct'][i],'target':predictions['id']['target'][i],'qwen_output':None,
                                'scope':'Actual fresh-seed confirmation event-task output, no label supplied to neural inference'})
                            events['answers']=events['answers'][-32:];status['latest_metrics']=metrics;publish();monitor.check()
                            append(run,'learning',{'task':task,'seed':seed,'condition':arm,'budget':budget,**training,'metrics':metrics,'actual':True,'fresh_confirmation':True})
                            if time.perf_counter()>job_deadline:raise TimeoutError('Confirmation per-job wall limit')
                        status['completed_jobs']+=1;publish()
                        print(json.dumps({'completed':status['completed_jobs'],'total':204,'task':task,'seed':seed,'arm':arm}),flush=True)
            assert_source_lock(lock);status['status']='CONFIRMATION_NUMERIC_COMPLETE_AWAITING_AUDIT'
        except Exception as exc:error=type(exc).__name__+': '+str(exc);status.update(status='CONFIRMATION_STOPPED',error=error)
    summary=summarize_confirmation(rows)
    result={'summary':summary,'row_count':len(rows),'completed_jobs':status['completed_jobs'],'error':error,
        'worker_resources':monitor.report(),'cap':cap,'fresh_confirmation':True,'forward_changed':False,'p2_tested':False,
        'production_ready':False,'decision':'NUMERIC_COMPLETE_AWAITING_INTEGRITY_AUDIT' if error is None else 'STOPPED_PRESERVE_PARTIAL'}
    write_json(run/'worker-result.json',result);status.update(decision=result['decision'],worker_resources=result['worker_resources']);publish()
    print(json.dumps({'worker_decision':result['decision'],'completed':status['completed_jobs'],'error':error}),flush=True)
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
