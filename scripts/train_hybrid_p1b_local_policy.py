"""Approved local CUDA ALIF utility learning on actual common VLM/tool outcomes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from datetime import datetime,timezone
import copy,hashlib,io,json
from pathlib import Path
import sys,time,traceback,uuid
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.learner import LocalALIF,features
from scarx.evaluation.recovery_policy import ACTION_ORDER
from scarx.hybrid.cuda_alif import cupy_backend
from scarx.hybrid.store import Store,encode
SEEDS=(260113,260137,260171)
ARMS={'frozen':'frozen','readout_only':'readout_only','hidden':'learned','shuffled':'shuffled'}
FAMILIES=('missing_parameter','unknown_parameter','invalid_format','invalid_value')


def input_sequence(row):
    public=json.loads(row['messages'][1]['content']);raw=public['last_tool_result'];error=raw.get('exception') or ''
    return np.concatenate([features(encode({'request':public['request'],'failed_request':raw['request']})),
        features('Actual observed tool error '+error,(bool(error),raw.get('executed',False),0,0,1/3))]).astype(np.float32)


def main():
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-p1b-local-error-learning-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    with (ROOT/'local/p1b-local-error-learning-v1-execution.json').open('x',encoding='utf-8') as f:json.dump({'run':str(run)},f)
    (ROOT/'local/current-p1b-local-learning.txt').write_text(str(run),encoding='utf-8')
    data_dir=Path((ROOT/'local/current-qualified-repair-data.txt').read_text(encoding='utf-8').strip())
    table_dir=Path((ROOT/'local/current-p1b-training-outcomes.txt').read_text(encoding='utf-8').strip())
    qualifier=Path((ROOT/'local/current-p1b-action-usefulness.txt').read_text(encoding='utf-8').strip())
    assert json.loads((qualifier/'result.json').read_text(encoding='utf-8'))['dataset_qualified_for_training']
    rows=json.loads((data_dir/'verified-data.json').read_text(encoding='utf-8'));bank=json.loads((table_dir/'result.json').read_text(encoding='utf-8'))
    assert not bank['failures'] and len(bank['cases'])==48
    lookup={r['id']:r for r in bank['cases']};train=[r for r in rows if r['split']=='TRAIN'];test=[r for r in rows if r['split']=='TEST']
    assert len(train)==32 and len(test)==16
    # A valid selector cannot exceed the best measured action on each task.
    # Retain the registered usefulness stop even when DEV initially qualified.
    for split,subset in [('TRAIN',train),('TEST',test)]:
        oracle=sum(any(lookup[r['id']]['outcomes'][a]['verified_task_effect'] for a in ACTION_ORDER) for r in subset)/len(subset)
        control=max(sum(lookup[r['id']]['outcomes'][a]['verified_task_effect'] for r in subset)/len(subset)
                    for a in ('show_schema','RAG_experience'))
        if oracle-control<.05:
            result={'verdict':'STOP_ACTION_BANK_CANNOT_MEET_REGISTERED_ADVANTAGE_GATE','split':split,
                'finite_action_oracle':oracle,'strongest_matched_control':control,'maximum_gain_pp':100*(oracle-control),
                'local_updates':0,'forward_calls':0,'retention_measured':False,
                'no_criterion_relaxed':True,'scientific_or_production_PASS':False}
            (run/'result.json').write_text(encode(result),encoding='utf-8')
            print(encode({'run':str(run),**result}),flush=True);return
    sequences={r['id']:input_sequence(r) for r in train+test};stream=[]
    for stage in range(20):
        choices=[r for r in train if r['error_kind']==FAMILIES[stage%4]]
        for j in range(12):
            row=choices[(stage//4*12+j)%len(choices)];behavior=(stage*12+j)%6
            stream.append({'stage':stage,'row':row['id'],'behavior_action':behavior,
                'reward':1. if lookup[row['id']]['outcomes'][ACTION_ORDER[behavior]]['verified_task_effect'] else -1.})
    (run/'locked-stream.json').write_text(encode(stream),encoding='utf-8')
    sources=[Path(__file__),ROOT/'src/scarx/hybrid/learner.py',ROOT/'src/scarx/hybrid/cuda_alif.py',ROOT/'src/scarx/neural/core.py',ROOT/'experiments/hybrid-p1b-error-learning-v1.md']
    for p in sources:
        out=run/'source'/p.relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes())
    manifest={'seeds':SEEDS,'arms':ARMS,'families':FAMILIES,'stage_count':20,'source_git_sha':os.environ.get('SCARX_P1B_SOURCE_SHA'),
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'data_run':data_dir.name,'action_table_run':table_dir.name,'qualifier_run':qualifier.name,
        'input_adapter':'owned-task-error-two-block64-v1','action_order':ACTION_ORDER,
        'behavior_origin':'fixed common exploration, not neural on-policy credit','new_unique_TRAIN_errors':32,'exposures':240,
        'replay_per_stage':12,'replay_capacity':64,'max_local_updates':4500,'max_forwards':20000,'max_seconds':1800,
        'TEST_ids':[r['id'] for r in test],'gates':{'acquisition':.65,'mean_forgetting_pp':3,'worst_family_forgetting_pp':5,'gain_pp':5,'CI95_lower_gt':0}}
    (run/'manifest.json').write_text(encode(manifest),encoding='utf-8')
    trace=Store(ROOT/'local/hybrid-v1');start=time.time();updates=0;forwards=0;resources=[]
    report={'run_id':run.name,'cells':[],'failures':[],'VLM_weights_changed':False,'main_or_live_weights_changed':False,'scientific_or_production_PASS':False}
    cp=cupy_backend();cp.cuda.Device(0).use()
    def check():
        free,total=cp.cuda.runtime.memGetInfo();m=psutil.Process().memory_info()
        resources.append({'epoch':time.time(),'CUDA_available_free_bytes':int(free),'RSS_bytes':m.rss,'private_bytes':getattr(m,'private',0)})
        if time.time()-start>1800 or max(m.rss,getattr(m,'private',0))>2_000_000_000:raise RuntimeError('Time/host cap')
        if free<512*1024**2:raise MemoryError('CUDA reserve')
    def charge(kind,seed,arm,row):
        nonlocal updates,forwards
        check()
        if kind=='learn':
            if updates>=4500:raise RuntimeError('Update cap')
            updates+=1
        else:
            if forwards>=20000:raise RuntimeError('Forward cap')
            forwards+=1
        with (run/'attempts.jsonl').open('a',encoding='utf-8') as f:f.write(encode({'epoch':time.time(),'kind':kind,'seed':seed,'arm':arm,'row':row,'updates':updates,'forwards':forwards})+'\n')
    def evaluate(model,seed,arm,stage):
        predictions=[]
        for row in test:
            charge('forward',seed,arm,row['id']);p=model.forward(sequences[row['id']])[0];a=int(np.argmax(p))
            predictions.append({'id':row['id'],'family':row['error_kind'],'action':ACTION_ORDER[a],
                'success':bool(lookup[row['id']]['outcomes'][ACTION_ORDER[a]]['verified_task_effect']),'probabilities':p.tolist()})
        value={'stage':stage,'accuracy':float(np.mean([r['success'] for r in predictions])),
            'by_family':{f:float(np.mean([r['success'] for r in predictions if r['family']==f])) for f in FAMILIES},'predictions':predictions}
        trace.event('work_trace.activity',{'title':'P1B heldout actual-effect score','kind':'actual_SNN_evaluation','run_id':run.name,'seed':seed,'arm':arm,**value},'orchestrator')
        return value
    try:
        for seed in SEEDS:
            for arm,mode in ARMS.items():
                model=LocalALIF(seed,mode).to_device('gpu',0);cell={'seed':seed,'arm':arm,'stages':[],'initial':evaluate(model,seed,arm,-1)}
                memory=[];rng=np.random.default_rng(seed+810000);seen=0;initial=model.dumps()
                (run/f'{seed}-{arm}-initial.npz').write_bytes(initial)
                for stage in range(20):
                    for item in stream[stage*12:(stage+1)*12]:
                        row=item['row'];a=item['behavior_action'];evidence={'kind':'machine_verifier','id':table_dir.name+':'+row+':'+ACTION_ORDER[a]}
                        if mode!='frozen':
                            charge('learn',seed,arm,row);metrics=model.learn(sequences[row],a,reward=item['reward'],evidence=evidence,remember=False)
                            trace.event('work_trace.activity',{'title':'P1B actual local synapse update','kind':'actual_local_learning','run_id':run.name,
                                'seed':seed,'arm':arm,'stage':stage,'row':row,'behavior_origin':'common fixed exploration','action':ACTION_ORDER[a],**metrics},'orchestrator')
                        seen+=1
                        if len(memory)<64:memory.append(copy.deepcopy(item))
                        else:
                            i=int(rng.integers(seen))
                            if i<64:memory[i]=copy.deepcopy(item)
                    if mode!='frozen':
                        for _ in range(12):
                            item=memory[int(rng.integers(len(memory)))];row=item['row'];a=item['behavior_action'];charge('learn',seed,arm,'replay:'+row)
                            model.learn(sequences[row],a,reward=item['reward'],evidence={'kind':'machine_verifier','id':table_dir.name+':'+row+':'+ACTION_ORDER[a]},remember=False)
                    cell['stages'].append(evaluate(model,seed,arm,stage))
                state={'format':'p1b-owned-policy-v1','action_order':list(ACTION_ORDER),'input_adapter':manifest['input_adapter'],'external_replay':memory,'rng':rng.bit_generator.state}
                final=model.dumps(state);(run/f'{seed}-{arm}-final.npz').write_bytes(final)
                cold,restored=LocalALIF.loads(final,device='gpu',gpu_device=0);assert cold.dumps(restored)==final
                charge('forward',seed,arm,'cold-control');p=model.forward(sequences[test[0]['id']])[0]
                charge('forward',seed,arm,'cold-restored');q=cold.forward(sequences[test[0]['id']])[0];np.testing.assert_array_equal(p,q)
                cell['cold_restore_exact']=True
                forget={f:100*max(0,max(s['by_family'][f] for s in cell['stages'])-cell['stages'][-1]['by_family'][f]) for f in FAMILIES}
                cell.update(forgetting_pp=forget,mean_forgetting_pp=float(np.mean(list(forget.values()))),worst_family_forgetting_pp=max(forget.values()),acquisition=max(s['accuracy'] for s in cell['stages']))
                cell['joint_acquisition_retention_gate']=cell['acquisition']>=.65 and cell['mean_forgetting_pp']<=3 and cell['worst_family_forgetting_pp']<=5 and mode!='frozen'
                if arm=='hidden':
                    reset,_=LocalALIF.loads(final,device='gpu',gpu_device=0);fresh,_=LocalALIF.loads(initial,device='gpu',gpu_device=0)
                    reset.core.w_in[:]=fresh.core.w_in;reset.core.w_rec[:]=fresh.core.w_rec
                    cell['hidden_reset_final']=evaluate(reset,seed,'hidden_reset',20)
                report['cells'].append(cell)
                trace.event('work_trace.activity',{'title':'P1B cell completed','run_id':run.name,**{k:v for k,v in cell.items() if k not in ('stages','initial','hidden_reset_final')}},'orchestrator')
        controls={name:[{'id':r['id'],'success':lookup[r['id']]['outcomes']['show_schema' if name=='fixed_rule' else 'RAG_experience']['verified_task_effect']} for r in test] for name in ('fixed_rule','RAG_native_experience')}
        report.update(completed=True,verdict='P1B_COMPLETED_REPORT_GATES_WITHOUT_LIVE_PROMOTION',controls=controls,
            TEST_finite_oracle=sum(any(lookup[r['id']]['outcomes'][a]['verified_task_effect'] for a in ACTION_ORDER) for r in test)/16)
    except Exception as exc:
        report['failures'].append(type(exc).__name__+': '+str(exc));report['verdict']='P1B_STOPPED_NOT_PASS'
        (run/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        report.update(local_update_attempts=updates,forward_attempts=forwards,elapsed_seconds=time.time()-start)
        (run/'result.json').write_text(encode(report),encoding='utf-8');(run/'resources.json').write_text(encode(resources),encoding='utf-8')
        trace.event('work_trace.activity',{'title':'P1B terminal','run_id':run.name,**{k:v for k,v in report.items() if k not in ('cells','controls')}},'orchestrator');trace.close()
    print(encode({'run':str(run),**{k:v for k,v in report.items() if k not in ('cells','controls')}}),flush=True)


if __name__=='__main__':main()
