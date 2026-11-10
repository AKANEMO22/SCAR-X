"""Bounded readout-only current-model clones, fixed dev selection and acceptance."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,itertools
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from scarx.neural.unified import UnifiedCore
from scarx.neural.language import tokenize
from scarx.learning.dialogue import normalized
from scarx.learning.verifiers import parse_integer_expression,integer_answer
from scarx.app.workload import plan_workload
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from rehearse_g1_basics import score,probe_idle,digit_score,STATES
from generate_g1_state_paraphrases import INTENTS

CONDITIONS=('raw_old','raw_expanded','surface_old','surface_expanded');SEEDS=(1801,1811,1823)


def train_trial(job):
    condition,seed,runpath,privatepath,pools,proofs,dev=job;run=Path(runpath);private=Path(privatepath)
    cap=apply_memory_limit();began=time.perf_counter();model=UnifiedCore.loads((private/'source-model.zip').read_bytes())
    if condition.startswith('surface'):model.encoding['surface_normalization']='terminal-punctuation-v1'
    source_hidden={n:getattr(model.core,n).copy() for n in ('w_in','w_rec')};allocation=model.core.config.hidden
    base_updates=model.core.updates;base_ticks=model.ticks
    schedule_rng=np.random.default_rng(seed);prompt_rng=np.random.default_rng(seed+1);group_counts=np.zeros(9,np.int32);losses=[]
    with ResourceMonitor(interval=.2) as monitor:
        before=model.core.w_out.copy()
        for index in range(3072):
            if index%5==0:
                sample=proofs[int(schedule_rng.integers(len(proofs)))];group_counts[8]+=1
            else:
                group=int(schedule_rng.integers(8));pool=pools[condition.endswith('expanded')][group]
                sample=pool[int(prompt_rng.integers(len(pool)))];group_counts[group]+=1
            for _ in range((0,12,48)[index%3]):model.tick()
            if index%8==0:probe_idle(model)
            model.select_capacity(plan_workload(model,sample['prompt'])['requested_neurons'])
            losses.append(float(model.train_pair(sample['prompt'],sample['target'],'readout')))
            if (index+1)%256==0:
                delta=float(np.linalg.norm(model.core.w_out-before))
                append(run,'learning',{'scope':'G1 curated curriculum clone · output-local/readout-only, not serving',
                    'prompt':f'{condition} · order{seed} · examples{index-254}–{index+1}','passes':1,'sample_count':256,
                    'loss_mean':float(np.mean(losses[-256:])),'losses':[float(np.mean(losses[-256:]))],
                    'update_steps':model.core.updates-base_updates,'update_unit':'token cumulative in trial','hidden_update_steps':0,
                    'weight_delta':{'w_in':0.,'w_rec':0.,'w_out':delta},'attempted_weight_delta':{'w_in':0.,'w_rec':0.,'w_out':delta},
                    'changed_synapses':{'w_in':0,'w_rec':0,'w_out':int(np.count_nonzero(model.core.w_out!=before))},
                    'accepted':True,'rolled_back':False,'guard_checked':False,'reason':'Private clone updates only; acceptance and serving commit not yet approved by measurements.'})
                before=model.core.w_out.copy();monitor.check()
                if time.perf_counter()-began>180:raise TimeoutError('Registered180s curriculum trial limit')
        for name,old in source_hidden.items():np.testing.assert_array_equal(getattr(model.core,name),old)
        assert model.core.config.hidden==allocation
        path=private/f'{condition}-{seed}.zip';path.write_bytes(model.dumps());assert UnifiedCore.loads(path.read_bytes()).dumps()==model.dumps()
        responses=score(model.dumps(),dev)
        states={state:float(np.mean([r['exact'] for r in responses if r['state']==state])) for state in STATES}
        record={'condition':condition,'curriculum_order_seed':seed,'development_by_state':states,'development_mean':float(np.mean(list(states.values()))),
            'training_examples':3072,'label_group_counts':group_counts.tolist(),'neural_updates':model.core.updates-base_updates,
            'neural_ticks':model.ticks-base_ticks,'hidden_weights_unchanged':True,'allocation_unchanged':True,
            'mean_loss':float(np.mean(losses)),'first256_loss':float(np.mean(losses[:256])),'last256_loss':float(np.mean(losses[-256:])),
            'candidate_private_path':path.relative_to(root).as_posix(),'candidate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'seconds':time.perf_counter()-began,'resources':monitor.report(),'cap':cap}
        write_json(private/f'{condition}-{seed}-development.json',responses);write_json(run/f'{condition}-{seed}.json',record)
        return record


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    if any((p.info['name'] or '').lower()=='llama-server.exe' for p in psutil.process_iter(['name'])):raise ValueError('Stop reference before curriculum training')
    cap=apply_memory_limit();began=time.perf_counter()
    info=json.loads((root/'artifacts/g1-curriculum-reviewed.json').read_text(encoding='utf-8'));raw=(root/info['data']).read_bytes();assert hashlib.sha256(raw).hexdigest()==info['sha256']
    data=json.loads(raw)['items'];old_run=json.loads((root/'artifacts/g1-basic-rehearsal-latest.json').read_text(encoding='utf-8'))
    previous=(root/old_run['conditions'][0]['candidate_private_path']).parent
    original=(previous/'source-model.zip').read_bytes();baseline_data=json.loads((previous/'source-outputs.json').read_text(encoding='utf-8'))
    cases={r['case_id']:{'id':r['case_id'],'prompt':r['prompt'],'target':r['target'],'group':r['group']} for r in baseline_data['outputs'] if r['group']!='personal_guard'}
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'));skipped=set(baseline_data['skipped'])
    old_pools=[]
    for i,family in enumerate(corpus['families']):old_pools.append([{'id':f'known-{i}-{j}','prompt':p,'target':family['answer'],'group':'known'} for j,p in enumerate(family['train']) if f'known-{i}-{j}' not in skipped])
    expanded=[old_pools[i]+[r for r in data if r['split']=='train' and r['intent']==intent] for i,intent in enumerate(INTENTS)]
    proofs={normalized(r['prompt']):r for r in baseline_data['proofs']}
    for r in proofs.values():
        expr=parse_integer_expression(r['prompt']);assert expr is not None and normalized(r['target'])==str(integer_answer(expr))
    proofs=list(proofs.values());assert proofs
    model=UnifiedCore.loads(original)
    if any(any(t not in model.ids for t in tokenize(r['target'])) for pool in expanded for r in pool):raise ValueError('Unrepresentable target')
    guards={r['id']:r for r in baseline_data['guards']}
    for r in proofs:guards[r['id']]={**r,'group':'personal_guard'}
    guard_list=list(guards.values());acceptance=list(cases.values())+guard_list
    dev=[{**r,'group':'development'} for r in data if r['split']=='development']
    run=new_run(root,'g1-curated-curriculum-v1',{'protocol':'experiments/g1-curated-curriculum-v1.md',
        'display_title':'G1 · curriculum Antigravity + proof replay · synapse học thật',
        'model_scope':'Private current clones, same hidden/vocab; output-local updates only, no automatic serving replacement',
        'conditions':CONDITIONS,'order_seeds':SEEDS,'examples':3072,'hidden_rule_changed':False,'workers':2},
        data_description='117reviewed public trainingprompts,31development; prior48diagnostic nevertrained, verifiedprivatearithmetic replay localonly',data_files=[info['data'],'data/dialogue/generation1.json'])
    private=root/'local/g1-curated-curriculum'/run.name;private.mkdir(parents=True);(private/'source-model.zip').write_bytes(original)
    results=[];pending={};jobs=iter([(c,s,str(run),str(private),[old_pools,expanded],proofs,dev) for c,s in itertools.product(CONDITIONS,SEEDS)])
    with ResourceMonitor(interval=.1) as monitor:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for _ in range(2):job=next(jobs);pending[pool.submit(train_trial,job)]=job
            while pending:
                ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED);monitor.check()
                if time.perf_counter()-began>1200:raise TimeoutError('Registered curriculum round deadline')
                for future in ready:
                    pending.pop(future);row=future.result();results.append(row)
                    progress(run,f'{row["condition"]} · order{row["curriculum_order_seed"]} · development',len(results),12,
                        metrics={'dev_style_exact':row['development_mean'],'loss':row['last256_loss'],'hidden_weights_unchanged':True})
                    print(json.dumps({k:row[k] for k in ('condition','curriculum_order_seed','development_mean','seconds')}),flush=True)
                    try:job=next(jobs);pending[pool.submit(train_trial,job)]=job
                    except StopIteration:pass
        means={c:float(np.mean([r['development_mean'] for r in results if r['condition']==c])) for c in CONDITIONS}
        tie=('raw_expanded','surface_expanded','raw_old','surface_old');chosen=max(tie,key=lambda c:(means[c],-tie.index(c)))
        selection={'condition':chosen,'order_seed_for_serving_if_eligible':1801,'development_scores':means,'acceptance_opened':False}
        write_json(run/'selection-frozen.json',selection)
        baseline=score(original,acceptance);write_json(private/'source-acceptance.json',baseline)
        from sklearn.datasets import load_digits
        digits=load_digits();split=json.loads((root/'runs/20261109T000625Z-generation1-joint-bd6c92/vision-split.json').read_text(encoding='utf-8'))
        images=[(int(i),digits.data[i],int(digits.target[i])) for i in split['diagnostic_test'][:48]];old_digits=digit_score(original,images)
        accepted=[]
        for row in [r for r in results if r['condition']==chosen]:
            payload=(root/row['candidate_private_path']).read_bytes();responses=score(payload,acceptance);states={};passed=True
            devpass=row['development_mean']>=.6 and all(v>=.5 for v in row['development_by_state'].values());passed&=devpass
            for state in STATES:
                values=[r for r in responses if r['state']==state];old=[r for r in baseline if r['state']==state]
                known=float(np.mean([r['exact'] for r in values if r['group']=='known']));agy=float(np.mean([r['exact'] for r in values if r['group']=='antigravity']))
                old_agy=float(np.mean([r['exact'] for r in old if r['group']=='antigravity']))
                correct={r['case_id'] for r in old if r['group']=='personal_guard' and normalized(r['generated'])==normalized(r['target'])}
                lost=[r['case_id'] for r in values if r['case_id'] in correct and normalized(r['generated'])!=normalized(r['target'])]
                proof_ids={r['id'] for r in proofs};proof_accuracy=float(np.mean([normalized(r['generated'])==normalized(r['target']) for r in values if r['case_id'] in proof_ids]))
                statepass=known>=(.8 if state=='canonicalreset' else .75) and agy>=.25 and agy-old_agy>=.1 and not lost and proof_accuracy>=.95
                states[state]={'known':known,'antigravity':agy,'source_antigravity':old_agy,'lost_personal_runtime_guards':len(lost),'proof_accuracy':proof_accuracy,'pass':statepass};passed&=statepass
            new_digits=digit_score(payload,images);digitlost=sum(a['predicted']==a['target'] and b['predicted']!=b['target'] for a,b in zip(old_digits,new_digits))
            diff=float(np.mean([r['predicted']==r['target'] for r in new_digits])-np.mean([r['predicted']==r['target'] for r in old_digits]));passed&=digitlost==0 and diff>=-.03
            acceptance_row={**row,'states':states,'screen_pass':bool(passed),'lost_old_correct_digits':digitlost,'development_pass':bool(devpass)};accepted.append(acceptance_row)
            write_json(private/f'{chosen}-{row["curriculum_order_seed"]}-acceptance.json',{'outputs':responses,'digits':new_digits})
            for output in responses:
                if output['group']!='antigravity' or output['state']!='public_idle_probes':continue
                append(run,'answer',{**output,'scope':f'G1 curated clone · {chosen}/order{row["curriculum_order_seed"]} · not live',
                    'input_scope':'Actual public prompt. Fixed private clone trained with existing local readout; no selflabel or lookup. Surface normalization if declared.',
                    'checkpoint':'private candidate in local/','qwen':None})
            progress(run,f'Acceptance {chosen} · order{row["curriculum_order_seed"]}',len(accepted),3,
                metrics={'screen_pass':bool(passed),'known_reset':states['canonicalreset']['known'],'new_phrasing_reset':states['canonicalreset']['antigravity'],'lost_digits':digitlost})
        allpass=all(r['screen_pass'] for r in accepted)
        selected=next((r for r in accepted if r['curriculum_order_seed']==1801),None) if allpass else None
        result={'run':run.relative_to(root).as_posix(),'decision':'CURATED_PRODUCT_READY_FOR_SEPARATE_GUARDED_TRANSACTION' if allpass else 'CURATED_SCREEN_FAILED_KEEP_LIVE',
            'selection':selection,'development_trials':results,'conditions':accepted,'selected':selected,
            'all_order_passes':{chosen:allpass},'original_source_sha256':hashlib.sha256(original).hexdigest(),
            'original_source_private_path':(private/'source-model.zip').relative_to(root).as_posix(),
            'curated_train_prompts':117,'curated_development_prompts':31,'personal_guards':len(guard_list),'proof_replay_types':len(proofs),
            'deployed':False,'new_hidden_rule':False,'normal_chat_or_vision_proven':False,'lifelong_P2_pass':False,'resources':monitor.report(),'cap':cap,
            'limits':['Same sourceinitialization,3curriculumorders, not independentmodel confirmation.','Closed8intent stylematching, not generalbilingualconversation or normalphotos.',
                'Privateweights/rawguardresponses staylocal; priorAGY48 diagnostic was alreadyinspected, nevertrained.']}
        (run/'report.md').write_text('# Bounded G1 curated curriculum\n\nActual localreadouttraining, fixeddevelopmentselection, strictguardedproductacceptance; no automaticdeployment or researchgatepass.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-curated-curriculum-latest.json',result)
        print(json.dumps({'run':str(run),'decision':result['decision'],'selection':selection,
            'acceptance':[{'order':r['curriculum_order_seed'],'pass':r['screen_pass'],'states':r['states']} for r in accepted]}),flush=True)


if __name__=='__main__':main()
