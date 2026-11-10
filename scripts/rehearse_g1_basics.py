"""Bounded current-model clone rehearsal. No live DB writes or automatic deployment."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,io,zipfile,itertools,urllib.request
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.unified import UnifiedCore
from scarx.neural.language import tokenize
from scarx.learning.dialogue import normalized
from scarx.app.workload import plan_workload
from scarx.learning.verifiers import parse_integer_expression,integer_answer
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from audit_g1_conversational_state import prepare

STATES=('savedstate','silent_idle','public_idle_probes','canonicalreset')
ORDERS=(1701,1709,1721)


def score(payload,cases):
    out=[]
    for state in STATES:
        for case in cases:
            m=UnifiedCore.loads(payload);prepare(m,state);load=plan_workload(m,case['prompt']);m.select_capacity(load['requested_neurons'])
            tick=m.ticks;began=time.perf_counter();reply=m.generate(case['prompt'],24)
            out.append({'state':state,'case_id':case['id'],'prompt':case['prompt'],'target':case['target'],'generated':reply['text'],
                'exact':tokenize(reply['text'])==tokenize(case['target']),'neural_ticks':m.ticks-tick,'seconds':time.perf_counter()-began,
                'active':m.active_count,'allocated':m.core.config.hidden,'group':case['group']})
    return out


def probe_idle(model):
    model.select_capacity(48);model.boundary();context=model.context('chào bạn');previous=1
    for pos in range(4):
        token=int(model.word_step(previous,context,pos).argmax())
        if token==0:break
        previous=token


def digit_score(payload,images):
    model=UnifiedCore.loads(payload);out=[]
    for i,pixels,target in images:
        model.select_capacity(plan_workload(model,pixels=pixels)['requested_neurons'])
        out.append({'id':i,'target':target,'predicted':model.image(pixels)['digit']})
    return out


def trial(job):
    mode,seed,runpath,privatepath,cases,training,guards,proofs,baseline,baseline_digits,images=job
    run=Path(runpath);private=Path(privatepath);cap=apply_memory_limit();began=time.perf_counter()
    payload=(private/'source-model.zip').read_bytes();m=UnifiedCore.loads(payload);initial_hidden=m.core.config.hidden
    rng=np.random.default_rng(seed);schedule=training+proofs;losses=[];start_updates=m.core.updates
    with ResourceMonitor(interval=.2) as monitor:
        for epoch in range(24):
            before={n:getattr(m.core,n).copy() for n in ('w_in','w_rec','w_out')};epoch_losses=[]
            for position,index in enumerate(rng.permutation(len(schedule))):
                sample=schedule[index]
                for _ in range((0,12,48)[(epoch*len(schedule)+position)%3]):m.tick()
                if position%8==0:probe_idle(m)
                m.select_capacity(plan_workload(m,sample['prompt'])['requested_neurons'])
                epoch_losses.append(float(m.train_pair(sample['prompt'],sample['target'],mode)))
            losses+=epoch_losses;delta={n:float(np.linalg.norm(getattr(m.core,n)-old)) for n,old in before.items()}
            append(run,'learning',{'scope':'G1 personal snapshot clone · existing local '+mode+' rule · not live',
                'prompt':f'Basic rehearsal · order{seed} · {mode} · pass{epoch+1}',
                'passes':1,'sample_count':len(schedule),'loss_mean':float(np.mean(epoch_losses)),'losses':[float(np.mean(epoch_losses))],
                'update_steps':m.core.updates-start_updates,'update_unit':'token cumulative in trial',
                'weight_delta':delta,'attempted_weight_delta':delta,'changed_synapses':{n:int(np.count_nonzero(getattr(m.core,n)!=old)) for n,old in before.items()},
                'accepted':True,'rolled_back':False,'guard_checked':False,'reason':'Updates only in disposable private research clone. Final guards pending; not serving commit.'})
            monitor.check()
            if time.perf_counter()-began>120:raise TimeoutError('Registered120s rehearsal trial limit')
        assert m.core.config.hidden==initial_hidden
        path=private/f'{mode}-{seed}.zip';path.write_bytes(m.dumps());candidate=path.read_bytes()
        model_copy=UnifiedCore.loads(candidate);assert m.dumps()==model_copy.dumps()
        actual=score(candidate,cases+guards);digits=digit_score(candidate,images);states={};passed=True
        for state in STATES:
            values=[r for r in actual if r['state']==state];old=[r for r in baseline if r['state']==state]
            known=float(np.mean([r['exact'] for r in values if r['group']=='known']))
            agy=float(np.mean([r['exact'] for r in values if r['group']=='antigravity']))
            old_agy=float(np.mean([r['exact'] for r in old if r['group']=='antigravity']))
            old_correct={r['case_id'] for r in old if r['group']=='personal_guard' and r['exact']}
            lost=[r['case_id'] for r in values if r['case_id'] in old_correct and not r['exact']]
            statepass=known>=(.8 if state=='canonicalreset' else .75) and agy>=.25 and agy-old_agy>=.1 and not lost
            states[state]={'known':known,'antigravity':agy,'source_antigravity':old_agy,'lost_personal_guards':len(lost),'pass':statepass};passed&=statepass
        old_digit={r['id']:r for r in baseline_digits};digitlost=sum(old_digit[r['id']]['predicted']==r['target'] and r['predicted']!=r['target'] for r in digits)
        digit_accuracy=float(np.mean([r['predicted']==r['target'] for r in digits]));old_accuracy=float(np.mean([r['predicted']==r['target'] for r in baseline_digits]))
        passed&=digitlost==0 and digit_accuracy-old_accuracy>=-.03
        if mode=='readout':
            base=UnifiedCore.loads(payload)
            for n in ('w_in','w_rec'):np.testing.assert_array_equal(getattr(m.core,n),getattr(base.core,n))
        record={'mode':mode,'curriculum_order_seed':seed,'states':states,'screen_pass':bool(passed),'digit_accuracy':digit_accuracy,
            'source_digit_accuracy':old_accuracy,'lost_old_correct_digits':digitlost,'passes':24,'training_examples':len(schedule)*24,
            'neural_updates':m.core.updates-start_updates,'mean_loss':float(np.mean(losses)),
            'source_sha256':hashlib.sha256(payload).hexdigest(),'candidate_sha256':hashlib.sha256(candidate).hexdigest(),
            'candidate_private_path':str(path.relative_to(root)).replace('\\','/'),'new_vocabulary':False,'hidden_allocated':initial_hidden,
            'seconds':time.perf_counter()-began,'resources':monitor.report(),'cap':cap,'checkpoint_exact':True}
        write_json(private/f'{mode}-{seed}-outputs.json',{'outputs':actual,'digits':digits,'summary':record})
        for row in actual:
            if row['group']=='personal_guard':continue
            if row['state']!='public_idle_probes' or row['group']!='antigravity':continue
            append(run,'answer',{**row,'scope':f'Private G1 clone · rehearsal {mode}/order{seed} · not live',
                'input_scope':'Actual public prompt only; targets never used at inference. Candidate not serving. Private weights excluded from package.',
                'checkpoint':'private candidate in local/','qwen':None})
        write_json(run/f'{mode}-{seed}.json',record);return record


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    run=new_run(root,'g1-basic-rehearsal-v1',{'protocol':'experiments/g1-basic-rehearsal-v1.md','display_title':'G1 · ôn lại hội thoại cơ bản trên bản sao · giữ lời dạy cũ',
        'model_scope':'Private current-model clones only, no automatic replacement or reset; existing train_pair full/readout',
        'modes':['readout','full'],'order_seeds':ORDERS,'passes':24,'new_hidden_rule':False,'workers':2})
    private=root/'local/g1-rehearsal'/run.name;private.mkdir(parents=True)
    bundle=urllib.request.urlopen('http://127.0.0.1:8765/api/export',timeout=10).read()
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:payload=archive.read('model.zip');meta=json.loads(archive.read('session.json'))
    (private/'source-model.zip').write_bytes(payload)
    controller=meta.get('dialogue_learning',{});lessons=controller.get('lessons',[]) if isinstance(controller,dict) else []
    latest={normalized(r['prompt']):r for r in lessons if r.get('status')!='superseded'}
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'));training=[];cases=[];skipped=[]
    for i,f in enumerate(corpus['families']):
        for j,p in enumerate(f['train']):
            case={'id':f'known-{i}-{j}','prompt':p,'target':f['answer'],'group':'known'}
            lesson=latest.get(normalized(p))
            if lesson and normalized(lesson['target'])!=normalized(f['answer']):skipped.append(case['id']);continue
            training.append(case);cases.append(case)
    info=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'))
    cases += [{**r,'group':'antigravity'} for r in json.loads((root/info['data']).read_text(encoding='utf-8'))['items']]
    guards=[{'id':'personal-'+r['id'],'prompt':r['prompt'],'target':r['target'],'group':'personal_guard'} for r in lessons if r.get('status')=='learned'][-16:]
    proofs=[]
    for r in lessons:
        expr=parse_integer_expression(r['prompt'])
        if r.get('independently_verified') and expr is not None and normalized(r['target'])==str(integer_answer(expr)):
            proofs.append({'id':'arithmetic-'+r['id'],'prompt':r['prompt'],'target':r['target'],'group':'proof'})
    base=UnifiedCore.loads(payload)
    rejected_proofs=sum(any(t not in base.ids for t in tokenize(r['target'])) for r in proofs)
    proofs=[r for r in proofs if all(t in base.ids for t in tokenize(r['target']))][-8:]
    if any(any(t not in base.ids for t in tokenize(r['target'])) for r in training):raise ValueError('Untrained public vocabulary in rehearsal')
    from sklearn.datasets import load_digits
    digits=load_digits();split=json.loads((root/'runs/20261109T000625Z-generation1-joint-bd6c92/vision-split.json').read_text(encoding='utf-8'))
    images=[(int(i),digits.data[i],int(digits.target[i])) for i in split['diagnostic_test'][:48]]
    baseline=score(payload,cases+guards);baseline_digits=digit_score(payload,images)
    write_json(private/'source-outputs.json',{'outputs':baseline,'digits':baseline_digits,'guards':guards,'proofs':proofs,'skipped':skipped})
    results=[]
    with ResourceMonitor(interval=.1) as monitor:
        pending={};jobs=iter([(m,s,str(run),str(private),cases,training,guards,proofs,baseline,baseline_digits,images) for m,s in itertools.product(('readout','full'),ORDERS)])
        with ProcessPoolExecutor(max_workers=2) as pool:
            for _ in range(2):job=next(jobs);pending[pool.submit(trial,job)]=job
            while pending:
                ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED);monitor.check()
                if time.perf_counter()-began>900:raise TimeoutError('Rehearsal round deadline')
                for future in ready:
                    pending.pop(future);result=future.result();results.append(result)
                    progress(run,f'{result["mode"]} · order{result["curriculum_order_seed"]} · final guards',len(results),6,
                        metrics={'screen_pass':result['screen_pass'],'reset_known':result['states']['canonicalreset']['known'],
                            'reset_antigravity':result['states']['canonicalreset']['antigravity'],'lost_digits':result['lost_old_correct_digits']})
                    print(json.dumps({'mode':result['mode'],'order_seed':result['curriculum_order_seed'],'screen_pass':result['screen_pass'],'states':result['states'],'lost_digits':result['lost_old_correct_digits']}),flush=True)
                    try:job=next(jobs);pending[pool.submit(trial,job)]=job
                    except StopIteration:pass
        passes={mode:all(r['screen_pass'] for r in results if r['mode']==mode) for mode in ('readout','full')}
        chosen=next((mode for mode in ('readout','full') if passes[mode]),None)
        selection=next((r for r in results if r['mode']==chosen and r['curriculum_order_seed']==1701),None) if chosen else None
        result={'run':run.relative_to(root).as_posix(),'decision':'PRODUCT_CLONE_READY_FOR_SEPARATE_GUARDED_TRANSACTION' if chosen else 'REHEARSAL_SCREEN_FAILED_KEEP_LIVE',
            'conditions':results,'all_order_passes':passes,'selected':selection,'skipped_personal_overrides':len(skipped),
            'personal_guards':len(guards),'independently_verified_proof_replays':len(proofs),'unrepresentable_proofs_not_replayed':rejected_proofs,'resources':monitor.report(),'cap':cap,
            'deployed':False,'new_hidden_rule':False,'normal_chat_or_vision_proven':False,'lifelong_P2_pass':False,
            'limitations':['Same current initialization, three curriculum orders; not independent models or new8seedconfirmation.',
                'AGY diagnostic prompts already inspected; basic8intent style matching, not broad normal language.',
                'No privateweights/rawoutputs included in public artifacts or transmitted to teacher.']}
        (run/'report.md').write_text('# G1 basic rehearsal product acceptance\n\nReal existing local training on private clones; no automatic serving replacement.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-basic-rehearsal-latest.json',result)
        print(json.dumps({'run':str(run),'decision':result['decision'],'deployed':False}),flush=True)


if __name__=='__main__':main()
