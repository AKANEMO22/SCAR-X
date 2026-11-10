"""Fixed15trial smallwordpilot; approvedmechanism, no change to maincore."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,itertools,subprocess,hashlib
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from direct_words_common import source_data,SEEDS,METHODS
from direct_word_worker import trial
from visual_word_control_common import atomic
from scarx.runtime.atomic_file import read_text_with_retry
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def child(job,run,monitor,deadline):
    seed,method,_=job['trial'];mode=job['mode'];path=run/f'job-word-{seed}-{method}-{mode}.json';atomic(path,job);signature=None;began=time.perf_counter()
    with (run/f'{seed}-{method}-{mode}.stdout.log').open('wb') as out,(run/f'{seed}-{method}-{mode}.stderr.log').open('wb') as err:
        p=subprocess.Popen([sys.executable,str(root/'scripts/direct_word_worker.py'),'--job',str(path)],cwd=root,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            while p.poll() is None:
                monitor.check()
                if time.perf_counter()>deadline:raise TimeoutError('Registeredwordchild deadline')
                beat=run/f'heartbeat-word-{seed}-{method}.json'
                if mode=='train' and beat.exists():
                    r=json.loads(read_text_with_retry(beat))
                    if r['responses']!=signature:signature=r['responses'];progress(run,f'WORD {seed}/{method} {signature}/1024',metrics={k:r[k] for k in ('seed','method','responses','loss_mean','hidden_updates','seconds')},neural=r['neural'])
                time.sleep(.25)
            if p.returncode:raise RuntimeError('Wordchildfailed; actual logs retained')
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait()
    return time.perf_counter()-began


def cold(row,run,monitor,deadline):
    checkpoint=root/row['checkpoint'];assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['sha256']
    elapsed=child({'mode':'cold','trial':(row['seed'],row['method'],str(run)),'checkpoint':row['checkpoint'],'sha256':row['sha256']},run,monitor,min(deadline,time.perf_counter()+300-row['seconds']))
    assert json.loads(read_text_with_retry(run/f"cold-word-{row['seed']}-{row['method']}.json"))==row.pop('cold_expected')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['sha256']
    row['cold_exact']=True;row['seconds_including_cold']=row['seconds']+elapsed;assert row['seconds_including_cold']<300


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();receipt,train,dev,vocab,files=source_data();pre=json.loads((root/'artifacts/direct-word-preflight.json').read_text(encoding='utf-8'));assert pre['formal_training_permitted']
    manifest=json.loads((root/pre['run']/'manifest.json').read_text(encoding='utf-8'))
    for p in ('src/scarx/evaluation/direct_words.py','src/scarx/evaluation/readout_eligibility.py','src/scarx/evaluation/controls.py','src/scarx/evaluation/readout_conditioning.py','src/scarx/neural/core.py','src/scarx/neural/unified.py','scripts/direct_words_common.py','experiments/direct-word-bridge-v1.md',receipt['data']):assert hashlib.sha256((root/p).read_bytes()).hexdigest()==manifest['source_sha256'][p],p
    run=new_run(root,'direct-word-bridge-v1',{'protocol':'experiments/direct-word-bridge-v1.md','display_title':'Proposal30 WORD · actualAGYdata/15neuraltrials',
        'model_scope':'Research48ALIF/tokenhead, approveddirectfeedback; no mainweights/Transformer/proxy/P1waiver. Small8stylepilot, not normalchat.',
        'proposal30_approved':True,'seeds':SEEDS,'methods':METHODS,'responses':1024,'dataset':receipt['data'],'dataset_sha256':receipt['sha256'],'train':64,'dev':64,'preflight_run':pre['run']},
        data_description='ActualAGY128prompts reviewed plusonevalidlanguagecorrection, ownedstylelabels,64train64dev. No private/teacher/Qwenanswers imported.',data_files=files)
    (run/'report.md').write_text('# Wordbridgepilot\n\nActualtraining inprogress, fixedcriteria. Not generalchat/main/vision or P1success.\n',encoding='utf-8')
    activity(run,'Bắt đầu15trial học sinh từ thật','3freshseed×5method,1024TRAINresponses, actualAGY64/64reviewed, unchangedeligibility35. Teacher/Qwenanswers không lànhãn. Không maincorechange.',status='started',evidence={'protocol':'experiments/direct-word-bridge-v1.md','preflight':pre['run'],'formal_trials':15})
    rows=[];deadline=began+2400;todo=iter([(s,c,str(run)) for s,c in itertools.product(SEEDS,METHODS[:-1])]);pending={}
    with ResourceMonitor(interval=.1) as monitor:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for _ in range(2):job=next(todo);pending[pool.submit(trial,job)]=job
            while pending:
                done,_=wait(pending,timeout=.25,return_when=FIRST_COMPLETED);monitor.check()
                if time.perf_counter()>deadline:raise TimeoutError('Registered2400s wordround')
                for future in done:
                    row,neural=future.result();pending.pop(future);cold(row,run,monitor,deadline);rows.append(row);atomic(run/'trials.json',rows)
                    progress(run,f"WORD {row['seed']}/{row['method']} done/coldexact",len(rows),15,metrics={'dev_macro':row['summary']['development']['macro_exact'],'probe_gain':row['probe_gain'],'train_seconds':row['train_seconds']},neural=neural)
                    print(json.dumps({'seed':row['seed'],'method':row['method'],'dev':row['summary']['development']['macro_exact'],'probe_gain':row['probe_gain']}),flush=True)
                    try:job=next(todo);pending[pool.submit(trial,job)]=job
                    except StopIteration:pass
        for seed in SEEDS:
            child({'mode':'train','trial':(seed,'bptt_control',str(run))},run,monitor,min(deadline,time.perf_counter()+300));row=json.loads(read_text_with_retry(run/f'word-{seed}-bptt_control.json'))['record'];cold(row,run,monitor,deadline);rows.append(row);atomic(run/'trials.json',rows)
            progress(run,f'WORD {seed}/BPTTcontrol done/coldexact',len(rows),15,metrics={'dev_macro':row['summary']['development']['macro_exact'],'probe_gain':row['probe_gain']})
        for seed in SEEDS:
            selected=[r for r in rows if r['seed']==seed];assert len({json.dumps(r['stratum_counts'],sort_keys=True) for r in selected})==1 and len({r['word_updates'] for r in selected})==1 and len({r['training_ticks'] for r in selected})==1
            zero=next(r for r in selected if r['method']=='zero');frozen=next(r for r in selected if r['method']=='frozen');assert zero['core_fingerprint']==frozen['core_fingerprint'] and zero['head_sha256']==frozen['head_sha256']
        def score(seed,method):return next(r for r in rows if r['seed']==seed and r['method']==method)
        full=[score(s,'correct') for s in SEEDS];gains=[score(s,'correct')['summary']['development']['macro_exact']-score(s,'frozen')['summary']['development']['macro_exact'] for s in SEEDS]
        gain=interval(gains);probe=interval([r['probe_gain'] for r in full]);gates={'macro65_eachseed':all(r['summary']['development']['macro_exact']>=.65 for r in full),
            'family50_eachseed':all(all(v>=.5 for v in r['summary']['development']['by_family'].values()) for r in full),'gain10_allpositive_lower95':gain['mean']>=.1 and all(x>0 for x in gains) and gain['lower95']>0,
            'resethead_probe_gain5_lower95':probe['mean']>=.05 and probe['lower95']>0}
        summary={c:{'macro_development':float(np.mean([score(s,c)['summary']['development']['macro_exact'] for s in SEEDS])),
            'macro_train':float(np.mean([score(s,c)['summary']['train']['macro_exact'] for s in SEEDS])),
            'probe_gain':float(np.mean([score(s,c)['probe_gain'] for s in SEEDS])),
            'restore_initial_hidden_dev':float(np.mean([score(s,c)['summary']['restore_initial_hidden']['macro_exact'] for s in SEEDS]))} for c in METHODS}
        result={'run':run.relative_to(root).as_posix(),'decision':'WORD_BRIDGE_FRESH_VALIDATION_REQUIRED' if all(gates.values()) else 'STOP_DIRECT_WORD_BRIDGE_V1','summary':summary,'gates':gates,
            'paired_hidden_dev_gain':gain,'paired_resethead_probe_gain':probe,'trials':rows,'proposal30_approved':True,'all_cold_exact':True,'zero_frozen_exact':True,
            'dataset':receipt['data'],'dataset_sha256':receipt['sha256'],'deployed':False,'main_weights_changed':False,'normal_chat_or_vision_proven':False,'retention_proven':False,
            'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-began,'preflight_run':pre['run'],
            'limits':['Smallclosed8style/no multi-turn/generalchat/vision/lifelong;3pairedseeds/inspectedstyles/3historicaloverlaps.',
                'Qwen64textreference separatecost/notlabels; no mainimport/Transformer/proxy.',
                'P1/P2failedgates unchanged. No automatic tuning/data/encoder/LR/epoch round after pilotfailure.']}
        (run/'report.md').write_text('# Directwordbridgepilot\n\n'+result['decision']+'\n\nNeuralfreewordoutputs, no targetlookup. No main/P1/P2/generalchatclaim.\n\n'+json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/direct-word-bridge-latest.json',result)
        activity(run,'Hoàn tất15trial wordbridge',result['decision']+'; giữcriteria/P1/P2, không mainimport hoặc gọi closedstyles là normalchat.',evidence={'summary':summary,'gates':gates,'all_cold_exact':True})
        print(json.dumps({'run':result['run'],'decision':result['decision'],'summary':summary,'gates':gates,'resources':result['resources']},ensure_ascii=False))


if __name__=='__main__':main()
