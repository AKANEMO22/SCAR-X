"""Registered21trial37screen; source32forward, approved30localeligibility."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,itertools,hashlib,subprocess
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from direct_visual_common import source_data,SEEDS,METHODS
from direct_visual_worker import trial
from visual_word_control_common import atomic
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval
from scarx.runtime.atomic_file import read_text_with_retry
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import activity
from scarx.runtime.experiment_status import progress


def child(job,run,monitor,deadline):
    seed,method,_=job['trial'];mode=job['mode'];path=run/f'job-visual37-{seed}-{method}-{mode}.json';atomic(path,job);began=time.perf_counter();signature=None
    with (run/f'{seed}-{method}-{mode}.stdout.log').open('wb') as out,(run/f'{seed}-{method}-{mode}.stderr.log').open('wb') as err:
        p=subprocess.Popen([sys.executable,str(root/'scripts/direct_visual_worker.py'),'--job',str(path)],cwd=root,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            while p.poll() is None:
                monitor.check()
                if time.perf_counter()>deadline:raise TimeoutError('Registered37childdeadline')
                beat=run/f'heartbeat-visual37-{seed}-{method}.json'
                if mode=='train' and beat.is_file():
                    r=json.loads(read_text_with_retry(beat))
                    if r['experiences']!=signature:
                        signature=r['experiences'];progress(run,f'VISUAL37 {seed}/{method} {signature}/1536',metrics={k:r[k] for k in ('seed','method','experiences','loss_mean','hidden_updates','seconds')},neural=r['neural'])
                time.sleep(.25)
            if p.returncode:raise RuntimeError('37childfailed, actual logs retained')
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait()
    return time.perf_counter()-began


def cold(row,run,monitor,deadline):
    checkpoint=root/row['checkpoint'];assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['sha256']
    seconds=child({'mode':'cold','trial':(row['seed'],row['method'],str(run)),'checkpoint':row['checkpoint'],'sha256':row['sha256']},run,monitor,min(deadline,time.perf_counter()+300-row['seconds']))
    assert json.loads(read_text_with_retry(run/f"cold-visual37-{row['seed']}-{row['method']}.json"))==row.pop('cold_expected') and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['sha256']
    row['cold_exact']=True;row['seconds_including_cold']=row['seconds']+seconds;assert row['seconds_including_cold']<300


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();receipt,data,train,dev,images,vocab,_,files=source_data();pre=json.loads((root/'artifacts/direct-visual-preflight.json').read_text(encoding='utf-8'));assert pre['formal_training_permitted']
    manifest=json.loads((root/pre['run']/'manifest.json').read_text(encoding='utf-8'))
    for p in ('src/scarx/evaluation/direct_visual.py','src/scarx/evaluation/readout_eligibility.py','src/scarx/evaluation/visual_words.py','src/scarx/evaluation/controls.py','src/scarx/neural/core.py','src/scarx/evaluation/readout_conditioning.py','src/scarx/neural/input_events.py','scripts/direct_visual_common.py','experiments/direct-visual-bridge-v1.md',receipt['data']):assert hashlib.sha256((root/p).read_bytes()).hexdigest()==manifest['source_sha256'][p],p
    run=new_run(root,'direct-visual-bridge-v1',{'protocol':'experiments/direct-visual-bridge-v1.md','display_title':'DIRECT VISUAL37 ·21trials/same32RGBforward',
        'model_scope':'Approved30one-stepfeedback/localcacheeligibility, SAME48ALIF/32inputs. Researchonly, no normalvision/main/Transformer.',
        'proposal30_approved':True,'seeds':SEEDS,'methods':METHODS,'experiences':1536,'dataset':receipt['data'],'dataset_sha256':receipt['sha256'],'train_cases':864,'dev_cases':864,'preflight_run':pre['run']},
        data_description='EXACT32owned192PNG/actualAGY32questionwordings, rendererlabels neverinput, no private or pretrainedweights.',data_files=files)
    (run/'report.md').write_text('# Directvisual37\n\nActual21trials inprogress. Same32forward, newapprovedlocalcredit/cachetrace. No quality/mainclaim.\n',encoding='utf-8')
    activity(run,'Bắt đầu21trial37 học ảnh/câu hỏi thật','P0same32forward21parities/cachefinite-diff64/partialimagecodec qua.3freshseed×7condition×1536exp, no mainchange/Transformer/labeloracle.',status='started',evidence={'protocol':'experiments/direct-visual-bridge-v1.md','preflight':pre['run'],'expected_trials':21})
    rows=[];deadline=began+2400;todo=iter([(s,c,str(run)) for s,c in itertools.product(SEEDS,METHODS[:-1])]);pending={}
    with ResourceMonitor(interval=.1) as monitor:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for _ in range(2):job=next(todo);pending[pool.submit(trial,job)]=job
            while pending:
                ready,_=wait(pending,timeout=.25,return_when=FIRST_COMPLETED);monitor.check()
                if time.perf_counter()>deadline:raise TimeoutError('Registered37round2400s')
                for future in ready:
                    row,neural=future.result();pending.pop(future);cold(row,run,monitor,deadline);rows.append(row);atomic(run/'trials.json',rows)
                    progress(run,f"VISUAL37 {row['seed']}/{row['method']} done/coldexact",len(rows),21,metrics={'dev_macro':row['summary']['development']['macro_exact'],'probe_gain':row['probe_gain'],'train_seconds':row['train_seconds']},neural=neural)
                    print(json.dumps({'seed':row['seed'],'method':row['method'],'dev':row['summary']['development']['macro_exact'],'probe_gain':row['probe_gain']}),flush=True)
                    try:job=next(todo);pending[pool.submit(trial,job)]=job
                    except StopIteration:pass
        for seed in SEEDS:
            child({'mode':'train','trial':(seed,'bptt_control',str(run))},run,monitor,min(deadline,time.perf_counter()+300));row=json.loads(read_text_with_retry(run/f'visual37-{seed}-bptt_control.json'))['record'];cold(row,run,monitor,deadline);rows.append(row);atomic(run/'trials.json',rows)
            progress(run,f'VISUAL37 {seed}/BPTTcontrol done/coldexact',len(rows),21,metrics={'dev_macro':row['summary']['development']['macro_exact'],'probe_gain':row['probe_gain']})
        def get(seed,method):return next(r for r in rows if r['seed']==seed and r['method']==method)
        for seed in SEEDS:
            group=[r for r in rows if r['seed']==seed];assert len({json.dumps(r['stratum_counts'],sort_keys=True) for r in group})==len({r['word_updates'] for r in group})==len({r['training_ticks'] for r in group})==1
            zero=get(seed,'zero');frozen=get(seed,'frozen');assert zero['core_fingerprint']==frozen['core_fingerprint'] and zero['head_sha256']==frozen['head_sha256']
        full=[get(s,'correct') for s in SEEDS];gains={c:interval([get(s,'correct')['summary']['development']['macro_exact']-get(s,c)['summary']['development']['macro_exact'] for s in SEEDS]) for c in ('frozen','shuffled','no_image','no_question')}
        drops={a:interval([r['summary']['development']['macro_exact']-r['causal'][a]['macro_exact'] for r in full]) for a in ('blank_image','blank_question','swapped_image')};probe=interval([r['probe_gain'] for r in full])
        gates={'macro65_eachseed':all(r['summary']['development']['macro_exact']>=.65 for r in full),'kind50_eachseed':all(all(v>=.5 for v in r['summary']['development']['by_kind'].values()) for r in full),
            'novel60_eachseed':all(r['summary']['development']['novel_composition_macro']>=.6 for r in full),
            'missingmodality_gains20_allpositive_lower95':all(gains[c]['mean']>=.2 and gains[c]['lower95']>0 and all(get(s,'correct')['summary']['development']['macro_exact']>get(s,c)['summary']['development']['macro_exact'] for s in SEEDS) for c in ('no_image','no_question')),
            'frozen_gain10_allpositive_lower95':gains['frozen']['mean']>=.1 and gains['frozen']['lower95']>0 and all(get(s,'correct')['summary']['development']['macro_exact']>get(s,'frozen')['summary']['development']['macro_exact'] for s in SEEDS),
            'causal_drop20_each':all(v['mean']>=.2 for v in drops.values()),'resethead_probe_gain5_lower95':probe['mean']>=.05 and probe['lower95']>0}
        summary={c:{'macro_exact':float(np.mean([get(s,c)['summary']['development']['macro_exact'] for s in SEEDS])),
            'macro_train':float(np.mean([get(s,c)['summary']['train']['macro_exact'] for s in SEEDS])),'novel_composition_macro':float(np.mean([get(s,c)['summary']['development']['novel_composition_macro'] for s in SEEDS])),
            'by_kind':{k:float(np.mean([get(s,c)['summary']['development']['by_kind'][k] for s in SEEDS])) for k in ('single_color','single_shape','object_count','left_color')},'probe_gain':float(np.mean([get(s,c)['probe_gain'] for s in SEEDS]))} for c in METHODS}
        result={'run':run.relative_to(root).as_posix(),'decision':'FRESH_DIRECT_VISUAL_VALIDATION_REQUIRED' if all(gates.values()) else 'STOP_DIRECT_VISUAL_BRIDGE_V1','summary':summary,'gates':gates,
            'paired_gains':gains,'paired_causal_drops':drops,'paired_resethead_probe_gain':probe,'trials':[{k:v for k,v in row.items() if k!='stratum_counts'} for row in rows],'all_cold_exact':True,'zero_frozen_exact':True,'proposal30_approved':True,
            'dataset':receipt['data'],'dataset_sha256':receipt['sha256'],'deployed':False,'main_weights_changed':False,'normal_photo_qa_proven':False,'normal_chat_or_vision_proven':False,'retention_proven':False,
            'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-began,'preflight_run':pre['run'],
            'limits':['3freshseeds on inspectedhistorical864devgeometriccases, not freshfinal/naturalphotos/OCR/lifelong.',
                'No teacher/Qwenanswers used; actualAGY32wordings reused, Qwen4Btext-only not visionbaseline.',
                'STOP32/33/35/36gates retained. No automaticencoder/cache/LR/epoch/modelscale/mainimport from results.']}
        (run/'report.md').write_text('# Directvisual37 research\n\n'+result['decision']+'\n\nActualneuralwords on same32forward; no main/P1/P2/normalvisionclaim.\n\n'+json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/direct-visual-bridge-latest.json',result)
        activity(run,'Hoàn tất21trial37',result['decision']+'; actualimage/wordoutputs andconditioncontrols retained. No changinggates/mainweights or normalvisionclaim.',evidence={'summary':summary,'gates':gates,'all_cold_exact':True})
        print(json.dumps({'run':result['run'],'decision':result['decision'],'summary':summary,'gates':gates,'resources':result['resources']},ensure_ascii=False))


if __name__=='__main__':main()
