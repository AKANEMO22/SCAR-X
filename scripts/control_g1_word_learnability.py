"""NumPy-only coordinator: ONE Torch child at a time, no deployed control."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,itertools,subprocess,hashlib,argparse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from train_g1_word_score_inputs import source_data,SEEDS,METRICS
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval
from scarx.runtime.experiment_status import progress
from scarx.runtime.experiment_status import RUN_ID,_process
from scarx.runtime.work_trace import activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

ENCODERS=('legacy','known-digest128-v1')


def run_child(command,run,monitor,deadline,stage,completed):
    name=stage.replace('/','-');stdout=(run/(name+'.stdout.log')).open('wb');stderr=(run/(name+'.stderr.log')).open('wb')
    process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    signature=None
    try:
        while process.poll() is None:
            monitor.check()
            if time.perf_counter()>deadline:raise TimeoutError('Registered control deadline')
            path=run/'worker-progress.json'
            if stage.startswith('control-') and path.exists() and path.stat().st_size<100000:
                measured=json.loads(path.read_text(encoding='utf-8'));s=(measured.get('seed'),measured.get('input_encoding'),measured.get('responses'))
                if s!=signature:
                    signature=s;progress(run,stage+' · responses'+str(measured['responses'])+'/1024',completed,6,
                        metrics={k:measured[k] for k in ('seed','input_encoding','responses','loss_mean','elapsed_seconds','backward_passes')},neural=measured.get('neural'))
            time.sleep(.5)
        if process.returncode:raise RuntimeError(stage+' failed; actual stdout/stderr retained in run')
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        stdout.close();stderr.close()


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--previous-io-failure');args=parser.parse_args();retry=None;consumed=0.
    if args.previous_io_failure:
        receipt=Path(args.previous_io_failure).resolve()
        if not receipt.is_relative_to(root/'artifacts') or receipt.stat().st_size>65536:raise ValueError('Invalid IO failure receipt')
        retry=json.loads(receipt.read_text(encoding='utf-8'))
        if retry.get('failure_kind')!='windows_atomic_replace_sharing' or retry.get('quality_evaluated') is not False or retry.get('technical_retry_count')!=1:raise ValueError('Not a registered one-off IO retry')
        consumed=retry['budget_consumed_seconds']
        if type(consumed) not in (int,float) or not 0<consumed<900:raise ValueError('Invalid prior budget')
        previous=retry.get('source_run','')
        if not previous.startswith('runs/') or not RUN_ID.fullmatch(previous[5:]):raise ValueError('Invalid prior run')
        journal=json.loads((root/'local/experiment-status'/(previous[5:]+'.json')).read_text(encoding='utf-8'))
        if journal['status']!='stopped_without_result' or _process(journal)[0]:raise ValueError('Previous process is not verified terminal')
        if abs(journal['updated']-journal['started']-consumed)>.01 or (root/previous/'trials.json').exists():raise ValueError('Unexpected prior quality results or budget')
        manifest=json.loads((root/previous/'manifest.json').read_text(encoding='utf-8'))
        for name in ('src/scarx/evaluation/controls.py','src/scarx/evaluation/score_word.py','src/scarx/neural/core.py','src/scarx/neural/unified.py','src/scarx/neural/paired_score.py','experiments/g1-word-learnability-control-v1.md'):
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=manifest['source_sha256'][name]:raise ValueError('Technical retry changed neural math/protocol')
        # Count calendar repair time too, conservatively preserving the original
        # round/firsttrial caps instead of silently resetting either deadline.
        consumed=time.time()-journal['started']
        if not 0<consumed<900:raise TimeoutError('Original firsttrial budget elapsed during IO repair')
    cap=apply_memory_limit();began=time.perf_counter()-consumed;preflight=json.loads((root/'artifacts/g1-word-control-preflight.json').read_text(encoding='utf-8'))
    if not preflight.get('formal_training_permitted') or not preflight.get('all_matched'):raise ValueError('Forward/timing preflight required')
    source_info=json.loads((root/'artifacts/g1-word-score-inputs-latest.json').read_text(encoding='utf-8'));source=root/source_info['run']
    old=json.loads((source/'trials.json').read_text(encoding='utf-8'));train,_,_,_,sources=source_data()
    run=new_run(root,'g1-word-learnability-control-v1',{'protocol':'experiments/g1-word-learnability-control-v1.md',
        'display_title':'ISOLATED BPTT CONTROL · ALIF word learnability',
        'model_scope':'CONTROL ONLY: surrogate-BPTT credit vs reusedfrozen/local references; NOT SCAR-X/Hermes main.',
        'control_only':True,'model_seeds':SEEDS,'encoders':ENCODERS,'responses':1024,'torch_children_at_once':1,
        'forecast_seconds':preflight['forecast_round_seconds'],'preflight_run':preflight['run'],'technical_retry':retry,
        'prior_attempt_and_repair_seconds_counted_against_original_budget':consumed},
        data_description='Same public181train/31dev/48prior and pairedseeds as v2; frozen/local results reused, no private models/data.',data_files=sources)
    (run/'report.md').write_text('# ISOLATED ALIF word learnability control\n\nActual controltraining in progress. NOT mainlocallearning, no deployment or qualityclaim.\n',encoding='utf-8')
    activity(run,'Bắt đầu đối chứng learnability cô lập','SáuBPTTcontrols, cùngforward/head/optimizer/blocktiming; một Torchchild tại một thời điểm. Baselinefrozen/local dùng số đo thật v2 đã kiểmparity.',status='started',evidence={'control_only':True,'formal_trials':6,'forecast_seconds':preflight['forecast_round_seconds']})
    records=[];cold='''import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from scarx.runtime.limits import apply_memory_limit
apply_memory_limit()
from scarx.evaluation.controls import WordLearnabilityControl,torch
torch.set_num_threads(1)
m=WordLearnabilityControl.loads(Path(sys.argv[2]).read_bytes());learn=m.learn(sys.argv[3],sys.argv[4]);reply=m.generate(sys.argv[3])
Path(sys.argv[5]).write_text(json.dumps({'learn':learn,'reply':reply,'final_sha256':hashlib.sha256(m.dumps()).hexdigest()}),encoding='utf-8')
'''
    with ResourceMonitor(interval=.1) as monitor:
        for seed,encoding in itertools.product(SEEDS,ENCODERS):
            if time.perf_counter()-began>6000:raise TimeoutError('Registered6000s round limit')
            job=run/f'job-{seed}-{encoding}.json';write_json(job,{'seed':seed,'input_encoding':encoding})
            # Old progress remains timestamped history; remove nothing, ignore stale
            # fields by resetting the file atomically before the new owned child.
            write_json(run/'worker-progress.json',{'seed':seed,'input_encoding':encoding,'responses':0,'loss_mean':None,
                'elapsed_seconds':0.,'backward_passes':0,'neural':None})
            progress(run,f'CONTROL {seed}/{encoding} · initializing',len(records),6)
            prior_for_trial=consumed if not records else 0.
            deadline=min(began+6000,time.perf_counter()+900-prior_for_trial)
            run_child([sys.executable,str(root/'scripts/word_control_worker.py'),'--job',str(job)],run,monitor,deadline,f'control-{seed}-{encoding}',len(records))
            data=json.loads((run/f'control-{seed}-{encoding}.json').read_text(encoding='utf-8'));row=data['record'];path=root/row['checkpoint'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==row['sha256'] and row['responses']==1024 and row['seconds']+prior_for_trial<900
            row['effective_trial_seconds_including_previous_attempt_and_repair']=row['seconds']+prior_for_trial
            cold_output=run/f'cold-{seed}-{encoding}.json'
            progress(run,f'CONTROL {seed}/{encoding} · kiểm tra process mới',len(records),6)
            run_child([sys.executable,'-c',cold,str(root/'src'),str(path),train[0]['prompt'],train[0]['target'],str(cold_output)],run,monitor,min(began+6000,time.perf_counter()+60),f'cold-{seed}-{encoding}',len(records))
            assert json.loads(cold_output.read_text(encoding='utf-8'))==row.pop('cold_expected') and digest==hashlib.sha256(path.read_bytes()).hexdigest()
            row['cold_exact']=True;records.append(row);write_json(run/'trials.json',records)
            progress(run,f'CONTROL {seed}/{encoding} · final+ coldexact',len(records),6,metrics={'dev_normalized':row['summary']['development']['normalized_style_exact'],'train_seconds':row['train_seconds']})
            print(json.dumps({k:row[k] for k in ('seed','input_encoding','summary','train_seconds','backward_passes')}),flush=True)
        frozen=[r for r in old if r['input_encoding'] in ENCODERS and r['condition']=='frozen'];local=[r for r in old if r['input_encoding'] in ENCODERS and r['condition']=='full']
        assert len(frozen)==len(local)==6
        summary={};statistics={};interpretations={}
        for encoding in ENCODERS:
            summary[encoding]={}
            for method,rows in [('full',records),('frozen',frozen),('local_reference',local)]:
                subset=[r for r in rows if r['input_encoding']==encoding]
                summary[encoding][method]={split:{k:float(np.mean([r['summary'][split][k] for r in subset])) for k in METRICS} for split in ('train16','development','prior')}
            def get(rows,seed,split):return next(r['summary'][split]['normalized_style_exact'] for r in rows if r['seed']==seed and r['input_encoding']==encoding)
            gain=[get(records,s,'development')-get(frozen,s,'development') for s in SEEDS]
            statistics[encoding]={'vs_frozen':{split:interval([get(records,s,split)-get(frozen,s,split) for s in SEEDS]) for split in ('train16','development','prior')},
                'vs_local_full':{split:interval([get(records,s,split)-get(local,s,split) for s in SEEDS]) for split in ('train16','development','prior')}}
            train_score=summary[encoding]['full']['train16']['normalized_style_exact'];dev_score=summary[encoding]['full']['development']['normalized_style_exact']
            interpretations[encoding]=('CONTROL_LIMITED_LEARNABILITY_MAIN_UNPROVEN' if train_score>=.8 and dev_score>=.4 and np.mean(gain)>=.1 and all(v>0 for v in gain)
                else 'CONTROL_FITS_TRAIN_GENERALIZATION_UNRESOLVED' if train_score>=.8 else 'CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET')
        result={'run':run.relative_to(root).as_posix(),'source_run':source_info['run'],'preflight_run':preflight['run'],
            'decision':'ISOLATED_WORD_LEARNABILITY_CONTROL_ONLY','control_only':True,'summary':summary,'paired_statistics':statistics,'interpretations':interpretations,
            'trials':records,'reused_frozen_trials':6,'reused_local_trials':6,'all_cold_exact':True,'resources':monitor.report(),'cap':cap,
            'technical_retry':retry,'total_program_wall_seconds_including_failed_attempt':time.perf_counter()-began,
            'deployed':False,'weights_imported_into_scarx':False,'main_local_learning_success':False,'normal_chat_or_vision_proven':False,
            'limits':['Threepairedseeds, known8styles and inspected dev/prior; not freshfinaltest or broadlanguage/vision.',
                'SurrogateBPTT is biased, global/temporal and isolated. Extra forward/backward compute, not matchedhardwarecost.',
                'Mainword-score route remains STOP; controlweights/algorithm cannot replace locallearning core.']}
        (run/'report.md').write_text('# ISOLATED ALIF word learnability control\n\nSurrogate/globaltemporal control only, never mainSCAR-X. Reused actualparitychecked frozen/local baselines.\n\n'+json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-word-learnability-control-latest.json',result)
        activity(run,'Hoàn tất đối chứng ALIF cô lập','Đã giữ mọi số đo và coldresume; không dùng controlweights/backward trong mainSCAR-X.',evidence={'interpretations':interpretations,'all_cold_exact':True,'control_only':True})
        print(json.dumps({'run':result['run'],'interpretations':interpretations,'summary':summary},ensure_ascii=False))


if __name__=='__main__':main()
