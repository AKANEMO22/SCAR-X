"""NumPy-only coordinator; one isolated Torch control/cold child at a time."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,subprocess,ast,zipfile
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from visual_word_control_common import source_data,SEEDS,SOURCE,atomic
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval
from scarx.runtime.experiment_status import progress
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import activity


def child(job,run,monitor,deadline,completed):
    mode=job['mode'];seed=job['seed'];path=run/f'job-visual-{seed}-{mode}.json';atomic(path,job)
    out=(run/f'{seed}-{mode}.stdout.log').open('wb');err=(run/f'{seed}-{mode}.stderr.log').open('wb')
    process=subprocess.Popen([sys.executable,str(root/'scripts/visual_word_control_worker.py'),'--job',str(path)],cwd=root,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);signature=None
    try:
        while process.poll() is None:
            monitor.check()
            if time.perf_counter()>deadline:raise TimeoutError('Registered visualcontrol deadline')
            measured=run/'visual-control-progress.json'
            if mode=='train' and measured.is_file() and measured.stat().st_size<65536:
                value=json.loads(measured.read_text(encoding='utf-8'))
                if value.get('seed')==seed and value.get('responses')!=signature:
                    signature=value['responses'];progress(run,f'CONTROL seed{seed} · {signature}/1536',completed,3,metrics={k:value[k] for k in ('seed','responses','loss_mean','elapsed_seconds','backward_passes')},neural=value['neural'])
            time.sleep(.25)
        if process.returncode:raise RuntimeError(f'Visualcontrol {seed}/{mode} failed; retained actual stdout/stderr in run')
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        out.close();err.close()


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();preflight=json.loads((root/'artifacts/visual-word-control-preflight.json').read_text(encoding='utf-8'))
    assert preflight['formal_training_permitted'] and preflight['all_matched']
    manifest=json.loads((root/preflight['run']/'manifest.json').read_text(encoding='utf-8'))
    for name in ('src/scarx/evaluation/controls.py','src/scarx/evaluation/visual_words.py','src/scarx/evaluation/readout_conditioning.py','src/scarx/neural/core.py','src/scarx/neural/input_events.py','experiments/visual-word-learnability-control-v1.md'):
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest['source_sha256'][name],name
    # Only atomic-file plumbing was added after preflight. Verify all consumed
    # data/calibration/sampler functions against its immutable source snapshot.
    helper='scripts/visual_word_control_common.py'
    with zipfile.ZipFile(root/preflight['run']/'source.zip') as z:old_helper=z.read(helper).decode('utf-8')
    def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    before=functions(old_helper);after=functions((root/helper).read_text(encoding='utf-8'))
    for name in ('source_data','calibrated','sample_stream','verify_baseline'):assert before[name]==after[name],name
    receipt,data,train,dev,images,vocab,baselines,files=source_data()
    run=new_run(root,'visual-word-learnability-control-v1',{'protocol':'experiments/visual-word-learnability-control-v1.md',
        'display_title':'ISOLATED ALIF-BPTT CONTROL · image/question→neuralwords',
        'model_scope':'CONTROL ONLY; SAME48ALIF/input/localwordhead, surrogateBPTT hidden. Main/proposal30 untouched.',
        'control_only':True,'model_seeds':SEEDS,'experiences':1536,'torch_children_at_once':1,'preflight_run':preflight['run'],
        'dataset':receipt['data'],'dataset_sha256':receipt['sha256']},data_description='Owned192PNG/AGY32wordings, samepairedsource32data/budget; no private/pretrainedweights.',data_files=files)
    (run/'report.md').write_text('# Isolated visualword ALIF-BPTT CONTROL\n\nActual training in progress. NOT main/locallearning, no deploy or qualityclaim.\n',encoding='utf-8')
    activity(run,'Bắt đầu3đối chứng visualALIF đãđăngký','Cùng48ALIF/input/head/1536experiences; hidden BPTT chỉ côlập. Preflight3×64parity qua, forecast317s; không mainchange/proposal30approval.',status='started',evidence={'protocol':'experiments/visual-word-learnability-control-v1.md','preflight':preflight['run'],'trials':3})
    records=[]
    with ResourceMonitor(interval=.1) as monitor:
        for seed in SEEDS:
            trial_began=time.perf_counter();deadline=min(began+6000,trial_began+1500)
            progress(run,f'CONTROL seed{seed} initializing',len(records),3)
            child({'mode':'train','seed':seed},run,monitor,deadline,len(records));value=json.loads((run/f'control-visual-{seed}.json').read_text(encoding='utf-8'));row=value['record']
            assert row['experiences']==row['optimizer_steps']==1536 and row['backward_passes']==1536
            artifact=root/row['checkpoint'];assert hashlib.sha256(artifact.read_bytes()).hexdigest()==row['sha256']
            progress(run,f'CONTROL seed{seed} freshprocess continuation',len(records),3)
            child({'mode':'cold','seed':seed,'checkpoint':row['checkpoint'],'sha256':row['sha256']},run,monitor,deadline,len(records))
            assert json.loads((run/f'cold-visual-{seed}.json').read_text(encoding='utf-8'))==row.pop('cold_expected')
            assert hashlib.sha256(artifact.read_bytes()).hexdigest()==row['sha256'] and time.perf_counter()<deadline
            row['cold_exact']=True;row['seconds_including_cold']=time.perf_counter()-trial_began;records.append(row);atomic(run/'trials.json',records)
            progress(run,f'CONTROL seed{seed} final/coldexact',len(records),3,metrics={'dev_macro':row['summary']['development']['macro_exact'],'train_macro':row['summary']['train864']['macro_exact'],'train_seconds':row['train_seconds']})
            print(json.dumps({'seed':seed,'train_macro':row['summary']['train864']['macro_exact'],'dev_macro':row['summary']['development']['macro_exact'],'seconds':row['seconds_including_cold']}),flush=True)
        def average(values):
            return {'macro_exact':float(np.mean([r['macro_exact'] for r in values])),'novel_composition_macro':float(np.mean([r['novel_composition_macro'] for r in values])),
                'by_kind':{k:float(np.mean([r['by_kind'][k] for r in values])) for k in ('single_color','single_shape','object_count','left_color')}}
        summary={'bptt_control':average([r['summary']['development'] for r in records])}
        for condition in ('full','no_image','no_question'):summary['frozen_'+condition]=average([r['summary']['development'] for r in baselines if r['condition']==condition])
        gains=[r['summary']['development']['macro_exact']-next(b['summary']['development']['macro_exact'] for b in baselines if b['seed']==r['seed'] and b['condition']=='full') for r in records]
        stats=interval(gains);causal={k:interval([r['summary']['development']['macro_exact']-r['causal'][k]['macro_exact'] for r in records]) for k in ('blank_image','blank_question','swapped_image')}
        gates={'dev65_eachseed':all(r['summary']['development']['macro_exact']>=.65 for r in records),'kind50_eachseed':all(all(v>=.5 for v in r['summary']['development']['by_kind'].values()) for r in records),
            'novel60_eachseed':all(r['summary']['development']['novel_composition_macro']>=.6 for r in records),'gain20_allpositive_lowerCIpositive':stats['mean']>=.2 and all(v>0 for v in gains) and stats['lower95']>0,
            'each_causal_mean_drop20':all(v['mean']>=.2 for v in causal.values())}
        interpretation='CONTROL_JOINT_LEARNABILITY_SUPPORTED' if all(gates.values()) else 'CONTROL_FITS_TRAIN_GENERALIZATION_UNRESOLVED' if all(r['summary']['train864']['macro_exact']>=.65 for r in records) else 'CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET'
        result={'run':run.relative_to(root).as_posix(),'source_run':SOURCE,'preflight_run':preflight['run'],'decision':'ISOLATED_VISUAL_WORD_CONTROL_ONLY','interpretation':interpretation,
            'control_only':True,'summary':summary,'paired_trained_control_gains':{'frozen_full':stats},'paired_causal_drops':causal,'gates':gates,'trials':records,
            'dataset':receipt['data'],'dataset_sha256':receipt['sha256'],'reused_frozen_trials':9,'all_cold_exact':True,'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-began,
            'deployed':False,'weights_imported_into_scarx':False,'main_local_learning_success':False,'normal_photo_qa_proven':False,'retention_confirmed':False,'proposal30_approved':False,
            'limits':['Threepairedseeds on inspectedsyntheticdev; not freshfinaltest or normalvision/chat/lifelong.',
                'BiasedGLOBAL/TEMPORAL surrogateBPTT onlycontrol; headlocal unchanged. Extra graphcompute/allrowsAdam not matchedcost.',
                'Frozenmissingmodality baselines are reused context, not matched BPTTmissingmodality ablations. Mainstilllocal/ALIF, STOP32retained.']}
        (run/'report.md').write_text('# ISOLATED visual ALIF-BPTT learnability CONTROL\n\n'+interpretation+'\n\nNo controlweights/algorithm imported into SCAR-X main.\n\n'+json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/visual-word-control-latest.json',result)
        activity(run,'Hoàn tất3đối chứng ALIF visualword',interpretation+'; giữ nguyên STOP32/mainlocal/proposal30pending. Complete864devoutputs/caseerrors đãlưu.',evidence={'control_only':True,'interpretation':interpretation,'all_cold_exact':True,'gates':gates})
        print(json.dumps({'run':result['run'],'interpretation':interpretation,'summary':summary,'resources':result['resources']},ensure_ascii=False))


if __name__=='__main__':main()
