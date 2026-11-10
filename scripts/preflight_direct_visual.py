"""37P0 pooledsensitivity/parity/fullstate and TRAIN-only timing, no devscore."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from direct_visual_common import source_data,sample_stream,SEEDS,METHODS,make,reference,load
from scarx.evaluation.visual_words import VisualWords
from scarx.evaluation.visual_eligibility_validation import check_pooled_sensitivities
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();receipt,data,train,_,images,vocab,_,files=source_data()
    run=new_run(root,'direct-visual-preflight-v1',{'protocol':'experiments/direct-visual-bridge-v1.md','display_title':'DIRECT VISUAL37 ·math/parity/timing',
        'model_scope':'No devqualityscoring, no mainchange. Cachedlocaleligibility for SAME32forward, approved30 research.','model_seeds':SEEDS,'timing_seed':37303},
        data_description='Pinnedowned192PNG/actualAGY32wordings, same32TRAINsampler. No newteacher/privateinputs.',data_files=files)
    (run/'report.md').write_text('# Directvisual37preflight\n\nMath/parity/timing inprogress, no devqualityselection.\n',encoding='utf-8')
    activity(run,'Bắt đầu37math/parity đãđăngký','5targetedtests qua, gồm cachedeligibility64finite-difference, midimage/wordcheckpoint/rollback/control. Không đổi mainALIF hoặc encoder/cache32.',status='started',evidence={'protocol':'experiments/direct-visual-bridge-v1.md','targeted_tests_passed':5})
    parity=[];timing={}
    with ResourceMonitor(interval=.1) as monitor:
        from scarx.evaluation.controls import torch
        torch.set_num_threads(1)
        for seed in SEEDS:
            for method in METHODS:
                # No hiddenupdates: validate exactforward/head for everyinputcondition.
                m=make(vocab,seed,method,train,images);r=reference(m)
                if method=='bptt_control':
                    from scarx.evaluation.controls import DirectVisualLearnabilityControl
                    r=make(vocab,seed,'frozen',train,images);m=DirectVisualLearnabilityControl(r)
                elif r.learner.condition not in ('zero','frozen'):r.learner.condition='zero'
                b=VisualWords(vocab,seed,method if method in ('no_image','no_question') else 'full');b.calibrate([{'prompt':x['prompt'],'image':images[x['scene_id']]} for x in train])
                if VisualWords.dumps(r)!=b.dumps():raise ValueError('37initialforward/head/calibration differs32')
                # Same sampler32 yields matching publictrainingexamples.
                cases=list(sample_stream(r,train,16));list(sample_stream(b,train,16))
                for case in cases:
                    assert m.learn(case['prompt'],images[case['scene_id']],case['target'])==b.learn(case['prompt'],images[case['scene_id']],case['target'])
                    if VisualWords.dumps(r)!=b.dumps():raise ValueError('37recordingonlyforward/headstate differs32')
                parity.append({'seed':seed,'method':method,'16responses_exact_visual32bytes':True})
                progress(run,f'Parity37 {len(parity)}/21',len(parity),24,metrics={'seed':seed,'method':method,'all_exact':True});monitor.check()
                if time.perf_counter()-began>180:raise TimeoutError('37preflight180s')
        m=make(vocab,37303,'correct',train,images);case=train[0];math=check_pooled_sensitivities(m,case['prompt'],images[case['scene_id']]);progress(run,'64cacheeligibilityfinite-difference matched',22,24,metrics=math)
        for method in ('correct','bptt_control'):
            start=time.perf_counter();m=make(vocab,37303,method,train,images);setup=time.perf_counter()-start;start=time.perf_counter()
            for case in sample_stream(reference(m),train,64):m.learn(case['prompt'],images[case['scene_id']],case['target'])
            elapsed=time.perf_counter()-start;payload=m.dumps();copy=load(payload,method);assert copy.dumps()==payload
            case=train[0];assert m.learn(case['prompt'],images[case['scene_id']],case['target'])==copy.learn(case['prompt'],images[case['scene_id']],case['target']) and m.dumps()==copy.dumps()
            assert m.generate(case['prompt'],images[case['scene_id']])==copy.generate(case['prompt'],images[case['scene_id']]) and m.dumps()==copy.dumps()
            forecast=1.25*(elapsed/64*1536+setup+40);timing[method]={'training_examples':64,'train_seconds':elapsed,'setup_seconds':setup,'forecast_trial_seconds':forecast,
                'array_bytes':reference(m).memory_bytes(),'checkpoint_bytes':len(payload),'nextlearn_generation_fullstate_exact':True}
            if method=='bptt_control':timing[method].update(max_graph_state_error=m.max_graph_state_error,max_graph_feature_error=m.max_graph_feature_error,max_graph_loss_error=m.max_graph_loss_error)
            progress(run,'Timing37 '+method,23 if method=='correct' else 24,24,metrics=timing[method]);monitor.check()
            if time.perf_counter()-began>180:raise TimeoutError('37preflight180s')
        forecast=18*timing['correct']['forecast_trial_seconds']+3*timing['bptt_control']['forecast_trial_seconds'];feasible=forecast<2400 and all(v['forecast_trial_seconds']<300 for v in timing.values())
        result={'run':run.relative_to(root).as_posix(),'decision':'DIRECT_VISUAL_PREFLIGHT_PASS' if feasible else 'STOP_DIRECT_VISUAL_BUDGET','all21_forwardparity_checks':parity,
            'mathematics':math,'timing':timing,'forecast_round_seconds':forecast,'formal_training_permitted':feasible,'no_development_quality_evaluated':True,
            'deployed':False,'main_weights_changed':False,'proposal30_approved':True,'resources':monitor.report(),'cap':cap}
        (run/'report.md').write_text('# Directvisual37preflight\n\nSame32forward/head, no devquality or mainchanges. Anchoredsurrogatefinite-difference is localmath, not hardspike/networkgradient.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/direct-visual-preflight.json',result)
        activity(run,'Hoàn tất37engineering/trainingforecast',result['decision']+'; same32forward/head,64cacheeligibilitymath, chưa qualityscore.',evidence={'forecast_round_seconds':forecast,'formal_training_permitted':feasible,'run':result['run']})
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
