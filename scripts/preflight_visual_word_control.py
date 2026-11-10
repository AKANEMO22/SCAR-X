"""Registered control parity and TRAIN-only timing, no devquality selection."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from visual_word_control_common import source_data,calibrated,sample_stream,verify_baseline,SEEDS,SOURCE
from scarx.evaluation.visual_words import VisualWords
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();receipt,data,train,dev,images,vocab,rows,files=source_data()
    run=new_run(root,'visual-word-control-preflight-v1',{'protocol':'experiments/visual-word-learnability-control-v1.md',
        'display_title':'CONTROL ONLY · ALIF visualwords parity/timing',
        'model_scope':'No devscores or mainlearning change; one isolated Torchprocess. Proposal30 pending.',
        'control_only':True,'paired_seeds':SEEDS,'timing_seed':34103,'training_responses_for_timing':64},
        data_description='Pinned owned192PNG/AGYwordings; only TRAINinputs used in this preflight, no privateweights.',data_files=files)
    (run/'report.md').write_text('# Visual word CONTROL preflight\n\nParity and TRAIN-only timing in progress. No qualityselection/mainchange.\n',encoding='utf-8')
    activity(run,'Bắt đầu parity đối chứng ALIF ảnh–câu hỏi','Giữ nguyên core/Hermes. BPTT chỉ đối chứng trong controls.py; rule/codec tests4 đã qua. Kiểm3pairedseed ×64trainresponses và timing64, không chấmdev.',status='started',evidence={'protocol':'experiments/visual-word-learnability-control-v1.md','main_algorithm_changed':False,'targeted_tests_passed':4})
    parity=[]
    with ResourceMonitor(interval=.1) as monitor:
        from scarx.evaluation.controls import VisualWordLearnabilityControl,torch
        torch.set_num_threads(1)
        for seed in SEEDS:
            a=calibrated(vocab,seed,train,images);verify_baseline(a,rows,seed);control=VisualWordLearnabilityControl(VisualWords.loads(a.dumps()))
            stream=list(sample_stream(a,train,64));stream_b=list(sample_stream(control.reference,train,64));assert [r['id'] for r in stream]==[r['id'] for r in stream_b]
            for case in stream:
                assert a.learn(case['prompt'],images[case['scene_id']],case['target'])==control.learn(case['prompt'],images[case['scene_id']],case['target'],train_hidden=False)
                assert a.dumps()==control.reference.dumps()
            assert control.backward_passes==control.optimizer_steps==0
            parity.append({'seed':seed,'64_trainresponses_fullstatebytes_exact':True,'source_initialhidden_calibration_exact':True})
            progress(run,'Recording-only parity '+str(len(parity))+'/3',len(parity),4,metrics={'seed':seed,'exact_state':True});monitor.check()
            if time.perf_counter()-began>300:raise TimeoutError('Registered preflight300s cap')
        started=time.perf_counter();m=calibrated(vocab,34103,train,images);control=VisualWordLearnabilityControl(m);setup=time.perf_counter()-started
        original=m.core.w_in.copy();started=time.perf_counter();losses=[]
        for i,case in enumerate(sample_stream(m,train,64),1):
            losses.append(control.learn(case['prompt'],images[case['scene_id']],case['target'])['loss'])
            if i%16==0:progress(run,f'TRAIN-only timing {i}/64',3,4,metrics={'responses':i,'loss_mean':sum(losses)/len(losses),'seconds':time.perf_counter()-started,'backward_passes':control.backward_passes});monitor.check()
            if time.perf_counter()-began>300:raise TimeoutError('Registered preflight300s cap')
        elapsed=time.perf_counter()-started
        assert (original!=m.core.w_in).any() and control.optimizer_steps==64 and control.last_gradient_norm is not None
        payload=control.dumps();copy=VisualWordLearnabilityControl.loads(payload);assert copy.dumps()==payload
        case=train[0];assert control.learn(case['prompt'],images[case['scene_id']],case['target'])==copy.learn(case['prompt'],images[case['scene_id']],case['target'])
        assert control.dumps()==copy.dumps() and control.generate(case['prompt'],images[case['scene_id']])==copy.generate(case['prompt'],images[case['scene_id']]) and control.dumps()==copy.dumps()
        forecast=1.25*(elapsed/64*1536+setup+40);round_forecast=3*forecast+90
        feasible=forecast<1500 and round_forecast<6000 and not monitor.report()['observed_over_limit']
        result={'run':run.relative_to(root).as_posix(),'source_run':SOURCE,'decision':'VISUAL_CONTROL_PREFLIGHT_FEASIBLE' if feasible else 'VISUAL_CONTROL_BUDGET_NOT_FEASIBLE',
            'all_matched':True,'frozen_baselines':parity,'timing':{'seed':34103,'responses':64,'training_seconds':elapsed,'setup_seconds':setup,
            'forecast_trial_with25pct_headroom':forecast,'forecast_round_seconds':round_forecast,'max_graph_state_error':control.max_graph_state_error,
            'max_graph_feature_error':control.max_graph_feature_error,'max_graph_loss_error':control.max_graph_loss_error,
            'backward_passes_timing_only':64,'codec_nextupdate_generation_exact':True},'formal_training_permitted':feasible,
            'control_only':True,'deployed':False,'normal_chat_or_vision_proven':False,'dataset':receipt['data'],'dataset_sha256':receipt['sha256'],
            'resources':monitor.report(),'cap':cap,'torch_version':torch.__version__}
        (run/'report.md').write_text('# Isolated visual ALIF CONTROL preflight\n\nNo devquality evaluated; mainALIF/locallearning unchanged.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/visual-word-control-preflight.json',result)
        activity(run,'Hoàn tất parity và dự toán visualcontrol',result['decision']+'; không chấmdev hoặc đổi mainweights. '+str(round(forecast,1))+'s/trial dự kiến.',evidence={'preflight':result['run'],'all_matched':True,'formal_training_permitted':feasible})
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
