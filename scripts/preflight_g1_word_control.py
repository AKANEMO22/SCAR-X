"""Isolated control parity and training-only timing forecast, no dev selection."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,itertools
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from train_g1_word_score_inputs import source_data,SEEDS
from generate_g1_state_paraphrases import INTENTS
from scarx.evaluation.score_word import ScoreWord
from scarx.evaluation.fixed_word_codes import fit
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def stream(train,seed,count=64):
    groups=[[r for r in train if r['intent']==intent] for intent in INTENTS];rng=np.random.default_rng(seed+10200000)
    for _ in range(count):
        group=groups[int(rng.integers(8))];yield group[int(rng.integers(len(group)))]


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();train,_,_,vocab,sources=source_data()
    source_info=json.loads((root/'artifacts/g1-word-score-inputs-latest.json').read_text(encoding='utf-8'))
    source=root/source_info['run'];old=json.loads((source/'trials.json').read_text(encoding='utf-8'))
    encodings={'legacy':None,'known-digest128-v1':fit([r['prompt'] for r in train],'known-digest128-v1')}
    run=new_run(root,'g1-word-control-preflight-v1',{'protocol':'experiments/g1-word-learnability-control-v1.md',
        'display_title':'CONTROL ONLY · parity và dự toán BPTT cô lập',
        'model_scope':'No heldout evaluation; mainALIF/locallearning untouched; ONE Torchprocess.',
        'control_only':True,'timing_seed':29011,'training_responses_for_timing':64,'paired_seeds':SEEDS},
        data_description='Public trainingprompts only; oldfrozencheckpoint calibration/parity, no dev/prior quality measurements.',data_files=sources)
    (run/'report.md').write_text('# Word learnability control preflight\n\nParity/timing in progress; no development/prior scores or mainlearning change.\n',encoding='utf-8')
    parity=[];timing={}
    with ResourceMonitor(interval=.1) as monitor:
        from scarx.evaluation.controls import WordLearnabilityControl,torch
        torch.set_num_threads(1)
        for seed,name in itertools.product(SEEDS,encodings):
            encoding=encodings[name];a=ScoreWord(vocab,seed,'frozen',encoding);b=ScoreWord(vocab,seed,'frozen',encoding)
            a.calibrate([r['prompt'] for r in train]);b.calibrate([r['prompt'] for r in train]);assert a.dumps()==b.dumps()
            previous=next(r for r in old if (r['seed'],r['input_encoding'],r['condition'])==(seed,name,'frozen'))
            frozen=ScoreWord.loads((root/previous['checkpoint']).read_bytes())
            for key in ('w_in','w_rec','indices','head_mean','head_scale'):np.testing.assert_array_equal(getattr(frozen.core,key),getattr(a.core,key))
            control=WordLearnabilityControl(b)
            for row in stream(train,seed):assert a.learn(row['prompt'],row['target'])==control.learn(row['prompt'],row['target'])
            assert a.dumps()==control.reference.dumps() and not control.backward_passes
            parity.append({'seed':seed,'encoding':name,'64_response_statebytes_exact':True,'source_frozen_calibration_exact':True})
            progress(run,'Frozen NumPy parity '+str(len(parity))+'/6',len(parity),8,metrics={'all_exact':True})
            monitor.check()
            if time.perf_counter()-began>300:raise TimeoutError('Preflight300s cap')
        for name,encoding in encodings.items():
            t0=time.perf_counter();m=ScoreWord(vocab,29011,'full',encoding);m.calibrate([r['prompt'] for r in train]);setup=time.perf_counter()-t0
            control=WordLearnabilityControl(m);t0=time.perf_counter()
            for index,row in enumerate(stream(train,29011),1):
                control.learn(row['prompt'],row['target'])
                if index%16==0:
                    progress(run,f'{name} · timing{index}/64 · no devscores',6+len(timing),8,metrics={'training_responses':index,'elapsed_training_seconds':time.perf_counter()-t0});monitor.check()
                if time.perf_counter()-began>300:raise TimeoutError('Preflight300s cap')
            elapsed=time.perf_counter()-t0
            forecast=(elapsed/64*1024+setup+15)*1.25
            timing[name]={'responses':64,'training_seconds':elapsed,'setup_seconds':setup,'forecast_trial_with25pct_headroom':forecast,
                'numpy_forward_ticks':m.core.ticks,'surrogate_forward_ticks':control.surrogate_forward_ticks,'backward_passes':control.backward_passes,
                'max_graph_state_error':control.max_graph_state_error,'max_graph_feature_error':control.max_graph_feature_error}
        forecast_round=3*sum(v['forecast_trial_with25pct_headroom'] for v in timing.values())+30
        feasible=forecast_round<6000 and all(v['forecast_trial_with25pct_headroom']<900 for v in timing.values()) and not monitor.report()['observed_over_limit']
        result={'run':run.relative_to(root).as_posix(),'source_run':source_info['run'],'decision':'CONTROL_PREFLIGHT_FEASIBLE' if feasible else 'CONTROL_BUDGET_NOT_FEASIBLE',
            'all_matched':True,'frozen_baselines':parity,'timing':timing,'forecast_round_seconds':forecast_round,'formal_training_permitted':feasible,
            'control_only':True,'new_training':'64trainingresponse timing copies only, no development/prior evaluation',
            'deployed':False,'normal_chat_or_vision_proven':False,'resources':monitor.report(),'cap':cap,'torch_version':torch.__version__}
        (run/'report.md').write_text('# Word learnability control preflight\n\nSixfrozen NumPy baselines match exactstatebytes. Training-only timing, no heldout selection.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-word-control-preflight.json',result)
        activity(run,'Hoàn tất parity và dự toán đối chứng',result['decision']+'; không đổi learningrule chính hoặc đánh giá heldout trong preflight.',evidence={'all_matched':True,'forecast_round_seconds':forecast_round,'formal_training_permitted':feasible})
        print(json.dumps({'run':result['run'],'decision':result['decision'],'timing':timing,'forecast_round_seconds':forecast_round,'resources':result['resources']},ensure_ascii=False))


if __name__=='__main__':main()
