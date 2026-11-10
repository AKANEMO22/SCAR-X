"""One isolated process, fixed checkpoints and inputs, no optimizer/learning."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,itertools
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from train_g1_word_score_inputs import source_data,SEEDS
from generate_g1_state_paraphrases import INTENTS
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.neural.language import tokenize

STATES=('frozen_final','local_final','surrogate_final')
METHODS=('existing_score','crossfit_baseline_score','biased_surrogate')


def finite_values(values):return [float(v) for v in values if v is not None and np.isfinite(v)]


def median(values):
    values=finite_values(values);return float(np.median(values)) if values else None


def mean(values):
    values=finite_values(values);return float(np.mean(values)) if values else None


def seed_interval(values):
    values=finite_values(values)
    return interval(values) if values else {'mean':None,'lower95':None,'upper95':None,'n_seeds':0}


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();train,_,_,_,datafiles=source_data()
    cases=[next(r for r in train if r['intent']==intent and r['id'].startswith('known-')) for intent in INTENTS]
    local_info=json.loads((root/'artifacts/g1-word-score-inputs-latest.json').read_text(encoding='utf-8'))
    local=json.loads((root/local_info['run']/'trials.json').read_text(encoding='utf-8'))
    control_info=json.loads((root/'artifacts/g1-word-learnability-control-latest.json').read_text(encoding='utf-8'));controls=control_info['trials']
    sources=[]
    for seed,state in itertools.product(SEEDS,STATES):
        source=next(r for r in (controls if state=='surrogate_final' else local) if r['seed']==seed and r['input_encoding']=='known-digest128-v1' and r['condition']==('frozen' if state=='frozen_final' else 'full'))
        path=(root/source['checkpoint']).resolve()
        if not path.is_relative_to(root/'artifacts'):raise ValueError('Source outside artifacts')
        sources.append({'seed':seed,'state':state,'path':source['checkpoint'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    run=new_run(root,'g1-credit-noise-audit-v1',{'protocol':'experiments/g1-credit-noise-audit-v1.md',
        'display_title':'Audit tín hiệu synapse · nhiễu và tái lập · KHÔNG HỌC',
        'model_scope':'Readonly publiccheckpointclones. Surrogate globalcredit ISOLATED CONTROL; source/mainweights untouched.',
        'control_only':True,'new_training':False,'model_seeds':SEEDS,'checkpoint_states':STATES,'conditional_inputs':144,'noise_pairs_per_input':128},
        data_description='Eightfirstauthoredtrainingprompts,firstword/EOS; nineimmutablepubliccheckpoints; no dev/prior/private inputs.',data_files=datafiles+[r['path'] for r in sources])
    (run/'report.md').write_text('# Conditional credit-signal noise audit\n\nSampling in progress, NO training. Global surrogate only isolated biased diagnostic, never groundtruth.\n',encoding='utf-8')
    write_json(run/'source-checkpoints.json',sources);write_json(run/'cases.json',cases)
    activity(run,'Bắt đầu audit tín hiệu tại checkpoint cố định','144conditionalinputs×128noise-pairs; chỉ mô phỏngbản sao. So localEMA/crossfitbaseline/surrogatebiased, không cập nhậtsynapse hoặc tạo câu trả lời giả.',status='started',evidence={'new_training':False,'source_checkpoints':9,'conditional_inputs':144})
    measurements=[];forecast=None;sample_arrays_path=None
    with ResourceMonitor(interval=.1) as monitor:
        from scarx.evaluation.controls import WordLearnabilityControl,torch
        from scarx.evaluation.score_word import ScoreWord
        from scarx.evaluation.credit_noise import sample
        torch.set_num_threads(1)
        for source in sources:
            path=root/source['path'];payload=path.read_bytes()
            for family_index,case in enumerate(cases):
                for position_index in (0,1):
                    model=WordLearnabilityControl.loads(payload).reference if source['state']=='surrogate_final' else ScoreWord.loads(payload)
                    ids=[model.ids[t] for t in tokenize(case['target'])]
                    prefix=[] if position_index==0 else ids;target=ids[0] if position_index==0 else 0
                    rows=np.arange(12)+((family_index+position_index)%4)*12
                    t0=time.perf_counter();measured,raw=sample(model,case['prompt'],prefix,int(target),rows,
                        source['seed']+9120000+family_index*1000+position_index*100,pairs=128)
                    elapsed=time.perf_counter()-t0
                    if forecast is None:
                        forecast=elapsed*144+(time.perf_counter()-began-elapsed)
                        write_json(run/'timing-forecast.json',{'first_configuration_seconds':elapsed,'forecast_round_seconds':forecast,'limit_seconds':1200,'forecast_based_on':'first predefined input, no quality selection'})
                        if forecast>1200:raise TimeoutError('First-configuration forecast exceeds original1200s cap; no downsampling')
                    if not measurements:
                        artifact=root/'artifacts/credit-noise'/run.name;artifact.mkdir(parents=True)
                        sample_arrays_path=(artifact/'first-predefined-input-signals.npz').relative_to(root).as_posix()
                        np.savez_compressed(root/sample_arrays_path,**raw)
                    del raw
                    neural=measured.pop('last_neural_sample')
                    row={'seed':source['seed'],'state':source['state'],'case_id':case['id'],'family':case['intent'],
                        'position':'first_word' if position_index==0 else 'gold_prefix_EOS','seconds':elapsed,**measured}
                    measurements.append(row)
                    progress(run,f'{source["state"]} · seed{source["seed"]} · {case["intent"]}/{row["position"]}',len(measurements),144,
                        metrics={'replicate_cosine_existing':row['summary']['existing_score']['all']['replicate_cosine'],
                            'replicate_cosine_surrogate':row['summary']['biased_surrogate']['all']['replicate_cosine'],
                            'crossfit_noise_ratio':row['comparisons']['all']['crossfit_vs_existing_noise_ratio'],'seconds_per_input':elapsed},neural=neural)
                    write_json(run/'measurements.json',measurements);monitor.check()
                    if time.perf_counter()-began>1200:raise TimeoutError('Registered1200s conditionalaudit limit')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
            activity(run,'Đã đo checkpoint, weights giữ nguyên',f'seed{source["seed"]}/{source["state"]}:16conditionalinputs; không Adam/headupdate, không RNG nguồn thay đổi.',evidence={'source_sha256':source['sha256'],'measurements_done':len(measurements),'new_training':False})
        per_seed=[]
        for seed,state in itertools.product(SEEDS,STATES):
            subset=[r for r in measurements if r['seed']==seed and r['state']==state]
            methods={m:{position:{k:median([r['summary'][m]['all'][k] for r in subset if position=='all' or r['position']==position]) for k in ('replicate_cosine','cross_signal_to_pair_noise','pair_noise_energy','sample_mean_norm')} for position in ('all','first_word','gold_prefix_EOS')} for m in METHODS}
            contrasts={block:{key:median([r['comparisons'][block][key] for r in subset]) for key in ('crossfit_vs_existing_noise_ratio','surrogate_minus_existing_replicate_cosine','existing_mean_cosine_to_biased_surrogate','crossfit_mean_cosine_to_biased_surrogate')} for block in ('all','input','recurrent')}
            per_seed.append({'seed':seed,'state':state,'methods':methods,'contrasts':contrasts})
        summary={state:{m:{k:mean([r['methods'][m]['all'][k] for r in per_seed if r['state']==state]) for k in ('replicate_cosine','cross_signal_to_pair_noise','pair_noise_energy','sample_mean_norm')} for m in METHODS} for state in STATES}
        paired={state:{key:seed_interval([r['contrasts']['all'][key] for r in per_seed if r['state']==state]) for key in ('crossfit_vs_existing_noise_ratio','surrogate_minus_existing_replicate_cosine')} for state in STATES}
        result={'run':run.relative_to(root).as_posix(),'decision':'READONLY_CREDIT_SIGNAL_DIAGNOSTIC_ONLY','summary':summary,'paired_statistics':paired,
            'per_seed':per_seed,'source_checkpoints':sources,'measurements':144,'pairs_per_input':128,'forecast_seconds':forecast,
            'all_source_checkpoints_unchanged':True,'all_weights_learningstate_rng_unchanged':all(r['weights_learningstate_rng_unchanged'] for r in measurements),
            'max_graph_state_error':max(r['max_graph_state_error'] for r in measurements),'max_graph_feature_error':max(r['max_graph_feature_error'] for r in measurements),
            'numpy_forward_ticks':sum(r['numpy_forward_ticks'] for r in measurements),'extra_surrogate_forward_ticks':sum(r['extra_surrogate_forward_ticks'] for r in measurements),
            'backward_passes':sum(r['backward_passes'] for r in measurements),'first_input_raw_signals':sample_arrays_path,
            'new_training':False,'control_only':True,'deployed':False,'main_local_learning_success':False,'normal_chat_or_vision_proven':False,
            'resources':monitor.report(),'cap':cap,'limits':['Finite MonteCarlo128pairs,3pairedmodels; nestedcases are not independentseeds.',
                'Crossmeanenergy maynegative; splitproxy not trueSNR. Surrogate is biased and not truegradient.',
                'No neuron/optimizer/head/RNGlearningupdate. Neuron/eligibilitysimulation only on discardedcopies.',
                'Crossfitbaseline variancecontrol not a newlearner, successfulquality or reopening scorefamilySTOP.']}
        (run/'report.md').write_text('# Conditional credit-signal noise audit\n\nReadonly measurements, no newlearner/qualityclaim. Source/mainweights unchanged. Detailed144inputs in measurements.json.\n\n'+json.dumps({k:v for k,v in result.items() if k not in ('per_seed','source_checkpoints')},indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-credit-noise-latest.json',result)
        activity(run,'Hoàn tất audit nhiễu tín hiệu','Không học thêm, không mở lại familySTOP. Đã lưu144inputmetrics và firstfixture rawsignals; surrogate chỉbiasedcontrol.',evidence={'source_weights_rng_unchanged':True,'paired_statistics':paired,'new_training':False})
        print(json.dumps({'run':result['run'],'summary':summary,'paired_statistics':paired,'resources':result['resources']},ensure_ascii=False))


if __name__=='__main__':main()
