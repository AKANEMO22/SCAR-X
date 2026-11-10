"""P0 proposal30 math, fullstate checks and TRAIN-only timing, no devscores."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from direct_eligibility_common import make,dataset,reference
from scarx.evaluation.readout_eligibility import ReadoutEligibility
from scarx.evaluation.eligibility_validation import finite_difference_check
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import activity
from scarx.runtime.experiment_status import progress


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    run=new_run(root,'direct-eligibility-engineering-v1',{'protocol':'experiments/direct-readout-eligibility-v1.md',
        'display_title':'PROPOSAL30 · localeligibility math/codec/timing',
        'model_scope':'Userapproved research only. No Transformer/BPTTmain/autogradcandidate, no devquality or deployment.',
        'proposal30_approved':True,'math_seed':35101,'timing_seed':35103,'timing_train_examples':64})
    (run/'report.md').write_text('# Proposal30 engineering\n\nMath/fullstate/TRAIN-only timing in progress; no devquality selection or mainchange.\n',encoding='utf-8')
    activity(run,'Bắt đầu nghiên cứu30 đãđượcduyệt','User: Duyệt nghiên cứu đề xuất30.6targetedtests qua; kiểm finite-difference64synapses, checkpoint và timingtrainonly. MainALIF/Hermes khôngđổi.',status='started',evidence={'approved_on':'2026-11-10','protocol':'experiments/direct-readout-eligibility-v1.md','targeted_tests_passed':6})
    with ResourceMonitor(interval=.1) as monitor:
        m=make('xor',35101,'correct');x,y=dataset('xor',35101+1370000,64)
        mathematics=finite_difference_check(m,x[0]);progress(run,'64local sensitivities finite-difference matched',1,5,metrics=mathematics)
        payload=m.dumps();copy=ReadoutEligibility.loads(payload);assert payload==copy.dumps()
        m.begin(True)
        for row in x[1,:5]:m.step(row)
        copy=ReadoutEligibility.loads(m.dumps())
        for row in x[1,5:]:np.testing.assert_array_equal(m.step(row),copy.step(row))
        assert m.accumulate(int(y[1]))==copy.accumulate(int(y[1]));m.finish();copy.finish();assert m.dumps()==copy.dumps()
        assert m.learn(x[2],int(y[2]))==copy.learn(x[2],int(y[2])) and m.dumps()==copy.dumps()
        progress(run,'Partialsegment/Adam/RNG continuation exact',2,5,metrics={'partial_state_exact':True});monitor.check()
        zero=make('xor',35101,'zero');frozen=make('xor',35101,'frozen')
        for seq,label in zip(x[:16],y[:16]):assert zero.learn(seq,int(label))==frozen.learn(seq,int(label))
        assert zero.core.dumps()==frozen.core.dumps() and zero.head.dumps()==frozen.head.dumps()
        progress(run,'Zero/frozen neural+head state exact',3,5,metrics={'zero_frozen_exact':True})
        timing={}
        for method in ('correct','bptt_control'):
            started=time.perf_counter()
            if method=='bptt_control':
                from scarx.evaluation.controls import torch
                torch.set_num_threads(1)
            model=make('xor',35103,method);setup=time.perf_counter()-started;train,labels=dataset('xor',35103+1370000,64);started=time.perf_counter()
            for seq,label in zip(train,labels):model.learn(seq,int(label))
            elapsed=time.perf_counter()-started;ref=reference(model);forecast=1.25*(elapsed/64*1024+setup+15)
            raw=model.dumps();from direct_eligibility_common import load
            duplicate=load(method,raw);assert duplicate.dumps()==raw
            assert model.learn(train[0],int(labels[0]))==duplicate.learn(train[0],int(labels[0])) and model.dumps()==duplicate.dumps()
            timing[method]={'training_examples':64,'training_seconds':elapsed,'setup_seconds':setup,'forecast_trial_with25pct_headroom':forecast,
                'trace_and_core_array_bytes':ref.memory_bytes(),'checkpoint_bytes':len(raw),'optimizer_hidden_updates_timing':64,'codec_nextupdate_exact':True}
            if method=='bptt_control':timing[method].update(max_graph_state_error=model.max_graph_state_error,max_graph_feature_error=model.max_graph_feature_error)
            progress(run,'TRAIN-only timing '+method,4 if method=='correct' else 5,5,metrics=timing[method]);monitor.check()
            if time.perf_counter()-began>180:raise TimeoutError('Registered180s P0cap')
        forecast_round=36*timing['correct']['forecast_trial_with25pct_headroom']+9*timing['bptt_control']['forecast_trial_with25pct_headroom']
        feasible=forecast_round<1800 and max(v['forecast_trial_with25pct_headroom'] for v in timing.values())<180 and not monitor.report()['observed_over_limit']
        result={'run':run.relative_to(root).as_posix(),'decision':'DIRECT_ELIGIBILITY_ENGINEERING_PASS' if feasible else 'DIRECT_ELIGIBILITY_BUDGET_NOT_FEASIBLE',
            'proposal30_approved':True,'mathematics':mathematics,'partialsegment_optimizer_rng_exact':True,'zero_frozen_neural_head_exact':True,
            'timing':timing,'forecast_screen_seconds':forecast_round,'formal_screen_permitted':feasible,'no_development_quality_evaluated':True,
            'deployed':False,'main_weights_changed':False,'normal_chat_or_vision_proven':False,'resources':monitor.report(),'cap':cap}
        (run/'report.md').write_text('# Proposal30 engineering\n\nNo devquality evaluation, no mainchange. Finite-difference oracle is anchored local smooth sensitivity, NOT hardspike/fullnetworkgradient.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/direct-eligibility-engineering.json',result)
        activity(run,'Hoàn tất engineering30, chưa cóqualityclaim',result['decision']+'; math64/partialstate/zero-frozen exact. Forecastscreen '+str(round(forecast_round,1))+'s, không mainchange.',evidence={'formal_screen_permitted':feasible,'engineering_run':result['run'],'forecast_seconds':forecast_round})
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
