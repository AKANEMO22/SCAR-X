"""Word bridge no-devscores parity/timing before formal fixedscreen."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from direct_words_common import source_data,make,stream,reference,load
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();receipt,train,_,vocab,files=source_data();began=time.perf_counter()
    run=new_run(root,'direct-word-preflight-v1',{'protocol':'experiments/direct-word-bridge-v1.md','display_title':'Proposal30 WORD ·parity/timing, chưa devscores',
        'model_scope':'Fixedwordbridge G1input, no private/mainchange. TRAINonly64responses seed36303, no hyperparameterselection.','timing_seed':36303,'new_training':'discardedTRAINonly timing copies'},
        data_description='ReviewedAGY64TRAINprompts only duringtiming; nodevquality.',data_files=files)
    (run/'report.md').write_text('# Wordbridgepreflight\n\nTRAINonly parity/timing, no devscore.\n',encoding='utf-8');timing={}
    activity(run,'Bắt đầu wordbridge engineering/timing','4tests parity/codec/rollback/control qua. ActualAGY128reviewed/1repair,64train/64dev. Dùng unchangedeligibility35, không tuning hoặc mainchange.',status='started',evidence={'reviewed_data':receipt['data'],'targeted_tests':4})
    with ResourceMonitor(interval=.1) as monitor:
        a=make(vocab,36303,'zero',train);b=make(vocab,36303,'frozen',train);sa=list(stream(a,train,64));sb=list(stream(b,train,64));assert [r['id'] for r in sa]==[r['id'] for r in sb]
        for row in sa:assert a.learn(row['prompt'],row['target'])==b.learn(row['prompt'],row['target'])
        assert a.core.dumps()==b.core.dumps() and a.head.dumps()==b.head.dumps();progress(run,'Zero/frozen64response exact',1,3)
        for method in ('correct','bptt_control'):
            start=time.perf_counter()
            if method=='bptt_control':
                from scarx.evaluation.controls import torch
                torch.set_num_threads(1)
            m=make(vocab,36303,method,train);setup=time.perf_counter()-start;start=time.perf_counter()
            for row in stream(reference(m),train,64):m.learn(row['prompt'],row['target'])
            elapsed=time.perf_counter()-start;payload=m.dumps();copy=load(payload,method);assert copy.dumps()==payload
            row=train[0];assert m.learn(row['prompt'],row['target'])==copy.learn(row['prompt'],row['target']) and m.dumps()==copy.dumps()
            assert m.generate(row['prompt'])==copy.generate(row['prompt']) and m.dumps()==copy.dumps()
            forecast=1.25*(elapsed/64*1024+setup+20)
            timing[method]={'responses':64,'training_seconds':elapsed,'setup_seconds':setup,'forecast_trial_seconds':forecast,'array_bytes':reference(m).memory_bytes(),'checkpoint_bytes':len(payload),'continuation_exact':True}
            if method=='bptt_control':timing[method].update(max_graph_state_error=m.max_graph_state_error,max_graph_feature_error=m.max_graph_feature_error,max_graph_loss_error=m.max_graph_loss_error)
            monitor.check();progress(run,'Timingcomplete '+method,2 if method=='correct' else 3,3,metrics=timing[method])
            if time.perf_counter()-began>180:raise TimeoutError('Wordpreflight180s cap')
        forecast=12*timing['correct']['forecast_trial_seconds']+3*timing['bptt_control']['forecast_trial_seconds'];feasible=forecast<2400 and all(v['forecast_trial_seconds']<300 for v in timing.values())
        result={'run':run.relative_to(root).as_posix(),'decision':'DIRECT_WORD_PREFLIGHT_PASS' if feasible else 'STOP_DIRECT_WORD_BUDGET','timing':timing,'forecast_round_seconds':forecast,
            'formal_training_permitted':feasible,'zero_frozen64exact':True,'no_development_quality_evaluated':True,'deployed':False,'resources':monitor.report(),'cap':cap}
        (run/'report.md').write_text('# Wordbridgepreflight\n\nNo devquality/selection, mainunchanged.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/direct-word-preflight.json',result);print(json.dumps(result))


if __name__=='__main__':main()
