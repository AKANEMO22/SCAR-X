import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.core import Config
from scarx.neural.local_target import LocalTargetCore
from scarx.evaluation.readout_conditioning import ConditionedReadout
from scarx.evaluation.tasks21 import representation,ridge_probe
from scarx.evaluation.binding_audit21 import strict_binding
from scarx.evaluation.runs import new_run,write_json,cluster_interval
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor

cap=apply_memory_limit();started=time.monotonic();run=new_run(root,'p2-readout-conditioning-v1',{'protocol':'experiments/p2-readout-conditioning-v1.md','seeds':[3109,3119,3137],'hidden':48,'hidden_updates':0,'calibration_examples':512,'labeled_updates':1536},data_description='Fixed hidden features from existing temporal generators; independent train-only per-neuron normalization control')
rows=[]
with ResourceMonitor(interval=.1) as monitor:
    for task in ('order','xor','strict_binding'):
      for seed in (3109,3119,3137):
        model=LocalTargetCore(Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed,anchor_strength=0))
        def generate(offset,count):return strict_binding(seed+offset,count) if task=='strict_binding' else representation(task,seed+offset,count)
        x,y=generate(1500000,2048);fit,fit_y=generate(1600000,128);test,test_y=generate(1700000,256)
        features=model.features_many(x);test_f=model.features_many(test);fit_f=model.features_many(fit)
        probe=ridge_probe(fit_f,fit_y,test_f,test_y)
        for mode in ('raw','centered','normalized'):
            head=ConditionedReadout(model.w_out,model.bias,mode)
            for f in features[:512]:head.observe(f)
            head.freeze();loss=[]
            for f,label in zip(features[512:],y[512:]):loss.append(head.learn(f,int(label)))
            predictions=np.argmax(head.probabilities(test_f),axis=1)
            item={'task':task,'seed':seed,'condition':mode,'accuracy':float(np.mean(predictions==test_y)),
              'ridge_probe':probe,'updates':head.updates,'loss':float(np.mean(loss)),'hidden_frozen':True}
            rows.append(item);write_json(run/f'{task}-{seed}-{mode}.json',{'summary':item,'targets':test_y.tolist(),'predictions':predictions.tolist()})
        monitor.check()
        if time.monotonic()-started>120:raise TimeoutError('Registered120s budget')
    summary={}
    for mode in ('centered','normalized'):
        summary[mode]={}
        for task in ('order','xor','strict_binding'):
            current=[r for r in rows if r['condition']==mode and r['task']==task]
            delta=[r['accuracy']-next(b['accuracy'] for b in rows if b['condition']=='raw' and b['task']==task and b['seed']==r['seed']) for r in current]
            summary[mode][task]={'accuracy':float(np.mean([r['accuracy'] for r in current])),'gain_vs_raw':cluster_interval(delta)}
    informative=any(sum(v['accuracy']>=.65 and v['gain_vs_raw']['mean']>=.10 for v in tasks.values())>=2 for tasks in summary.values())
    result={'run':run.relative_to(root).as_posix(),'trials':rows,'summary':summary,'screen_informative':informative,'resources':monitor.report(),'cap':cap,
      'hidden_representation_gain':False,'retention_tested':False,'deployed':False,'normal_chat_or_vision':False,
      'decision':'REGISTER_CONFIRMATION_AND_RETENTION_NEXT' if informative else 'STOP_CONDITIONING_V1'}
    write_json(run/'result.json',result);write_json(root/'artifacts/p2-readout-conditioning-latest.json',result)
    (run/'report.md').write_text('# P2 output-local conditioning diagnostic\n\nNo hidden updates. A possible decoding/acquisition control, not useful hidden representation or solved continual learning.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':str(run),'summary':summary,'screen_informative':informative}),flush=True)
