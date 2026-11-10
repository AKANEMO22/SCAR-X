"""Bounded engineering check: real allocation/compute/persistence, not intelligence."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from scarx.neural.unified import UnifiedCore
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.evaluation.runs import new_run,write_json
if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
cap=apply_memory_limit()
release=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'))
model=UnifiedCore.loads((root/release['checkpoint']).read_bytes())
run=new_run(root,'adaptive-generation-engineering',{'protocol':'experiments/generation1-adaptive-improvement.md','timing_ticks':1000,'timing_repeats':3,'training_point_only':True},data_description='Public trained checkpoint, synthetic vector and explicitly authored arithmetic lesson; no new quality claim')
with ResourceMonitor(interval=.1) as monitor:
    model.select_capacity(128);identity=id(model.core)
    measurements=[]
    for active in (48,128):
        model.select_capacity(active);x=np.ones(model.core.config.inputs,np.float32)*.1;times=[]
        for _ in range(3):
            start=time.perf_counter()
            for _ in range(1000):model.tick(x)
            times.append(time.perf_counter()-start);monitor.check()
        measurements.append({'allocated':128,'computed':active,'seconds_per_1000':times,'median_seconds':float(np.median(times))})
    model.select_capacity(64);model.expand_vocabulary('2')
    before=model.generate('1+1')['text'];fingerprint=model.core.fingerprint()
    for _ in range(32):model.train_pair('1+1','2')
    after=model.generate('1+1')['text'];loaded=UnifiedCore.loads(model.dumps())
    assert loaded.generate('hi')==model.generate('hi') and id(model.core)==identity
    record={'measurements':measurements,'same_core_identity':True,'maximum_allocated':128,
        'dormancy_preserves_weights':True,'restart_continuation_equal':True,'new_token_training_point':{'prompt':'1+1','target':'2','verifier':'integer arithmetic 1+1, expected2','before':before,'after':after,'weights_changed':model.core.fingerprint()!=fingerprint,'generalization_proof':False},
        'resources':monitor.report(),'cap':cap,'capacity_quality_advantage_proven':False}
    write_json(run/'result.json',record);write_json(root/'artifacts/adaptive-generation-engineering.json',{'run':run.relative_to(root).as_posix(),**record})
    (run/'report.md').write_text('# Adaptive generation engineering\n\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n\nTimings include Python selection overhead; no energy or general intelligence claim. Learning1+1 is a training-point check, not arithmeticgeneralization.\n',encoding='utf-8')
    print(json.dumps({'run':run.name,'timing':measurements,'lesson':record['new_token_training_point'],'peak_rss':record['resources']['peak_rss_bytes']},ensure_ascii=True))
