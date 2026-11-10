import json
from pathlib import Path
import time
import numpy as np
from ..neural.core import Config
from ..neural.plasticity import PlasticCore
from ..runtime.resources import ResourceMonitor
from .tasks21 import ridge_probe
from .phase21 import interval,signflip
from .runs import new_run,write_json

SEEDS=(307,331,353,379,401,431,457,479)


def strict_binding(seed,count):
    rng=np.random.default_rng(seed); y=np.arange(count)%2
    x=rng.normal(0,.08,(count,12,8)).astype(np.float32)
    for i in range(count):
        query=int(rng.integers(2)); a=int(y[i])^query; values=(a,1-a)
        order=rng.permutation(2); positions=(int(rng.integers(1,3)),int(rng.integers(4,7)))
        for position,key in zip(positions,order):
            x[i,position,key]+=1.5; x[i,position,2+values[key]]+=1.5
        x[i,10,4+query]+=1.5; x[i,11,7]+=1
    order=rng.permutation(count)
    return x[order],y[order].astype(np.int32)


def run(root):
    root=Path(root); output=new_run(root,'phase21-binding-audit-v1',{'seeds':SEEDS,'selected_rule':'score','posthoc_task_validity_audit':True}); rows=[]
    with ResourceMonitor() as monitor:
        for seed in SEEDS:
            c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed)
            x,y=strict_binding(seed+200000,512); fx,fy=strict_binding(seed+210000,128); tx,ty=strict_binding(seed+220000,256)
            bag=ridge_probe(fx.sum(1),fy,tx.sum(1),ty)
            initial=PlasticCore(c,'readout')
            initial_probe=ridge_probe(initial.features_many(fx,901),fy,initial.features_many(tx,902),ty)
            for condition in ('selected','readout','shuffled'):
                m=PlasticCore(c,'readout' if condition=='readout' else 'score',shuffle=condition=='shuffled')
                start=time.perf_counter()
                for seq,label in zip(x,y): m.forward(seq,track=True); m.learn(int(label))
                elapsed=time.perf_counter()-start
                p=ridge_probe(m.features_many(fx,901),fy,m.features_many(tx,902),ty)
                row={'seed':seed,'condition':condition,'bag_accuracy':bag,'initial_probe':initial_probe,'probe':p,'probe_gain':p-initial_probe,
                     'accuracy':float(np.mean(m.predict_many(tx,902)==ty)),'train_seconds':elapsed}
                rows.append(row); print('binding audit '+json.dumps(row),flush=True); monitor.check()
        selected=[r for r in rows if r['condition']=='selected']
        summary={'probe_gain':interval([r['probe_gain'] for r in selected]),'accuracy':interval([r['accuracy'] for r in selected]),
                 'bag_accuracy':interval([r['bag_accuracy'] for r in selected]),'probe':interval([r['probe'] for r in selected]),
                 'signflip_p':signflip([r['probe_gain'] for r in selected])}
        summary['pass']=summary['probe_gain']['lower95']>=.05 and summary['accuracy']['mean']>=.65 and summary['signflip_p']<=.05
        result={'trials':rows,'summary':summary,'resources':monitor.report()}
        write_json(output/'result.json',result)
        (output/'report.md').write_text('# Strict binding audit\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print('RESULT '+str(output),flush=True)
    return output


if __name__=='__main__': run(Path.cwd())
