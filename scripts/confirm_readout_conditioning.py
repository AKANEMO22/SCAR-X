import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.core import Config
from scarx.neural.local_target import LocalTargetCore
from scarx.evaluation.readout_conditioning import ConditionedReadout
from scarx.evaluation.tasks21 import representation,context_task,ridge_probe
from scarx.evaluation.binding_audit21 import strict_binding
from scarx.evaluation.phase21 import interval,signflip,holm
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor

SEEDS=(4103,4111,4127,4133,4153,4159,4177,4201)


class BalancedReplay:
    def __init__(self,capacity=128):self.capacity=capacity;self.rows=[]
    def add(self,context,label,feature):
        self.rows.append((context,int(label),feature.copy()))
        while len(self.rows)>self.capacity:
            groups={key:sum((a,b)==key for a,b,_ in self.rows) for key in {(a,b) for a,b,_ in self.rows}}
            largest=max(sorted(groups),key=groups.get)
            index=next(i for i,(a,b,_) in enumerate(self.rows) if (a,b)==largest)
            self.rows.pop(index)
    def sample(self,rng):
        groups=sorted({(a,b) for a,b,_ in self.rows});key=groups[int(rng.integers(len(groups)))];items=[r for r in self.rows if r[:2]==key]
        _,label,feature=items[int(rng.integers(len(items)))];return feature,label


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.monotonic();run=new_run(root,'p2-conditioning-confirmation-v1',{'protocol':'experiments/p2-readout-conditioning-confirmation-v1.md','seeds':SEEDS,'hidden':48,'hidden_updates':0,'contexts':20},data_description='New independent temporal and20context generators; calibration context0only; no private data')
    direct=[];streams=[]
    with ResourceMonitor(interval=.1) as monitor:
      for seed in SEEDS:
        for task in ('order','xor','strict_binding'):
            model=LocalTargetCore(Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed,anchor_strength=0))
            def generate(offset,count):return strict_binding(seed+offset,count) if task=='strict_binding' else representation(task,seed+offset,count)
            x,y=generate(1500000,2048);test,ty=generate(1700000,512);fit,fy=generate(1600000,128)
            f=model.features_many(x);tf=model.features_many(test);probe=ridge_probe(model.features_many(fit),fy,tf,ty)
            for mode in ('raw','normalized'):
                head=ConditionedReadout(model.w_out,model.bias,mode)
                for row in f[:512]:head.observe(row)
                head.freeze()
                for row,label in zip(f[512:],y[512:]):head.learn(row,int(label))
                predictions=np.argmax(head.probabilities(tf),axis=1)
                item={'seed':seed,'task':task,'condition':mode,'accuracy':float(np.mean(predictions==ty)),'ridge_probe':probe,'updates':head.updates}
                direct.append(item);write_json(run/f'direct-{task}-{seed}-{mode}.json',{'summary':item,'targets':ty.tolist(),'predictions':predictions.tolist()})
        model=LocalTargetCore(Config(inputs=24,outputs=4,hidden=48,fan_in=8,plastic_rows=12,seed=seed,anchor_strength=0))
        fingerprint=model.core.fingerprint() if hasattr(model,'core') else model.fingerprint()
        calibration,_=context_task(seed+2000000,128,0,20);cal=model.features_many(calibration)
        train=[];test=[]
        for context in range(20):
            x,y=context_task(seed+2100000+context*101,32,context,20);tx,ty=context_task(seed+2200000+context*101,128,context,20)
            train.append((model.features_many(x),y));test.append((model.features_many(tx),ty))
        for condition in ('raw','normalized','raw_replay','normalized_replay'):
            t0=time.monotonic();head=ConditionedReadout(model.w_out,model.bias,'normalized' if condition.startswith('normalized') else 'raw')
            for row in cal:head.observe(row)
            head.freeze();rng=np.random.default_rng(seed+2300000);buffer=BalancedReplay();matrix=[]
            for context,(f,labels) in enumerate(train):
                prior=BalancedReplay();prior.rows=list(buffer.rows)
                for slot in range(128):
                    if condition.endswith('_replay') and prior.rows and slot%4==3:feature,label=prior.sample(rng)
                    else:feature,label=f[slot%32],int(labels[slot%32])
                    head.learn(feature,int(label))
                for feature,label in zip(f,labels):buffer.add(context,int(label),feature)
                matrix.append([float(np.mean(np.argmax(head.probabilities(tf),axis=1)==ty)) for tf,ty in test[:context+1]])
            diagonal=np.array([row[i] for i,row in enumerate(matrix)]);final=np.array(matrix[-1]);best=np.array([max(matrix[j][i] for j in range(i,20)) for i in range(20)]);forget=best-final
            clone=ConditionedReadout.loads(head.dumps());assert np.array_equal(clone.probabilities(test[0][0]),head.probabilities(test[0][0]))
            clone.learn(train[-1][0][0],int(train[-1][1][0]));head.learn(train[-1][0][0],int(train[-1][1][0]));assert head.dumps()==clone.dumps()
            assert model.fingerprint()==fingerprint
            passed=bool(diagonal.mean()>=.65 and final.mean()>=.65 and forget.mean()<=.03 and forget.max()<=.05)
            item={'seed':seed,'condition':condition,'acquisition':float(diagonal.mean()),'final':float(final.mean()),'mean_forgetting_pp':float(100*forget.mean()),'worst_forgetting_pp':float(100*forget.max()),
              'signed_bwt':float(np.mean(final[:-1]-diagonal[:-1])),'gate':passed,'updates':2560,'resume_exact':True,'replay_examples':len(buffer.rows),'seconds':time.monotonic()-t0}
            streams.append(item);write_json(run/f'stream-{seed}-{condition}.json',{'summary':item,'accuracy_matrix':matrix,'forgetting_by_context':forget.tolist()})
        monitor.check()
        if time.monotonic()-began>180:raise TimeoutError('Registered180s total budget')
        print(json.dumps({'seed':seed,'direct_done':len(direct),'streams_done':len(streams)}),flush=True)
      summary={};pvalues=[]
      for task in ('order','xor','strict_binding'):
          rs=[r for r in direct if r['task']==task and r['condition']=='normalized']
          delta=[r['accuracy']-next(b['accuracy'] for b in direct if b['task']==task and b['condition']=='raw' and b['seed']==r['seed']) for r in rs]
          summary[task]={'accuracy':interval([r['accuracy'] for r in rs]),'gain_vs_raw':interval(delta),'signflip_p':signflip(delta)};pvalues.append(summary[task]['signflip_p'])
      adjusted=holm(pvalues)
      for i,t in enumerate(('order','xor','strict_binding')):
          v=summary[t];v['holm_p']=adjusted[i];v['condition_gate']=v['accuracy']['mean']>=.65 and v['gain_vs_raw']['lower95']>=.10 and adjusted[i]<=.05
      stream_summary={}
      for condition in ('raw','normalized','raw_replay','normalized_replay'):
          rs=[r for r in streams if r['condition']==condition]
          stream_summary[condition]={k:interval([r[k] for r in rs]) for k in ('acquisition','final','mean_forgetting_pp','worst_forgetting_pp')}
          stream_summary[condition]['every_seed_gate']=all(r['gate'] for r in rs)
      result={'run':run.relative_to(root).as_posix(),'direct_confirmation':summary,'direct_gate':sum(v['condition_gate'] for v in summary.values())>=2,
        'continual_summary':stream_summary,'continual_every_seed_gate':any(v['every_seed_gate'] for v in stream_summary.values()),
        'direct_trials':direct,'stream_trials':streams,'resources':monitor.report(),'cap':cap,'deployed':False,'hidden_frozen':True,
        'hidden_representation_gate':False,'equal_time_comparison_performed':False,'normal_chat_or_vision_proven':False,
        'decision':'P2_DIRECT_CONDITIONING_CONFIRMED_RETENTION_PENDING' if sum(v['condition_gate'] for v in summary.values())>=2 else 'P2_CONDITIONING_NOT_CONFIRMED'}
      write_json(run/'result.json',result);write_json(root/'artifacts/p2-conditioning-confirmation-latest.json',result)
      (run/'report.md').write_text('# P2 conditioning confirmation /20context stress\n\nAll8fresh independent seeds; identical fixed hidden cores and local readout budget. No hidden learning gain, new language/vision competence or deployment.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
      print('RESULT '+str(run),flush=True)


if __name__=='__main__':main()
