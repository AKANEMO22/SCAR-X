import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,sys,time,itertools,argparse,subprocess
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.evaluation.working_byte_dialogue import WorkingByteDialogue
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.evaluation.runs import new_run,write_json


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--probe',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    cap=apply_memory_limit();data=root/'data/candidates/antigravity-events-curated-20261109T094142Z';rows=[json.loads(l) for l in (data/'experiences.jsonl').read_text(encoding='utf-8').splitlines()]
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='development']
    if args.probe:write_json(args.output,WorkingByteDialogue.loads(args.probe.read_bytes()).score(dev));return
    began=time.monotonic();run=new_run(root,'byte-working-memory-v2',{'protocol':'experiments/byte-working-memory-v2.md','seeds':[9103,9113,9133],'conditions':['no_snapshot','snapshot_raw','snapshot_normalized'],'passes':4,'hidden':48},data_description='Same public96train/24development Antigravity prefix/target examples; activation snapshot within same core, no extra learned knowledge')
    artifact=root/'artifacts/byte-working-memory'/run.name;artifact.mkdir(parents=True);results=[];baseline={}
    with ResourceMonitor(interval=.25) as monitor:
      for condition,seed in itertools.product(('no_snapshot','snapshot_raw','snapshot_normalized'),(9103,9113,9133)):
        t0=time.monotonic();m=WorkingByteDialogue(seed,condition);m.calibrate(train);initial=m.score(dev);wi=m.core.w_in.copy();wr=m.core.w_rec.copy()
        groups={key:[r for r in train if r['dialogue_id']==key] for key in sorted({r['dialogue_id'] for r in train})};stages=[]
        for epoch in range(1,5):
            loss=[]
            for key in m.order_rng.permutation(list(groups)):
                for row in groups[key]:loss.append(m.learn(row['messages'],row['target']))
            m.passes_done=epoch;monitor.check()
            if time.monotonic()-t0>180 or time.monotonic()-began>1200:raise TimeoutError('Registered working-memory budget')
            if epoch in (2,4):
                score=m.score(dev)
                if condition=='no_snapshot':baseline[(seed,epoch)]=score['exact']
                stages.append({'pass':epoch,'loss':float(np.mean(loss)),'development':score,'gain_vs_no_snapshot':score['exact']-baseline[(seed,epoch)]})
                print(json.dumps({'seed':seed,'condition':condition,'pass':epoch,'dev_exact':score['exact'],'byte_accuracy':score['teacher_forced_byte_accuracy'],'seconds':time.monotonic()-t0}),flush=True)
        np.testing.assert_array_equal(m.core.w_in,wi);np.testing.assert_array_equal(m.core.w_rec,wr)
        checkpoint=artifact/f'{condition}-{seed}.zip';checkpoint.write_bytes(m.dumps());cold=run/f'{condition}-{seed}-cold.json'
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--probe',str(checkpoint),'--output',str(cold)],check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        assert json.loads(cold.read_text(encoding='utf-8'))==stages[-1]['development']
        result={'seed':seed,'condition':condition,'initial':initial,'stages':stages,'train_final':m.score(train),
          'zero_snapshot':m.score(dev,'zero'),'swapped_snapshot':m.score(dev,'swap'),'cold_equal':True,
          'checkpoint':checkpoint.relative_to(root).as_posix(),'hidden_changed':False,'extra_readout_parameters':268*48,
          'held_activation_bytes':m.held.nbytes,'updates':m.core.updates,'seconds':time.monotonic()-t0}
        results.append(result);write_json(run/'progress.json',results)
      summary={}
      for condition in ('no_snapshot','snapshot_raw','snapshot_normalized'):
        rs=[r for r in results if r['condition']==condition]
        summary[condition]={'development_exact':float(np.mean([r['stages'][-1]['development']['exact'] for r in rs])),
          'train_exact':float(np.mean([r['train_final']['exact'] for r in rs])),
          'byte_accuracy':float(np.mean([r['stages'][-1]['development']['teacher_forced_byte_accuracy'] for r in rs])),
          'zero_snapshot_exact':float(np.mean([r['zero_snapshot']['exact'] for r in rs])),'swapped_snapshot_exact':float(np.mean([r['swapped_snapshot']['exact'] for r in rs])),
          'gain_vs_no_snapshot':float(np.mean([r['stages'][-1]['gain_vs_no_snapshot'] for r in rs]))}
      useful=any(v['gain_vs_no_snapshot']>=.10 and v['development_exact']>v['zero_snapshot_exact'] for c,v in summary.items() if c!='no_snapshot')
      result={'run':run.relative_to(root).as_posix(),'summary':summary,'trials':results,'resources':monitor.report(),'cap':cap,'deployed':False,'normal_chat_vision_proven':False,
        'retention_proven':False,'hidden_representation_gain':False,'useful_screen':useful,'decision':'REGISTER_CONFIRMATION_NO_DEPLOY' if useful else 'STOP_BYTE_DIALOGUE_FAMILY_AFTER_V2'}
      write_json(run/'result.json',result);write_json(root/'artifacts/byte-working-memory-latest.json',result)
      (run/'report.md').write_text('# Held-neural working-memory byte trial\n\nSame48ALIF population with48held activation values and extra readout synapses; no hidden updates. A snapshot is not learned long-term memory.\n\n'+json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2)+'\n',encoding='utf-8')
      print('RESULT '+str(run),flush=True)


if __name__=='__main__':main()
