"""Registered load ladder; isolated clones, no deployed weights changed."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,json,sys,time,subprocess,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
import psutil
from scarx.neural.unified import UnifiedCore
from scarx.app.generation import Generation
from scarx.evaluation.knowledge_capacity import dataset,probe,fraction,frozen_slot,retention,forgetting
from scarx.evaluation.runs import new_run,write_json,cluster_interval
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe',type=Path);parser.add_argument('--items',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.probe:
        apply_memory_limit();m=UnifiedCore.loads(args.probe.read_bytes())
        write_json(args.output,{'fingerprint':m.core.fingerprint(),'outputs':probe(m,json.loads(args.items.read_text(encoding='utf-8')))})
        return
    cap=apply_memory_limit()
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    conditions=('frozen','full','readout','full_replay','guarded');seeds=(17,29,43)
    run=new_run(root,'knowledge-load-retention-v1r1',{'protocol':'experiments/knowledge-load-retention-v1-amendment.md','order_seeds':seeds,'independent_initializations':1,'conditions':conditions,'max_lessons':128,'slots':32,'hidden':48,'nonnegative_only':True},data_description='Closed-world deterministic nonnegative arithmetic128curriculum/32unseen/8distractors; no personal data',data_files=['data/dialogue/generation1.json'])
    rows=dataset(nonnegative_only=True);train=rows[:128];unseen=rows[128:160];distractors=rows[160:]
    write_json(run/'data.json',rows)
    artifact=root/'artifacts/knowledge-retention'/run.name;artifact.mkdir(parents=True)
    release=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'));initial=(root/release['checkpoint']).read_bytes()
    assert hashlib.sha256(initial).hexdigest()==release['sha256']
    write_json(run/'starting-release.json',release)
    base=UnifiedCore.loads(initial);baseline=probe(base,train);write_json(run/'baseline-predictions.json',baseline)
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
    old=[{'id':f'old-{i}-{j}','prompt':p,'paraphrase':p,'target':f['answer']} for i,f in enumerate(corpus['families']) for j,p in enumerate(f['development'])]
    old_base=probe(base,old);write_json(run/'old-baseline.json',old_base)
    start=time.monotonic();results=[];frozen_scores={}
    with ResourceMonitor(interval=.1) as monitor:
      for condition in conditions:
       for seed in seeds:
        if time.monotonic()-start>720:break
        rng=np.random.default_rng(seed);order=list(rng.permutation(len(train)));m=UnifiedCore.loads(initial);app=None
        if condition=='guarded':
            app=Generation(root/'local'/run.name/f'guarded-{seed}',background=False);app.dialogue.ask_enabled=False;m=app.model
        stages=[];immediate=[];presented=[];previous=0;failures=0;why='maximum128 reached';began=time.monotonic();slots=0
        initial_updates=m.core.updates
        old_best=[fraction([x for x in old_base if x['id'].startswith(f'old-{i}-')]) for i in range(8)]
        try:
          for n,index in enumerate(order,1):
            row=train[index];m.expand_vocabulary(row['target'])
            if condition=='guarded':
                app.dialogue.enqueue('capacity-qa',row['prompt'],row['target'],row['verified_by'])
                for _ in range(32):
                    if not any(l['status']=='queued' and l['remaining']>0 for l in app.dialogue.lessons):break
                    app._online_microstep('registered_capacity_diagnostic');slots+=1
            else:
                for k in range(32):
                    example=train[int(rng.choice(presented))] if condition=='full_replay' and presented and k%4==3 else row
                    if condition=='frozen':frozen_slot(m,example['prompt'],example['target'])
                    else:m.train_pair(example['prompt'],example['target'],'readout' if condition=='readout' else 'full')
                    slots+=1
            immediate+=probe(m,[row]);presented.append(int(index));monitor.check()
            if n in (8,16,32,64,128):
                current_rows=[train[i] for i in presented];current=probe(m,current_rows);old_now=probe(m,old)
                block=fraction(current[previous:n]);frozen=frozen_scores.get((seed,n),0.)
                if condition=='frozen':frozen_scores[(seed,n)]=block;frozen=block
                loss=max(0.,fraction(old_base)-fraction(old_now))
                family_now=[]
                for i in range(8):
                    b=[x for x in old_now if x['id'].startswith(f'old-{i}-')]
                    family_now.append(fraction(b))
                forgotten=forgetting(old_best,family_now);old_best=forgotten['next_best']
                gain=block-frozen
                stage={'lessons_presented':n,'trained_prompt_exact':fraction(probe(m,current_rows,'prompt')),
                  'unseen_wording_exact':fraction(current),'current_block_exact':block,'current_block_gain_vs_frozen':gain,
                  'old_skills_exact':fraction(old_now),'old_mean_forgetting_pp':100*forgotten['mean'],'old_worst_family_forgetting_pp':100*forgotten['worst'],'old_net_accuracy_drop_pp':100*loss,
                  'retention':retention(immediate,baseline,current),'slots':slots,'actual_neural_updates':m.core.updates-initial_updates,'seconds':time.monotonic()-began}
                stages.append(stage);write_json(run/f'{condition}-{seed}-stage-{n}.json',{'summary':stage,'current':current,'old':old_now,'immediate':immediate})
                print(json.dumps({'condition':condition,'order_seed':seed,**stage}),flush=True)
                failures=failures+1 if gain<.10 else 0;previous=n
                if condition!='frozen' and failures>=2:why='two successive acquisition failures';break
            if time.monotonic()-began>180:why='180s condition limit';break
            if time.monotonic()-start>720:why='720s overall limit';break
          current_rows=[train[i] for i in presented];outputs=probe(m,current_rows);path=artifact/f'{condition}-{seed}.zip';path.write_bytes(m.dumps())
          selected=run/f'{condition}-{seed}-probe-items.json';write_json(selected,current_rows)
          restart=run/f'{condition}-{seed}-restart.json'
          subprocess.run([sys.executable,str(Path(__file__).resolve()),'--probe',str(path),'--items',str(selected),'--output',str(restart)],check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
          restored=json.loads(restart.read_text(encoding='utf-8'));assert restored['outputs']==outputs and restored['fingerprint']==m.core.fingerprint()
          branches={}
          for branch in ('quiet','interference','interference_replay'):
            b=UnifiedCore.loads(m.dumps());ticks=b.ticks;updates=b.core.updates;t0=time.monotonic()
            if branch=='quiet':
                for _ in range(10000):b.tick()
            else:
                for row in distractors:
                    b.expand_vocabulary(row['target'])
                    for k in range(32):
                        item=train[int(rng.choice(presented))] if branch=='interference_replay' and k%4==3 else row
                        b.train_pair(item['prompt'],item['target'],'readout' if condition=='readout' else 'full') if condition!='frozen' else frozen_slot(b,item['prompt'],item['target'])
            after=probe(b,current_rows);branches[branch]={'accuracy':fraction(after),'retention':retention(immediate,baseline,after),'ticks':b.ticks-ticks,'updates':b.core.updates-updates,'seconds':time.monotonic()-t0}
            write_json(run/f'{condition}-{seed}-{branch}.json',{'summary':branches[branch],'outputs':after})
          item={'condition':condition,'order_seed':seed,'initialization_seed':release['seed'],'lessons_presented':len(presented),'stopping_reason':why,
            'stages':stages,'final_exact':fraction(outputs),'stream_exact':fraction(probe(m,current_rows,independent=False)),
            'unseen_expressions_exact':fraction(probe(m,unseen)),'retention':retention(immediate,baseline,outputs),'branches':branches,
            'checkpoint':path.relative_to(root).as_posix(),'restart_exact':True,'slots':slots,'updates':m.core.updates-initial_updates,
            'guard_accepted':app.dialogue.accepted_updates if app else None,'guard_rejected':app.dialogue.rejected_updates if app else None,
            'hidden':m.core.config.hidden,'seconds':time.monotonic()-began}
          results.append(item);write_json(run/'progress.json',results)
        finally:
          if app:app.close()
      # Shared early milestone only: no comparisons of differently truncated loads.
      effects={}
      for condition in ('full','readout','full_replay','guarded'):
          values=[s['current_block_gain_vs_frozen'] for r in results if r['condition']==condition for s in r['stages'] if s['lessons_presented']==16]
          if len(values)==3:effects[condition]=cluster_interval(values)
      result={'run':run.relative_to(root).as_posix(),'protocol':'experiments/knowledge-load-retention-v1-amendment.md','results':results,
        'paired_gain_at16':effects,'order_seeds':list(seeds),'independent_initializations':1,'source_checkpoint':release['checkpoint'],
        'resources':monitor.report(),'hard_cap':cap,'deployed':False,'four_billion_equivalence_tested':False,
        'retention_is_not_calendar_duration':True,'decision':'PIVOT_NO_SCALE' if all(r['stopping_reason']=='two successive acquisition failures' for r in results if r['condition']!='frozen') else 'REVIEW_DIAGNOSTIC_ONLY'}
      write_json(run/'result.json',result);write_json(root/'artifacts/knowledge-retention-latest.json',result)
      lines=['# Knowledge load / retention diagnostic','','One fixed48-neuron public G1 initialization; three curriculum orders. No private weights changed.','',
       '| Condition | Order | Lessons | Worded recall | Acquired | Retained | Old forgetting at last stage | Stop |','|---|---:|---:|---:|---:|---:|---:|---|']
      for r in results:
          last=r['stages'][-1] if r['stages'] else {}
          lines.append(f"| {r['condition']} | {r['order_seed']} | {r['lessons_presented']} | {r['final_exact']:.3f} | {r['retention']['acquired']} | {r['retention']['retained']} | {last.get('old_mean_forgetting_pp')}pp | {r['stopping_reason']} |")
      lines+=['','Retention is undefined when no examples were acquired. Quiet10k ticks and exact restart do not establish days/months retention. Same primitive update slots only for frozen/full/readout/full_replay; guarded has additional guard/SQLite overhead.','',json.dumps({'effects_at16':effects,'resources':result['resources'],'decision':result['decision']},indent=2)]
      (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
      print('RESULT '+str(run),flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
