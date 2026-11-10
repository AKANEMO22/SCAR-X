"""Registered UTF8 decoder screen and cold-process audit; private state untouched."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,time,itertools,argparse,subprocess
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.byte_codec import UTF8Codec
from scarx.evaluation.byte_dialogue import ByteDialogue
from scarx.evaluation.runs import new_run,write_json,cluster_interval
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--probe',type=Path);parser.add_argument('--data',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    cap=apply_memory_limit()
    if args.probe:
        rows=[json.loads(l) for l in args.data.read_text(encoding='utf-8').splitlines()];dev=[r for r in rows if r['split']=='development']
        m=ByteDialogue.loads(args.probe.read_bytes());write_json(args.output,m.score(dev));return
    start=time.monotonic();data=root/'data/candidates/antigravity-events-curated-20261109T094142Z'
    rows=[json.loads(l) for l in (data/'experiences.jsonl').read_text(encoding='utf-8').splitlines()]
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='development']
    for r in rows:
        tokens=UTF8Codec.encode(r['target']);assert 1<=len(tokens)<=256 and UTF8Codec.decode(tokens)==r['target']
    conditions=('byte_frozen','byte_raw','byte_normalized','byte_normalized_full');seeds=(6101,6113,6121)
    run=new_run(root,'byte-dialogue-prototype-v1',{'protocol':'experiments/byte-dialogue-prototype-v1.md','seeds':seeds,'conditions':conditions,'passes':4,'hidden':48,'inputs':454,'outputs':268},
      data_description='Same reviewed public Antigravity96train/24devresponses; fixed256byte alphabet, no vocabulary fitting',data_files=[str((data/n).relative_to(root)) for n in ('dialogue.json','manifest.json','validation.json','experiences.jsonl')])
    artifact=root/'artifacts/byte-dialogue'/run.name;artifact.mkdir(parents=True)
    representable={'training':len(train),'development':len(dev),'all_roundtrip_exact':True,'oov_bytes':0,'alphabet_fit_from_data':False,
      'reference_old_word_exact_representable':9,'reference_old_word_dev_total':24,'constraint':'Representable does not mean a neural model can generate a coherent answer'}
    write_json(run/'representability.json',representable);results=[];frozen={}
    with ResourceMonitor(interval=.25) as monitor:
      for condition,seed in itertools.product(conditions,seeds):
        began=time.monotonic();m=ByteDialogue(seed,condition);m.calibrate(train)
        initial=m.score(dev);fingerprint=m.core.fingerprint();stages=[]
        by_dialog={key:[r for r in train if r['dialogue_id']==key] for key in sorted({r['dialogue_id'] for r in train})}
        for epoch in range(1,5):
            losses=[]
            for key in m.order_rng.permutation(list(by_dialog)):
                for r in by_dialog[key]:losses.append(m.learn(r['messages'],r['target']))
            m.passes_done=epoch;monitor.check()
            if time.monotonic()-began>180:raise TimeoutError('Registered180s condition budget')
            if time.monotonic()-start>1200:raise TimeoutError('Registered1200s round budget')
            if epoch in (2,4):
                score=m.score(dev)
                if condition=='byte_frozen':frozen[(seed,epoch)]=score['exact']
                stage={'pass':epoch,'train_loss':float(np.mean(losses)),'development':score,
                  'gain_vs_frozen':score['exact']-frozen[(seed,epoch)],'seconds':time.monotonic()-began}
                stages.append(stage);write_json(run/f'{condition}-{seed}-pass-{epoch}.json',stage)
                print(json.dumps({'condition':condition,'seed':seed,'pass':epoch,'dev_exact':score['exact'],'byte_accuracy':score['teacher_forced_byte_accuracy'],'nonempty':score['nonempty_rate'],'seconds':time.monotonic()-began}),flush=True)
        checkpoint=artifact/f'{condition}-{seed}.zip';checkpoint.write_bytes(m.dumps())
        cold=run/f'{condition}-{seed}-cold.json'
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--probe',str(checkpoint),'--data',str(data/'experiences.jsonl'),'--output',str(cold)],check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        assert json.loads(cold.read_text(encoding='utf-8'))==stages[-1]['development']
        if condition=='byte_frozen':assert m.core.fingerprint()==fingerprint
        row={'condition':condition,'seed':seed,'initial':initial,'stages':stages,'train_final':m.score(train),'checkpoint':checkpoint.relative_to(root).as_posix(),
          'cold_process_exact':True,'responses_seen':m.responses_seen,'synaptic_updates':m.core.updates,'head_updates':m.head.updates,
          'arrays_bytes':m.core.memory_bytes(),'current_weight_parameters':sum(getattr(m.core,n).size for n in ('w_in','w_rec','w_out','bias')),
          'stopped_for_no_dev_gain':condition!='byte_frozen' and all(s['gain_vs_frozen']<.10 for s in stages),'seconds':time.monotonic()-began}
        results.append(row);write_json(run/'progress.json',results)
      summary={}
      for condition in conditions:
        rs=[r for r in results if r['condition']==condition]
        summary[condition]={k:float(np.mean([r['stages'][-1]['development'][k] for r in rs])) for k in ('exact','raw_exact','utf8_valid_rate','nonempty_rate','repetition_rate','byte_nll','teacher_forced_byte_accuracy')}
        summary[condition]['train_exact']=float(np.mean([r['train_final']['exact'] for r in rs]))
      effects={}
      for a,b,label in [('byte_normalized','byte_raw','normalization'),('byte_normalized_full','byte_normalized','hidden')]:
        effects[label]=cluster_interval([next(r for r in results if r['seed']==s and r['condition']==a)['stages'][-1]['development']['exact']-next(r for r in results if r['seed']==s and r['condition']==b)['stages'][-1]['development']['exact'] for s in seeds])
      result={'run':run.relative_to(root).as_posix(),'protocol':'experiments/byte-dialogue-prototype-v1.md','data':data.relative_to(root).as_posix(),
        'representability':representable,'summary':summary,'effects':effects,'trials':results,'resources':monitor.report(),'cap':cap,
        'historical_word_trial_not_compute_matched':True,'syntax_validity_is_host_enforced':True,'normal_chat_or_vision_proven':False,'retention_proven':False,
        'deployed':False,'decision':'STOP_BYTE_DIALOGUE_V1' if all(r['stopped_for_no_dev_gain'] for r in results if r['condition']!='byte_frozen') else 'DEVELOPMENT_ONLY_NO_DEPLOY'}
      write_json(run/'result.json',result);write_json(root/'artifacts/byte-dialogue-latest.json',result)
      lines=['# Fixed UTF8 byte dialogue prototype','','Codec removes word-dictionary OOV, but valid UTF8 is enforced by a deterministic host mask. Semantic task success and lifelong retention remain separate.','',
        '| Condition | Dev exact | Train exact | Forced byte accuracy | Nonempty |','|---|---:|---:|---:|---:|']
      for c,s in summary.items():lines.append(f"| {c} | {s['exact']:.4f} | {s['train_exact']:.4f} | {s['teacher_forced_byte_accuracy']:.4f} | {s['nonempty_rate']:.4f} |")
      lines+=['',json.dumps({k:v for k,v in result.items() if k not in ('trials','summary')},indent=2),'','Historical word-vs-byte compute and output dimensions differ. Same48ALIFneuron topology; no private session/weights replaced.']
      (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
      print('RESULT '+str(run),flush=True)


if __name__=='__main__':main()
