import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,itertools
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.evaluation.event_dialogue import EventDialogue
from scarx.neural.language import tokenize
from scarx.evaluation.runs import new_run,write_json,cluster_interval
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();start=time.monotonic();pointer=json.loads((root/'artifacts/antigravity-events-latest.json').read_text(encoding='utf-8'))
    path=root/pointer['candidate_directory'];validation=json.loads((path/'validation.json').read_text(encoding='utf-8'));assert validation['schema_and_split_screen_pass']
    records=[json.loads(line) for line in (path/'experiences.jsonl').read_text(encoding='utf-8').splitlines()]
    train=[r for r in records if r['split']=='train'];dev=[r for r in records if r['split']=='development']
    vocab=['<eos>','<bos>']+sorted({t for r in train for t in tokenize(r['target'])});assert len(vocab)<=256
    run=new_run(root,'event-dialogue-prototype-v1',{'protocol':'experiments/event-dialogue-prototype-v1.md','seeds':[5101,5113,5119],'passes':8,'hidden':48,'conditions':['bag_raw','ordered_raw','ordered_normalized','ordered_normalized_full']},data_description='Small reviewed Antigravity dialogue prefixes/next utterance; no teacher weights, private session or finaltest',data_files=[str((path/name).relative_to(root)) for name in ('dialogue.json','manifest.json','validation.json','experiences.jsonl')])
    artifacts=root/'artifacts/event-dialogue'/run.name;artifacts.mkdir(parents=True);results=[]
    with ResourceMonitor(interval=.25) as monitor:
      for seed,condition in itertools.product((5101,5113,5119),('bag_raw','ordered_raw','ordered_normalized','ordered_normalized_full')):
        began=time.monotonic();model=EventDialogue(vocab,seed,condition);model.calibrate(train)
        initial=model.score(dev);rng=np.random.default_rng(seed);stages=[]
        dialogs=sorted({r['dialogue_id'] for r in train});by_dialog={key:[r for r in train if r['dialogue_id']==key] for key in dialogs}
        hidden_before=model.model.core.w_in.copy()
        for epoch in range(1,9):
            loss=[]
            for key in rng.permutation(dialogs):
                for example in by_dialog[key]:loss.append(model.learn(example['messages'],example['target']))
            monitor.check()
            if time.monotonic()-began>90:raise TimeoutError('Registered90s condition limit')
            if time.monotonic()-start>900:raise TimeoutError('Registered900s round limit')
            if epoch in (4,8):
                score=model.score(dev);stages.append({'pass':epoch,'train_loss':float(np.mean(loss)),'development':score})
                print(json.dumps({'seed':seed,'condition':condition,'pass':epoch,'dev_exact':score['exact'],'token_accuracy':score['teacher_forced_token_accuracy'],'seconds':time.monotonic()-began}),flush=True)
        checkpoint=artifacts/f'{condition}-{seed}.zip';checkpoint.write_bytes(model.dumps());clone=EventDialogue.loads(checkpoint.read_bytes())
        assert model.score(dev)==clone.score(dev)
        row={'seed':seed,'condition':condition,'initial':initial,'stages':stages,'train_final':model.score(train),'checkpoint':checkpoint.relative_to(root).as_posix(),
          'hidden_input_delta':float(np.linalg.norm(model.model.core.w_in-hidden_before)),'resume_exact':True,
          'no_exact_gain_after4and8':all(s['development']['exact']-initial['exact']<.10 for s in stages),'seconds':time.monotonic()-began}
        results.append(row);write_json(run/f'{condition}-{seed}.json',row)
      effects={}
      for left,right,label in [('ordered_normalized','ordered_raw','normalization'),('ordered_normalized_full','ordered_normalized','hidden'),('ordered_raw','bag_raw','order')]:
        effects[label]=cluster_interval([next(r for r in results if r['seed']==s and r['condition']==left)['stages'][-1]['development']['exact']-next(r for r in results if r['seed']==s and r['condition']==right)['stages'][-1]['development']['exact'] for s in (5101,5113,5119)])
      result={'run':run.relative_to(root).as_posix(),'data':pointer['candidate_directory'],'training_responses':len(train),'development_responses':len(dev),'training_vocabulary':len(vocab),
        'trials':results,'effects':effects,'resources':monitor.report(),'cap':cap,'live_model_replaced':False,'normal_chat_vision_proven':False,
        'retention_gate_still_failed':True,'decision':'STOP_EVENT_DIALOGUE_V1' if all(r['no_exact_gain_after4and8'] for r in results) else 'DEVELOPMENT_ONLY_REVIEW_NO_DEPLOY'}
      write_json(run/'result.json',result);write_json(root/'artifacts/event-dialogue-latest.json',result)
      (run/'report.md').write_text('# Event dialogue end-to-end prototype\n\nOne48-ALIF core, real word decoding from ordered events, raw/normalized/hidden controls. No serving weights replaced; no natural-photo skill claim.\n\n'+json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2)+'\n',encoding='utf-8')
      print('RESULT '+str(run),flush=True)

if __name__=='__main__':main()
