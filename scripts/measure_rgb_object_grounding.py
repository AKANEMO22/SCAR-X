"""Tiny real-photo grounding precondition; one fixed ALIF core, no language claim."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,io
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from PIL import Image
from scarx.neural.input_events import InputEvents
from scarx.evaluation.byte_dialogue import ByteBackbone
from scarx.evaluation.readout_conditioning import ConditionedReadout
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.rgb_grounding import different_class_pairing
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor

NAMES=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')


def main():
    cap=apply_memory_limit();receipt=json.loads((root/'artifacts/rgb-object-dataset.json').read_text(encoding='utf-8'))
    path=Path(receipt['local_batch']);payload=path.read_bytes();assert hashlib.sha256(payload).hexdigest()==receipt['batch_sha256']
    records=np.frombuffer(payload,np.uint8).reshape(10000,3073);labels=records[:,0];assert labels.max()<=9
    pixels=records[:,1:].reshape(-1,3,32,32).transpose(0,2,3,1);rng=np.random.default_rng(10103)
    train=[];dev=[]
    for label in range(10):
        indices=rng.permutation(np.flatnonzero(labels==label));train+=indices[:64].tolist();dev+=indices[64:80].tolist()
    assert not set(train)&set(dev)
    train_hash={hashlib.sha256(pixels[i].tobytes()).hexdigest() for i in train}
    overlap=[i for i in dev if hashlib.sha256(pixels[i].tobytes()).hexdigest() in train_hash]
    assert not overlap,'Duplicate training/development image content detected; stop without repairing split'
    run=new_run(root,'rgb-object-grounding-v1',{'protocol':'experiments/rgb-object-grounding-v1-source-amendment.md','split_seed':10103,'network_seeds':[10111,10133,10139],'train':640,'development':160,'hidden':48,'hidden_updates':0},data_description='Small CIFAR10 pinnedUofT-CS first10000HFtrainingrows; source labels, no redistributed/privateimages or originaltest')
    write_json(run/'data-split.json',{'train_indices':train,'development_indices':dev,'label_names':NAMES,'batch_sha256':receipt['batch_sha256'],'train_dev_hash_overlap':overlap,'original_test_opened':False})
    encoder=InputEvents(256);text=encoder.text('What is in the photo?');events={};selected=train+dev
    for i in selected:
        stream=io.BytesIO();Image.fromarray(pixels[i]).save(stream,format='PNG');events[i]=encoder.image(stream.getvalue())
    began=time.monotonic();results=[]
    with ResourceMonitor(interval=.1) as monitor:
      for seed in (10111,10133,10139):
        model=ByteBackbone(seed);fingerprint=model.core.fingerprint()
        def features(indices,blank=False):
            output=[]
            for i in indices:
                model.core.reset_state()
                for row in text:model.tick(row)
                patches=events[i].copy()
                if blank:patches[:,encoder.image_start:encoder.image_start+48]=0
                values=[]
                for step,row in enumerate(patches):
                    model.tick(row)
                    if 1<=step<=16:values.append(model.core.features.copy())
                output.append(np.mean(values,axis=0))
            return np.asarray(output,np.float32)
        f=features(train);tf=features(dev);blank=features(dev,True);shuffled=tf[different_class_pairing(labels[dev])]
        y=labels[train];ty=labels[dev]
        for mode in ('raw','normalized'):
            head=ConditionedReadout(model.core.w_out[:10],model.core.bias[:10],mode)
            for row in f:head.observe(row)
            head.freeze();order_rng=np.random.default_rng(seed);loss=[]
            for epoch in range(4):
                for i in order_rng.permutation(len(train)):loss.append(head.learn(f[i],int(y[i])))
            p=np.argmax(head.probabilities(tf),axis=1);pb=np.argmax(head.probabilities(blank),axis=1);ps=np.argmax(head.probabilities(shuffled),axis=1)
            confusion=np.zeros((10,10),int)
            for a,b in zip(ty,p):confusion[int(a),int(b)]+=1
            correct=float(np.mean(p==ty));blank_acc=float(np.mean(pb==ty));swapped_acc=float(np.mean(ps==ty))
            result={'seed':seed,'condition':mode,'accuracy':correct,'train_accuracy':float(np.mean(np.argmax(head.probabilities(f),axis=1)==y)),
              'blank_accuracy':blank_acc,'mismatched_image_accuracy':swapped_acc,'gain_over_blank':correct-blank_acc,'gain_over_mismatched':correct-swapped_acc,
              'per_class_accuracy':{NAMES[i]:float(np.mean(p[ty==i]==i)) for i in range(10)},'updates':head.updates,'hidden_changed':model.core.fingerprint()!=fingerprint,
              'mean_train_loss':float(np.mean(loss)),'informative':correct>=.30 and correct-blank_acc>=.10 and correct-swapped_acc>=.10}
            assert not result['hidden_changed']
            results.append(result);write_json(run/f'{mode}-{seed}.json',{'summary':result,'development_indices':dev,'targets':ty.tolist(),'predictions':p.tolist(),'blank_predictions':pb.tolist(),'mismatched_predictions':ps.tolist(),'confusion_matrix':confusion.tolist()})
            print(json.dumps(result),flush=True)
        monitor.check()
        if time.monotonic()-began>120:raise TimeoutError('Registered120s vision budget')
      summary={mode:{key:float(np.mean([r[key] for r in results if r['condition']==mode])) for key in ('accuracy','train_accuracy','blank_accuracy','mismatched_image_accuracy','gain_over_blank','gain_over_mismatched')} for mode in ('raw','normalized')}
      informative=any(v['accuracy']>=.30 and v['gain_over_blank']>=.10 and v['gain_over_mismatched']>=.10 for v in summary.values())
      result={'run':run.relative_to(root).as_posix(),'summary':summary,'trials':results,'source_receipt':{k:v for k,v in receipt.items() if k!='local_batch'},
        'resources':monitor.report(),'cap':cap,'image_path_informative_screen':informative,'normal_photo_qa_proven':False,'learned_caption_or_language_proven':False,
        'class_name_formatting_is_not_language_generation':True,'hidden_representation_learning':False,'deployed':False,'original_test_used':False,
        'decision':'REGISTER_INDEPENDENT_VISION_CONFIRMATION' if informative else 'PIVOT_VISUAL_REPRESENTATION_NO_SCALE'}
      write_json(run/'result.json',result);write_json(root/'artifacts/rgb-object-grounding-latest.json',result)
      (run/'report.md').write_text('# RGB real-object diagnostic\n\nSmall independent-image classification, fixed48ALIFcore, source dataset labels. Not normal vision, captioning or learned hidden representation.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
      print('RESULT '+str(run),flush=True)


if __name__=='__main__':main()
