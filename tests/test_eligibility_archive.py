import hashlib,json
from scarx.app.observatory import eligibility_answer_archive
from scarx.evaluation.tasks21 import representation


def test_archive_accepts_only_registered_exact_public_tensors_and_consistent_neural_argmax(tmp_path):
    run=tmp_path/'runs'/'source';run.mkdir(parents=True)
    (run/'manifest.json').write_text(json.dumps({'config':{'protocol':'experiments/direct-readout-eligibility-v1.md','proposal30_approved':True,'examples_per_trial':1024}}),encoding='utf-8')
    x,y=representation('xor',35113+1390000,256);outputs=[{'id':f'development-xor-35113-correct/{i}','input':seq.tolist(),'target':int(label),'prediction':0,'probabilities':[.6,.4]} for i,(seq,label) in enumerate(zip(x,y))]
    record={'phase':'development','task':'xor','seed':35113,'method':'correct','experiences':1024,'head_updates':1024,'proposal30_approved':True,
        'test_inputs_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'test_targets_sha256':hashlib.sha256(y.astype('int32').tobytes()).hexdigest()}
    path=run/'development-xor-35113-correct.json';data={'record':record,'outputs':outputs};path.write_text(json.dumps(data),encoding='utf-8')
    r=eligibility_answer_archive(tmp_path,'source');assert r['case_count']==256 and r['learning_mechanism']=='direct_readout_eligibility' and r['method']=='correct'
    assert not eligibility_answer_archive(tmp_path,'../../private')
    outputs[0]['prediction']=1;path.write_text(json.dumps(data),encoding='utf-8');assert not eligibility_answer_archive(tmp_path,'source')
    outputs[0]['prediction']=0;outputs[0]['input'][0][0]+=1;path.write_text(json.dumps(data),encoding='utf-8');assert not eligibility_answer_archive(tmp_path,'source')
