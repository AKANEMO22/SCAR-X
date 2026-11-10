import io,json,zipfile
import numpy as np
import pytest
from scarx.evaluation.visual_words import VisualWords
from scarx.evaluation.vision_fixtures import scenes,render
from scarx.neural.unified import UnifiedCore

VOCAB=['<eos>','<bos>','red','green','circle','square','hình','tròn']


def prepared():
    m=VisualWords(VOCAB,33113);image=render(scenes()[0]);m.calibrate([{'prompt':'What color?','image':image}]);return m,image


def test_recording_control_is_exact_local_reference_and_rejects_target_forward():
    from scarx.evaluation.controls import VisualWordLearnabilityControl,torch
    torch.set_num_threads(1);a,image=prepared();b=VisualWords.loads(a.dumps());control=VisualWordLearnabilityControl(b)
    for question,target in [('What color?','red'),('What shape?','circle'),('Màu gì?','green')]:
        assert a.learn(question,image,target)==control.learn(question,image,target,train_hidden=False)
        assert a.dumps()==control.reference.dumps()
    before=control.dumps()
    with pytest.raises(TypeError):control.generate('What color?',image,target='red')
    assert control.dumps()==before and control.backward_passes==0


def test_isolated_control_updates_hidden_with_parity_and_exact_optimizer_resume():
    from scarx.evaluation.controls import VisualWordLearnabilityControl,torch
    torch.set_num_threads(1);m,image=prepared();initial=m.core.w_in.copy();stable=m.core.stable_in.copy();control=VisualWordLearnabilityControl(m)
    for target in ['red','green','red']:control.learn('What color?',image,target)
    assert np.any(m.core.w_in!=initial) and control.optimizer_steps==3 and control.backward_passes==3
    np.testing.assert_array_equal(stable,m.core.stable_in)
    assert control.max_graph_state_error<=1e-4 and control.max_graph_feature_error<=1e-6 and control.max_graph_loss_error<=1e-4
    m.prepare('What color?',image);m.word_feature(1,0);payload=control.dumps();copy=VisualWordLearnabilityControl.loads(payload)
    assert copy.dumps()==payload
    np.testing.assert_array_equal(m.word_feature(m.ids['red'],1),copy.reference.word_feature(m.ids['red'],1))
    assert control.dumps()==copy.dumps() and control.generate('What color?',image)==copy.generate('What color?',image)
    assert control.learn('What color?',image,'green')==copy.learn('What color?',image,'green') and control.dumps()==copy.dumps()
    assert int(m.sampler_rng.integers(1000))==int(copy.reference.sampler_rng.integers(1000)) and control.dumps()==copy.dumps()
    for loader in (UnifiedCore.loads,VisualWords.loads):
        with pytest.raises(ValueError):loader(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:parts={name:z.read(name) for name in z.namelist()}
    meta=json.loads(parts['control.json']);meta['optimizer_steps']+=1;parts['control.json']=json.dumps(meta).encode();bad=io.BytesIO()
    with zipfile.ZipFile(bad,'w') as z:
        for name,value in parts.items():z.writestr(name,value)
    with pytest.raises(ValueError,match='checksum'):VisualWordLearnabilityControl.loads(bad.getvalue())
