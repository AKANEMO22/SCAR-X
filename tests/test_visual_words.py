import io,json,zipfile
import numpy as np
import pytest
from scarx.evaluation.visual_words import VisualWords
from scarx.evaluation.vision_fixtures import scenes,render
from scarx.evaluation.visual_word_data import scene_split,signature
from scarx.neural.unified import UnifiedCore

VOCAB=['<eos>','<bos>','red','green','circle','square','hình','tròn']


def test_fixed_compositions_have_explicit_noveldev_and_no_labelinput_or_timing_sidechannel():
    train=scene_split('train');dev=scene_split('development');assert len(train)==len(dev)==96
    known={signature(s) for s in train};assert sum(signature(s) not in known for s in dev)==84
    image=render(scenes()[0]);m=VisualWords(VOCAB,33113);inputs=[];original=m.core.step
    def recorder(x,track=True):inputs.append(x.copy());return original(x,track)
    m.core.step=recorder;m.prepare('What color?',image);assert len(inputs)==33
    tape=np.stack(inputs);assert tape[:,len(VOCAB):m.encoder.image_start].any() and tape[:,m.encoder.image_start:m.encoder.image_start+48].any()
    m.prepare('Longer question still same phase length?',image);assert len(inputs)==66
    before=m.dumps()
    with pytest.raises(TypeError):m.generate('What color?',image,target='red')
    assert m.dumps()==before
    for condition,block in [('no_question',slice(len(VOCAB),m.encoder.image_start)),('no_image',slice(m.encoder.image_start,m.encoder.image_start+48))]:
        n=VisualWords(VOCAB,33113,condition);captured=[];actual=n.core.step
        def observe(x,track=True):captured.append(x.copy());return actual(x,track)
        n.core.step=observe;n.prepare('What color?',image);assert not np.stack(captured)[:,block].any() and len(captured)==33


def test_actual_localhead_learning_frozenhidden_and_workingstate_coldcontinuation():
    image=render(scenes()[0]);m=VisualWords(VOCAB,33113);examples=[{'prompt':'What color?','image':image}]
    old=m.core.w_in.copy();rec=m.core.w_rec.copy();m.calibrate(examples)
    assert m.head.count==128 and m.head.frozen
    before=m.core.w_out.copy();m.learn('What color?',image,'red');assert np.any(before!=m.core.w_out)
    np.testing.assert_array_equal(old,m.core.w_in);np.testing.assert_array_equal(rec,m.core.w_rec)
    m.prepare('What color?',image);m.word_feature(1,0);copy=VisualWords.loads(m.dumps());assert m.dumps()==copy.dumps()
    np.testing.assert_array_equal(m.word_feature(m.ids['red'],1),copy.word_feature(copy.ids['red'],1))
    assert m.dumps()==copy.dumps() and m.generate('What color?',image)==copy.generate('What color?',image)
    assert m.learn('What color?',image,'red')==copy.learn('What color?',image,'red') and m.dumps()==copy.dumps()
    assert int(m.sampler_rng.integers(10000))==int(copy.sampler_rng.integers(10000)) and m.dumps()==copy.dumps()
    with pytest.raises(ValueError):UnifiedCore.loads(m.dumps())
    with zipfile.ZipFile(io.BytesIO(m.dumps())) as z:parts={n:z.read(n) for n in z.namelist()}
    meta=json.loads(parts['metadata.json']);meta['condition']='no_image';parts['metadata.json']=json.dumps(meta).encode();bad=io.BytesIO()
    with zipfile.ZipFile(bad,'w') as z:
        for name,value in parts.items():z.writestr(name,value)
    with pytest.raises(ValueError,match='checksum'):VisualWords.loads(bad.getvalue())
