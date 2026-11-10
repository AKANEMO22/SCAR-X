import io
import numpy as np
import pytest
from PIL import Image
from scarx.neural.input_events import InputEvents


def test_order_speaker_and_no_target_channel_survive_compilation():
    enc=InputEvents(32)
    a=enc.text('Start. Code is 1. Code is 2. End.')
    b=enc.text('Start. Code is 2. Code is 1. End.')
    assert a.shape==b.shape and a.shape[1]==230 and not np.array_equal(a,b)
    assert np.all(a[:,:32]==0)
    np.testing.assert_allclose(a.sum(0),b.sum(0),atol=1e-6)
    dialogue=enc.dialogue([{'role':'user','text':'hi'},{'role':'assistant','text':'hello'}])
    assert 0 in dialogue[:,-2] and 1 in dialogue[:,-2]
    with pytest.raises(ValueError):enc.text('word '*193)
    with pytest.raises(ValueError):enc.dialogue([{'role':'user','text':'a '*100}]*4)


def test_color_and_spatial_positions_survive_same_luminance_images():
    enc=InputEvents(24);outputs=[]
    for color in ((255,40,40),(0,178,0)):
        image=Image.new('RGB',(32,32),color);stream=io.BytesIO();image.save(stream,format='PNG')
        outputs.append(enc.image(stream.getvalue()))
    assert outputs[0].shape==(18,222) and not np.array_equal(*outputs)
    assert np.all(outputs[0][:,:enc.image_start]==0)
    np.testing.assert_array_equal(outputs[0][1,enc.image_start+48:enc.image_start+50],[0,0])
    np.testing.assert_array_equal(outputs[0][-2,enc.image_start+48:enc.image_start+50],[1,1])
    assert np.isfinite(outputs[0]).all()
