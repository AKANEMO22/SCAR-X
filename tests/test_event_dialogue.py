import io
import numpy as np
from PIL import Image
from scarx.evaluation.event_dialogue import EventDialogue


def test_one_core_word_readout_resume_and_inference_do_not_learn_or_use_labels():
    vocab=['<eos>','<bos>','hi','hello','.']
    m=EventDialogue(vocab,5101,'ordered_normalized')
    example={'id':'a','messages':[{'role':'user','text':'hi'}],'target':'hello.'}
    m.calibrate([example]);before=m.model.core.fingerprint();m.learn(example['messages'],example['target'])
    assert m.head.weights is m.model.core.w_out and m.head.bias is m.model.core.bias
    assert m.model.core.fingerprint()!=before
    clone=EventDialogue.loads(m.dumps());assert clone.dumps()==m.dumps()
    assert m.generate(example['messages'])==clone.generate(example['messages'])
    for name,array in m.model.core.arrays().items():np.testing.assert_array_equal(array,clone.model.core.arrays()[name])
    saved=m.dumps();m.score([example]);assert m.dumps()==saved
    image=Image.new('RGB',(32,32),(255,40,40));stream=io.BytesIO();image.save(stream,format='PNG')
    ticks=m.model.ticks;fingerprint=m.model.core.fingerprint()
    m.generate([{'role':'user','text':'what is visible?'}],image=stream.getvalue(),max_words=2)
    assert m.model.ticks>=ticks+18 and m.model.core.fingerprint()==fingerprint
