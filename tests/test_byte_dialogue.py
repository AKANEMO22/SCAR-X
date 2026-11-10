import numpy as np
import pytest
from scarx.neural.byte_codec import UTF8Codec,UTF8Cursor
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.byte_dialogue import ByteDialogue


def test_fixed_codec_handles_unseen_unicode_and_rejects_invalid_utf8():
    texts=['Xin chào bạn.','-17','new_word_Δ','🧠🌿','漢字','e\u0301','\U0010ffff','\u0080','\u07ff','\u0800','\ud7ff','\ue000']
    for text in texts:
        tokens=UTF8Codec.encode(text);cursor=UTF8Cursor()
        for i,t in enumerate(tokens):assert cursor.mask(len(tokens)-i)[t];cursor.accept(t)
        assert cursor.remaining==0 and UTF8Codec.decode(tokens)==text and cursor.mask(1)[0]
    assert not UTF8Cursor().mask(1)[0]
    for raw in (b'\xc0\x80',b'\xed\xa0\x80',b'\xf4\x90\x80\x80',b'\x80',b'\xe1'):
        with pytest.raises(ValueError):UTF8Codec.decode([b+2 for b in raw])
    with pytest.raises(ValueError):UTF8Codec.encode('\ud800')
    with pytest.raises(ValueError):UTF8Codec.encode('\x00')
    assert not UTF8Cursor().mask(1)[0xe1+2]
    assert not UTF8Cursor().mask(2)[0xf0+2]


def test_full_core_head_rng_resume_and_no_inference_learning():
    m=ByteDialogue(6101,'byte_normalized_full');example={'messages':[{'role':'user','text':'chào'}],'target':'Chào.'}
    m.calibrate([example]);m.learn(example['messages'],example['target'])
    clone=ByteDialogue.loads(m.dumps());assert m.dumps()==clone.dumps()
    np.testing.assert_array_equal(m.order_rng.permutation(12),clone.order_rng.permutation(12))
    m.learn(example['messages'],'-1');clone.learn(example['messages'],'-1')
    assert m.dumps()==clone.dumps()
    fingerprint=m.core.fingerprint()
    for budget in (1,2,3,4,7):
        output=m.generate(example['messages'],max_bytes=budget)
        assert output['utf8_valid'] and len(output['text'].encode())<=budget
    assert fingerprint==m.core.fingerprint()
    with pytest.raises((ValueError,KeyError)):UnifiedCore.loads(m.dumps())
