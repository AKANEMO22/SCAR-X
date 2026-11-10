import io,json,zipfile
import numpy as np
import pytest
from scarx.evaluation.fixed_word_codes import fit,context,coverage,validate,vector
from scarx.evaluation.score_word import ScoreWord


def test_input_inventory_is_train_prompt_only_all_words_and_fixed128_dimensions():
    prompts=[' '.join('word'+str(i) for i in range(150)),'word0 word1']
    dense=fit(prompts,'known-digest128-v1');sparse=fit(prompts,'known-sparse128-v1')
    assert dense['idf']==sparse['idf'] and len(dense['idf'])==150
    assert 'unseen' not in dense['idf'] and dense['idf']['word0']==1
    for metadata in (dense,sparse):
        a=context(metadata,'word0 unseen word149');b=context(metadata,'word0 word149')
        np.testing.assert_array_equal(a,b);assert a.shape==(128,) and a.dtype==np.float32
        assert not context(metadata,'unseen').any() and not context(metadata,'?!').any()
        assert coverage(metadata,['word0 unseen'])['mean_known_word_fraction']==.5
        with pytest.raises(ValueError):validate({**metadata,'idf':{'unseen':float('nan')}})
        with pytest.raises(ValueError):validate({**metadata,'target':'not allowed'})
    assert len(set(tuple(vector(t,dense['kind'])) for t in dense['idf']))==150
    assert np.all(vector('word0',dense['kind'])!=0) and not vector('word0',dense['kind']).flags.writeable


def test_new_numeric_input_keeps_weight_shapes_and_full_future_learning_codec():
    vocab=['<eos>','<bos>','hello','bye','.'];encoding=fit(['hi','bye'],'known-digest128-v1')
    legacy=ScoreWord(vocab,28001,'full');model=ScoreWord(vocab,28001,'full',encoding)
    for name,value in legacy.core.arrays().items():np.testing.assert_array_equal(value,model.core.arrays()[name])
    assert legacy.core.memory_bytes()==model.core.memory_bytes()
    x=model.sequence('hi',[]);assert not x[:,model.ids['hello']].any() and not x[:,model.ids['bye']].any()
    model.calibrate(['hi','bye']);model.learn('hi','hello.')
    restored=ScoreWord.loads(model.dumps());assert restored.dumps()==model.dumps() and restored.encoding==encoding
    assert model.learn('bye','bye.')==restored.learn('bye','bye.') and restored.dumps()==model.dumps()
    assert model.generate('hi')==restored.generate('hi') and restored.dumps()==model.dumps()
    frozen=ScoreWord(vocab,28001,'frozen',encoding);frozen.calibrate(['hi']);old=frozen.core.w_in.copy();frozen.learn('hi','hello.')
    np.testing.assert_array_equal(old,frozen.core.w_in)
    with zipfile.ZipFile(io.BytesIO(legacy.dumps())) as z:assert 'encoding' not in json.loads(z.read('metadata.json'))
    with zipfile.ZipFile(io.BytesIO(model.dumps())) as z:core=z.read('core.npz');meta=json.loads(z.read('metadata.json'))
    meta['encoding']['idf']['hi']=2.;tampered=io.BytesIO()
    with zipfile.ZipFile(tampered,'w') as z:z.writestr('core.npz',core);z.writestr('metadata.json',json.dumps(meta))
    with pytest.raises(ValueError,match='encoding checksum'):ScoreWord.loads(tampered.getvalue())
