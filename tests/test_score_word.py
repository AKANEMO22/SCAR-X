import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.neural.paired_score import PairedScoreCore
from scarx.neural.masked_score import MaskedScoreCore
from scarx.evaluation.score_word import ScoreWord


def test_mask_adapter_matches_original_when_all_allowed_and_rejects_target_before_mutation():
    c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12);a=PairedScoreCore(c);b=MaskedScoreCore(c)
    x=np.random.default_rng(1001).normal(0,.2,(12,8)).astype(np.float32)
    features=a.features_many(np.stack((x,x+.01)),91);a.calibrate(features);b.calibrate(features)
    assert a.learn_example(x,1)==b.learn_example(x,1)
    for name,value in a.arrays().items():np.testing.assert_array_equal(value,b.arrays()[name])
    b.output_mask[1]=False;state=b.dumps()
    with pytest.raises(ValueError):b.learn_example(x,1)
    assert b.dumps()==state
    with pytest.raises(ValueError):PairedScoreCore.loads(state)


def test_prefix_has_no_current_target_and_word_codec_continues_neural_learning_exactly():
    model=ScoreWord(['<eos>','<bos>','hello','bye','.'],27103,'full')
    x=model.sequence('hi',[]);assert x.shape==(15,model.core.config.inputs)
    assert not x[:,model.ids['hello']].any() and not x[:,model.ids['bye']].any()
    model.calibrate(['hi','bye']);model.learn('hi','hello.')
    clone=ScoreWord.loads(model.dumps());assert clone.dumps()==model.dumps()
    assert model.learn('bye','bye.')==clone.learn('bye','bye.') and model.dumps()==clone.dumps()
    assert model.generate('hi',991)==clone.generate('hi',991)
    before=model.core.w_in.copy();frozen=ScoreWord(['<eos>','<bos>','hello','.'],27109,'frozen');frozen.calibrate(['hi'])
    old=frozen.core.w_in.copy();frozen.learn('hi','hello.');np.testing.assert_array_equal(old,frozen.core.w_in)
