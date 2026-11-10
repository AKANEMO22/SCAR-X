import numpy as np
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.dialogue_probe import prompt_features,inspect_example,ridge,local_probe


def test_prompt_representation_does_not_depend_on_target_and_diagnostic_leak_is_explicit():
    model=UnifiedCore(['<eos>','<bos>','hello','bye','.']);payload=model.dumps()
    _,first,_=prompt_features(payload,'hi')
    a,fa=inspect_example(payload,'hi','hello.');b,fb=inspect_example(payload,'hi','bye.')
    np.testing.assert_array_equal(first,fa['prompt_hidden']);np.testing.assert_array_equal(fa['prompt_hidden'],fb['prompt_hidden'])
    assert 'gold_teacher_hidden_leaky' in fa and fa['prompt_hidden'].shape==fb['gold_teacher_hidden_leaky'].shape
    assert model.dumps()==payload and a['vocabulary_expressible'] and np.isfinite(a['teacher_forced_nll'])
    x=np.tile(np.eye(8,dtype=np.float32),(5,1));y=np.tile(np.arange(8),5)
    assert ridge(x,y,x,y)['accuracy']==1 and local_probe(x,y,x,y,1901)['accuracy']==1
