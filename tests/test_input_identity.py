import numpy as np
from scarx.evaluation.input_identity import fit,transform,dual_ridge
from scarx.evaluation.dialogue_probe import ridge


def test_vocabulary_is_train_only_fixed_width_and_dual_matches_existing_ridge():
    train=['hello friend','goodbye friend','thank you','your name']
    metadata=fit(train);assert not metadata['selection_uses_labels'] and 'unseen' not in metadata['vocabulary']
    x,c=transform(metadata,train,'dictionary128');test,coverage=transform(metadata,['unseen'], 'dictionary128')
    assert x.shape==(4,128) and not test.any() and coverage['all_unknown_prompts']==1
    hashed,_=transform(metadata,train,'hash_same_dictionary128');assert hashed.shape==x.shape
    y=np.arange(4);np.testing.assert_allclose(dual_ridge(x,y,x,y,4)['accuracy'],ridge(x,y,x,y,4)['accuracy'])
