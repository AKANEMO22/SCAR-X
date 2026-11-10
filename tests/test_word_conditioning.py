import numpy as np
from scarx.evaluation.score_word import ScoreWord
from scarx.evaluation.word_conditioning import measure


def test_firstword_intervention_uses_existing_forward_and_preserves_fullstate():
    model=ScoreWord(['<eos>','<bos>','hi','bye','.'],27103,'full')
    prompts=['prompt'+str(i) for i in range(8)];model.calibrate(prompts)
    cases=[{'id':str(i),'intent':str(i),'prompt':p,'target':'hi.' if i%2 else 'bye.'} for i,p in enumerate(prompts)]
    payload=model.dumps();result=measure(payload,cases,991000)
    assert result['model_unchanged'] and model.dumps()==payload and result['cases']==8
    assert result['actual_ticks']==3*8*8*15
    clone=ScoreWord.loads(payload);p=clone.distribution(prompts[0],[],991000)
    isolated=clone.core.probabilities(clone.core.features_many(clone.sequence(prompts[0],[])[None,:,:],991000,samples=8))[0]
    np.testing.assert_array_equal(p,isolated)
    assert len(result['outputs']['zero'])==8 and result['scope'].startswith('firsttokenonly')
