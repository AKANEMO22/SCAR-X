import numpy as np
from scarx.evaluation.knowledge_capacity import dataset,retention,frozen_slot,probe,forgetting
from scarx.learning.verifiers import parse_integer_expression,integer_answer
from scarx.neural.unified import UnifiedCore


def test_arithmetic_splits_are_disjoint_verified_and_reproducible():
    a=dataset();assert a==dataset() and len(a)==168
    assert len({tuple(x['key']) for x in a})==168
    for row in a:
        assert str(integer_answer(parse_integer_expression(row['prompt'])))==row['target']
        assert parse_integer_expression(row['prompt'])==parse_integer_expression(row['paraphrase'])
    assert sum(x['split']=='curriculum' for x in a)==128
    assert sum(x['split']=='unseen' for x in a)==32
    b=dataset(nonnegative_only=True)
    assert len(b)==168 and len({tuple(x['key']) for x in b})==168
    assert all(int(x['target'])>=0 for x in b)


def test_no_acquisition_is_not_perfect_retention():
    assert retention([{'id':'x','correct':False}],[],[{'id':'x','correct':False}])['rate'] is None
    result=retention([{'id':'x','correct':True}],[],[{'id':'x','correct':False}])
    assert result['acquired']==1 and result['retained']==0 and result['rate']==0


def test_skill_loss_is_not_hidden_by_new_skill_gains():
    result=forgetting([1.,0.],[.5,1.])
    assert result['mean']==.25 and result['worst']==.5 and result['next_best']==[1.,1.]
    assert forgetting(result['next_best'],[.5,.5])['mean']==.5


def test_probes_do_not_mutate_live_neural_state_and_frozen_does_not_learn():
    m=UnifiedCore(['<eos>','<bos>','0','1','2'],seed=17)
    saved=m.dumps();probe(m,dataset()[:2]);assert m.dumps()==saved
    before={k:getattr(m.core,k).copy() for k in m.core.weight_names}
    frozen_slot(m,'1 + 1','2')
    for k,a in before.items():np.testing.assert_array_equal(a,getattr(m.core,k))
