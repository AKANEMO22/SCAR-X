import inspect
import numpy as np
import pytest
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.neural_beam import decode


def test_greedy_search_matches_exact_original_neural_state_and_weights_once():
    model=UnifiedCore(['<eos>','<bos>','hello','bye','.']);payload=model.dumps()
    a=UnifiedCore.loads(payload);b=UnifiedCore.loads(payload);expected=a.generate('hi',24)
    actual=decode(b,'hi',width=1,alpha=0.,max_words=24)
    for key in ('text','tokens','stopped','stop_reason','mean_probability'):assert actual[key]==expected[key]
    assert a.dumps()==b.dumps() and actual['search']['shared_weights_once']
    assert 'target' not in inspect.signature(decode).parameters


def test_bounded_neural_search_terminates_without_weight_copies_or_label_access():
    model=UnifiedCore(['<eos>','<bos>','hello','bye','.']);fingerprint=model.core.fingerprint()
    ids={n:id(getattr(model.core,n)) for n in model.core.weight_names}
    result=decode(model,'a new prompt',width=8,alpha=.6,node_budget=1,max_words=24)
    assert result['search']['expanded_word_steps']==1 and result['search']['computation_ticks']==15
    assert result['search']['budget_exhausted'] and not result['stopped'] and result['stop_reason']=='work_budget'
    assert model.core.fingerprint()==fingerprint and all(id(getattr(model.core,n))==value for n,value in ids.items())
    assert result['search']['peak_unique_state_array_bytes']<1000000
    with pytest.raises(ValueError):decode(model,'hi',width=9)
