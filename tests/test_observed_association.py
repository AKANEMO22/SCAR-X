import io,json,zipfile
import numpy as np
import pytest
from scarx.evaluation.observed_association import ObservedAssociation
from scarx.evaluation.fixed_word_codes import fit
from scarx.neural.unified import UnifiedCore

VOCAB=['<eos>','<bos>','hi','bye','.']


def model(condition='paired'):return ObservedAssociation(VOCAB,31013,condition,fit(['hello','goodbye'],'known-digest128-v1'))


def test_initial_actual_forward_matches_g1_and_prediction_accepts_no_current_target():
    m=model();g=UnifiedCore(VOCAB,31013);q=np.linspace(-.2,.2,128,dtype=np.float32)
    m._reset_prediction();g.core.reset_state();g.boundary()
    p=m._word_step(1,q,0);reference=g.word_step(1,q,0)
    np.testing.assert_array_equal(m.core.features,g.core.features);np.testing.assert_array_equal(p,reference)
    before=m.dumps()
    with pytest.raises(TypeError):m.generate('hello',target='hi.')
    with pytest.raises(ValueError):m.learn('hello','unrepresentable.')
    assert m.dumps()==before


def test_hand_delta_locality_and_trace_ablation_keep_unrelated_synapses_fixed():
    m=model();old=m.core.w_in.copy();rec=m.core.w_rec.copy();head=m.core.w_out.copy();m.question_trace[:]=.3;m.mean_post[:]=.1
    post=np.linspace(0,1,48,dtype=np.float32);expected=old.copy()
    expected[:12,m.context_start:m.context_end]=np.clip(expected[:12,m.context_start:m.context_end]+np.clip(.02*(post[:12]-.1)[:,None]*.3,-.025,.025),-3,3)
    m._associate_slot(post);np.testing.assert_array_equal(m.core.w_in,expected);np.testing.assert_array_equal(m.core.w_rec,rec);np.testing.assert_array_equal(m.core.w_out,head)
    frozen=model('frozen');erased=model('erase_trace')
    for _ in range(8):
        assert frozen.learn('hello','hi.')==erased.learn('hello','hi.')
        for name,value in frozen.core.arrays().items():np.testing.assert_array_equal(value,erased.core.arrays()[name])
    assert not erased.hidden_transactions and not frozen.hidden_transactions


def test_observation_cannot_leak_into_prediction_head_and_codec_continues_exactly():
    m=model();initial=m.core.w_out.copy();m.observe('hello','hi.');np.testing.assert_array_equal(initial,m.core.w_out)
    assert m.hidden_transactions>0
    a=ObservedAssociation.loads(m.dumps());b=ObservedAssociation.loads(m.dumps())
    # Pollute ALL neuronworkingstates but keep synaptic knowledge equal.
    for name in ('voltage','adaptation','spikes','features'):getattr(b.core,name).fill(5.)
    assert a.generate('hello')==b.generate('hello')
    a=ObservedAssociation.loads(m.dumps());b=ObservedAssociation.loads(m.dumps())
    for name in ('voltage','adaptation','spikes','features'):getattr(b.core,name).fill(5.)
    before=a.core.w_in.copy();mean=a.mean_post.copy()
    assert a.learn_prediction('hello','hi.')==b.learn_prediction('hello','hi.')
    np.testing.assert_array_equal(before,a.core.w_in);np.testing.assert_array_equal(mean,a.mean_post)
    m.learn('goodbye','bye.');copy=ObservedAssociation.loads(m.dumps());assert copy.dumps()==m.dumps()
    assert m.learn('hello','hi.')==copy.learn('hello','hi.') and m.dumps()==copy.dumps()
    assert m.sampler_rng.integers(100)==copy.sampler_rng.integers(100) and m.dumps()==copy.dumps()
    with pytest.raises(ValueError):UnifiedCore.loads(m.dumps())
    with zipfile.ZipFile(io.BytesIO(m.dumps())) as z:parts={n:z.read(n) for n in z.namelist()}
    meta=json.loads(parts['metadata.json']);meta['encoding']['idf']['hello']=3.;parts['metadata.json']=json.dumps(meta).encode();bad=io.BytesIO()
    with zipfile.ZipFile(bad,'w') as z:
        for name,raw in parts.items():z.writestr(name,raw)
    with pytest.raises(ValueError,match='checksum'):ObservedAssociation.loads(bad.getvalue())
