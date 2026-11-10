import numpy as np
import pytest
from scarx.evaluation.direct_words import DirectWords
from scarx.evaluation.direct_word_data import TARGETS
from scarx.neural.language import tokenize
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.readout_eligibility import ReadoutEligibility

VOCAB=['<eos>','<bos>']+sorted({t for s in TARGETS.values() for t in tokenize(s)})


def make(condition='correct'):
    m=DirectWords(VOCAB,36313,condition);m.calibrate(['chào bạn','hello assistant','bạn tên gì','thank you']);return m


def test_word_inputs_match_g1_encoder_and_forward_without_target_or_second_core():
    m=make();g=UnifiedCore(VOCAB,36313);seen=[];original=m.learner.step
    def record(x):seen.append(x.copy());return original(x)
    m.learner.step=record;m.prepare('Bạn tên gì?');m.word(1,0)
    np.testing.assert_array_equal(m.context,g.context('Bạn tên gì?'));assert len(seen)==15
    expected=[];actual=g.tick
    def capture(x=None,*args,**kwargs):expected.append(x.copy());return actual(x,*args,**kwargs)
    g.tick=capture;g.word_step(1,g.context('Bạn tên gì?'),0)
    np.testing.assert_array_equal(np.stack(seen[-3:]),np.stack(expected));assert not np.stack(seen)[:,len(VOCAB)+128:len(VOCAB)+192].any()
    before=m.dumps()
    with pytest.raises(TypeError):m.generate('hello',target='Hello.')
    assert before==m.dumps()


def test_word_learning_zero_frozen_and_partial_endpoints_checkpoint_continue_exact():
    zero=make('zero');frozen=make('frozen')
    for p,t in [('hello','Hello.'),('bạn là ai','Tôi là SCAR-X.')]:assert zero.learn(p,t)==frozen.learn(p,t)
    assert zero.core.dumps()==frozen.core.dumps() and zero.head.dumps()==frozen.head.dumps()
    m=make();before=m.core.w_in.copy();m.learn('hello','Hello.');assert np.any(before!=m.core.w_in)
    m.prepare('hello',True);m.word(1,0);m.learner.accumulate(m.ids['hello']);payload=m.dumps();copy=DirectWords.loads(payload)
    assert payload==copy.dumps()
    for a in (m,copy):a.word(a.ids['hello'],1);a.learner.accumulate(a.ids['.']);a.word(a.ids['.'],2);a.learner.accumulate(0);a.learner.finish();a.responses_seen+=1
    assert m.dumps()==copy.dumps() and m.generate('hello')==copy.generate('hello') and m.dumps()==copy.dumps()
    assert m.learn('bạn là ai','Tôi là SCAR-X.')==copy.learn('bạn là ai','Tôi là SCAR-X.') and m.dumps()==copy.dumps()
    for loader in (UnifiedCore.loads,ReadoutEligibility.loads):
        with pytest.raises(ValueError):loader(payload)


def test_word_transaction_restores_all_state_after_hidden_update_failure():
    m=make();before=m.dumps()
    def fail(direction):raise MemoryError('testinjection')
    m.learner._apply_direction=fail
    with pytest.raises(MemoryError):m.learn('hello','Hello.')
    assert m.dumps()==before and m.core.w_out is m.head.weights


def test_word_control_frozen_parity_and_trained_optimizer_continuation():
    from scarx.evaluation.controls import DirectWordLearnabilityControl,torch
    torch.set_num_threads(1);a=make('frozen');control=DirectWordLearnabilityControl(DirectWords.loads(a.dumps()))
    assert a.learn('hello','Hello.')==control.learn('hello','Hello.') and a.dumps()==control.reference.dumps()
    control=DirectWordLearnabilityControl(make());before=control.reference.core.w_in.copy()
    control.learn('hello','Hello.');assert np.any(before!=control.reference.core.w_in) and control.backward_passes==1
    copy=DirectWordLearnabilityControl.loads(control.dumps());assert copy.dumps()==control.dumps()
    assert control.learn('bạn là ai','Tôi là SCAR-X.')==copy.learn('bạn là ai','Tôi là SCAR-X.') and control.dumps()==copy.dumps()
    assert control.max_graph_feature_error<=1e-6
    with pytest.raises(ValueError):DirectWords.loads(control.dumps())
