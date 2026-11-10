import numpy as np
import pytest
from scarx.evaluation.direct_words import DirectWords
from scarx.evaluation.direct_word_data import TARGETS
from scarx.neural.language import tokenize
from scarx.evaluation.word_training_audit import protected,free_forward,prefix_forward,potential_direction,credit_summary,corpus_structure

VOCAB=['<eos>','<bos>']+sorted({t for target in TARGETS.values() for t in tokenize(target)})


def model():
    m=DirectWords(VOCAB,36313);m.calibrate(['hello','chào bạn','bạn là ai','thanks']);m.learn('hello','Hello.');return m


def test_free_has_no_target_input_and_matches_original_forward_while_teacherprefix_is_explicit():
    m=model();copy=DirectWords.loads(m.dumps());before=protected(m);a=free_forward(m,'hello');b=copy.generate('hello')
    assert a['text']==b['text'] and a['stopped']==b['stopped'] and protected(m)==before
    assert a['actual_ticks']==12+3*len(a['rows']) and a['rows'][-1]['spikes'].shape==(48,)
    with pytest.raises(TypeError):free_forward(m,'hello',target='Hello.')
    tokens=m.target_tokens('Hello.');row=prefix_forward(m,'hello',tokens[:-1],measure_credit=True)
    assert len(row)==len(tokens) and protected(m)==before
    dirs=[potential_direction(m,r,t) for r,t in zip(row,tokens)];summary=credit_summary(m,dirs);assert summary['groups']['reserved_image']['first_norm']==0
    assert protected(m)==before and m.core.updates==3 # hello punctuation EOS


def test_question_zero_hook_changes_only_question_channels_and_is_restored():
    m=model();before=protected(m);seen=[];actual=m.learner.step
    def record(x):seen.append(x.copy());return actual(x)
    m.learner.step=record;prefix_forward(m,'hello',[],zero_question=True);assert not np.stack(seen)[:,len(VOCAB):len(VOCAB)+128].any()
    assert '_context' not in m.__dict__ and protected(m)==before
    seen.clear();prefix_forward(m,'hello',[]);assert np.stack(seen)[:,len(VOCAB):len(VOCAB)+128].any() and protected(m)==before


def test_template_structure_exposes_prefix_shortcut_without_any_neuralfit():
    m=model();r=corpus_structure([(m.target_tokens(t),m.vocabulary) for t in TARGETS.values()])
    assert r['unique_first_words']==8 and r['PREFIX_STRUCTURE_SHORTCUT'] and r['nonempty_prefix_determines_next_fraction']==1.
    assert r['later_positions']==r['total_positions_uniform_templates']-8


def test_aggregate_keeps_token_wholeanswer_and_censored_positions_separate():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_word_training_objective import aggregate
    row={'family':'vi_greet','free_exact':False,'tf_first':0.,'tf_later':1.,'wrong_first':0.,'wrong_later':1.,'zero_first':0.,'zero_later':.5,
        'tf_later_content':None,'tf_later_punctuation':1.,'tf_same_free_later':1.,'free_prefix_same_later':0.,'tf_first_nll':2.,'tf_later_nll':.1,
        'gold_later_positions':3,'free_later_positions_reached':1,'free_gold_later_positions_censored':2,'credit':None}
    r=aggregate([row]);assert r['free_exact']==0 and r['tf_later']==1 and r['matched_later_exposure_gap']==1 and r['free_gold_later_positions_censored']==2
    assert r['potential_credit'] is None and r['tf_later_content'] is None and r['flags']['DIRECTION_CONFLICT'] is None
