"""Regression tests for two live chat defects found by scripts/probe_live_chat.py.

1. A stale open question from an earlier prompt hijacked later turns in the same
   session, so asking `chào bạn` returned a clarification about an unrelated «ê».
2. The word readout emitted verbatim repetition cycles such as
   `Tôi là scar-x là scar-x` with no termination.
"""
import pytest

from scarx.learning.dialogue import DialogueLearning, normalized
from scarx.neural.unified import UnifiedCore


def learner():
    return DialogueLearning()


def test_stale_pending_question_never_answers_a_different_prompt():
    d = learner()
    first = d.ask('web', 'ê', 'unsupported coverage')
    assert first is not None
    later = d.ask('web', 'chào bạn', 'unsupported coverage', force=True)
    # The new turn must get a question about ITS prompt, never the stale one.
    assert later is not None
    assert 'chào bạn' in later['question']
    assert 'ê' not in later['question']
    assert normalized(later['prompt']) == normalized('chào bạn')


def test_retiring_stale_question_keeps_at_most_one_open_per_session():
    d = learner()
    d.ask('web', 'ê', 'reason a')
    d.ask('web', 'chào bạn', 'reason b')
    open_rows = [q for q in d.questions if q['session'] == 'web' and q['status'] == 'open']
    assert len(open_rows) == 1
    assert normalized(open_rows[0]['prompt']) == normalized('chào bạn')
    # The retired one is retained as history, not deleted.
    assert any(q['status'] == 'skipped' for q in d.questions)


def test_same_prompt_still_reuses_its_own_open_question():
    d = learner()
    first = d.ask('web', 'chào bạn', 'reason')
    again = d.ask('web', 'chào bạn', 'reason', force=True)
    assert again['id'] == first['id']
    assert len([q for q in d.questions if q['status'] == 'open']) == 1


def test_stale_question_in_one_session_does_not_affect_another():
    d = learner()
    d.ask('web', 'ê', 'reason')
    other = d.ask('cli', 'chào bạn', 'reason', force=True)
    assert other is not None and 'chào bạn' in other['question']


def test_skipped_prompt_is_not_nagged_by_background_work_but_new_user_turn_may_ask():
    # ADR0009: "bỏ qua" must not be re-raised by an idle review, but a NEW explicit
    # user turn may ask again. `force` alone (used by observe) must not resurrect it.
    d = learner()
    asked = d.ask('web', 'chào bạn', 'reason')
    asked['status'] = 'skipped'
    assert d.ask('web', 'chào bạn', 'idle review') is None
    again = d.ask('web', 'chào bạn', 'new user turn', force=True, user_turn=True)
    assert again is not None and 'chào bạn' in again['question']


@pytest.mark.parametrize('words,expected', [
    (['Tôi', 'là', 'scar-x', 'là', 'scar-x'], True),
    (['a', 'b', 'a', 'b'], True),
    (['a', 'a'], True),
    (['Tôi', 'là', 'SCAR-X'], False),
    ([], False),
    (['a'], False),
])
def test_repetition_detector_identifies_immediate_blocks(words, expected):
    assert UnifiedCore._repeating(words) is expected
