"""The acceptance metric must FAIL a broken reply.

This exists because the metric itself was wrong: a looping, mid-sentence prefix
('Tôi là scar-x là') was counted correct for 'bạn là ai' merely because it
contained 'scar-x'. That inflated free-answer accuracy from 0.000 to 0.500 and
retention from 0.000 to 1.000 for several rounds.

An acceptance metric that passes a broken reply is worse than no metric: it hides
exactly the defect it exists to expose. These tests pin the strict definition.
"""
import pytest

from scarx.evaluation.product_acceptance import _judge, CASES


def test_looping_prefix_is_not_correct():
    row = _judge('Tôi là scar-x là', 'scar-x', 'bạn là ai', ('scar-x',))
    assert row['has_expected_words'] is True
    assert row['complete'] is False
    assert row['correct'] is False, 'a truncated reply must not count as correct'


def test_bare_word_without_punctuation_is_not_complete():
    row = _judge('Hello', 'hello', 'hi', ('hello',))
    assert row['has_expected_words'] is True
    assert row['complete'] is False
    assert row['correct'] is False


def test_finished_answer_is_correct():
    row = _judge('Chào bạn.', 'chào bạn', 'chào', ('chào', 'bạn'))
    assert row['complete'] is True
    assert row['repetition'] is False
    assert row['correct'] is True


def test_repetition_loop_is_never_correct():
    row = _judge('Tạm biệt tạm biệt.', 'tạm biệt', 'tạm biệt', ('tạm', 'biệt'))
    assert row['repetition'] is True
    assert row['correct'] is False


def test_partial_content_is_not_correct():
    # Missing the required second word.
    row = _judge('Chào.', 'chào bạn', 'chào', ('chào', 'bạn'))
    assert row['has_expected_words'] is False
    assert row['correct'] is False


def test_controller_routing_text_is_never_correct():
    text = '[Hỏi để học] SCAR-X cần bạn hướng dẫn cho: «chào».'
    row = _judge(text, 'chào bạn', 'chào', ('chào', 'bạn'))
    assert row['routed'] is True
    assert row['usable'] is False
    assert row['correct'] is False


def test_every_case_declares_required_words():
    # Guards against reintroducing a loose single-token expectation.
    for case in CASES:
        assert len(case) == 3, f'{case[0]!r} must declare require_all'
        assert case[2], f'{case[0]!r} must require at least one word'
