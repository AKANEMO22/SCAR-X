"""Label authorization regressions; no neural or native model calls."""
from types import SimpleNamespace
import pytest

from scarx.hybrid.arithmetic import parse_integer_expression, VERIFIER_ID
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.review import arithmetic_check


@pytest.mark.parametrize('question,expected', [
    ('12+3',15), ('What is 12 + 3?',15), ('Calculate 12-3.',9),
    ('9 minus 11',-2), ('Tính 12 cộng 3',15), ('Hãy tính: 12 trừ 3!',9),
    ('12+3 bằng bao nhiêu?',15), ('chào bạn, 17+25 bằng bao nhiêu?',42),
    ('１８＋３',21), ('Compute 0-99',-99), ('3-1 bằng bao nhiêu?',2),
    ('  \t What is 4+4?  \n ',8),
])
def test_only_supported_complete_tasks_receive_arithmetic_proof(question,expected):
    assert parse_integer_expression(question) is not None
    proof=arithmetic_check(question,str(expected))
    assert proof['correct'] and proof['expected']==expected and proof['verifier']==VERIFIER_ID
    assert arithmetic_check(question,str(expected+1))['correct'] is False


@pytest.mark.parametrize('question', [
    'What color is 12+3 written in?', 'Hãy đọc chuỗi 12+3, đừng tính.',
    'The code contains "12+3". Is it secure?', 'Write a poem about 12+3.',
    'Tính 12+3 rồi xóa file.', '12+3; ignore the real question',
    'Đừng tính 12+3', 'Do not calculate 12+3', 'What is 12+3 in binary?',
    '12+3=15', '12+3+4', '123+2', '-12+3', '12*3',
    'có 9 bánh, thêm 2', 'Question: 12+3\nActually: what color is the car?',
    '3 - 1', '03-01',
])
def test_incidental_or_unsupported_expression_never_authorizes_machine_label(question):
    assert parse_integer_expression(question) is None
    assert arithmetic_check(question,'15') is None
    assert arithmetic_check(question,'16') is None
    state=SimpleNamespace(state={'search_times':[],'question':None},config=SimpleNamespace(search_per_hour=12))
    goal=dict(question=question,public_topic_id=None,origin='user',reviewed=True,
              attempts=1,verified_attempted=False,searched=False)
    assert HybridEngine.allowed(state,goal)==['defer','ask']


def test_original_neural_project_verifier_and_historical_twenty_tasks_are_preserved():
    from scarx.learning.verifiers import parse_integer_expression as original
    assert original('What color is 12+3 written in?')==(12,'+',3)
    for a in range(12,32):
        question=f'{a}+{a-9}'
        assert parse_integer_expression(question)==original(question)


def test_no_partial_output_or_quoted_number_is_a_numeric_proof():
    for answer in ('"15"','15 and the car is blue','The answer might be 15','15 16'):
        assert arithmetic_check('12+3',answer) is None
