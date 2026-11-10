"""Regression tests for the guard-on-manual-teach policy and reject observability.

Bug 1: the guard only ran for _automatic lessons, so explicit teaching (web form,
Hermes) replaced weights with no regression check.
Bug 2: a rejected lesson left no trace at all and reported rolled_back=None, so a
refused lesson was indistinguishable from one that never ran.

No learning rule, topology or threshold is touched by these tests.
"""
import numpy as np
import pytest

from scarx.app.generation import Generation, LearningGuardRejected


@pytest.fixture
def app(tmp_path):
    session = Generation(tmp_path, background=False)
    yield session
    session.close()


def test_manual_teach_runs_the_guard(app):
    # The guard must be active on the explicit path, not only for idle lessons.
    result = app.teach({'prompt': 'chào bạn', 'answer': 'Chào bạn.',
                        'repeats': 2, 'confirmation': True})
    metrics = result['learning_metrics']
    assert metrics['guard_checked'] is True
    assert metrics['guard_probe_count'] > 0


def test_rejection_is_labelled_rolled_back_and_leaves_no_weight_change(app):
    from scarx.evaluation.product_acceptance import weights_sha256
    before = weights_sha256(app.model)
    with pytest.raises(LearningGuardRejected) as excinfo:
        # This lesson is known to lose a passing pair on the shipped corpus.
        app.teach({'prompt': 'cảm ơn', 'answer': 'Không có gì.',
                   'repeats': 16, 'confirmation': True})
    metrics = excinfo.value.metrics
    assert metrics['accepted'] is False
    assert metrics['rolled_back'] is True
    assert metrics['guard_lost'] >= 1
    assert metrics['weight_delta'] == {'w_in': 0.0, 'w_rec': 0.0, 'w_out': 0.0}
    assert weights_sha256(app.model) == before


def test_rejection_is_recorded_in_the_trace_for_the_observer(app):
    with pytest.raises(LearningGuardRejected):
        app.teach({'prompt': 'cảm ơn', 'answer': 'Không có gì.',
                   'repeats': 16, 'confirmation': True})
    rejected = [e for e in app.trace if e.get('kind') == 'learning_rejected']
    assert rejected, 'a refused lesson must be observable'
    details = rejected[-1]['details']
    assert details['accepted'] is False and details['rolled_back'] is True
    assert 'guard' in details['reason']


def test_identical_rejection_twice_stops_instead_of_retrying(app, monkeypatch):
    # Bounded retry: the second identical rejection must stop the lesson rather
    # than burning the whole remaining budget. The guard verdict is injected so the
    # test exercises the retry plumbing deterministically instead of depending on
    # whichever pairs happen to pass in a fresh session.
    from scarx.app.generation import LearningGuardRejected
    app.dialogue.enqueue('web', 'cảm ơn', 'Không có gì.', 'test')
    lesson = app.dialogue.lessons[-1]
    lesson['remaining'] = 32
    monkeypatch.setattr(app, '_guard_pairs', lambda _: [('old skill', 'target')])
    # teach() scores the guard twice per attempt (before and after). Report True
    # for every "before" call and False for every "after" call so the pair is
    # always counted as lost, whatever the internal call count is.
    calls = {'n': 0}

    def scores(*_args):
        calls['n'] += 1
        return [calls['n'] % 2 == 1]

    monkeypatch.setattr(app, '_guard_scores', scores)

    first = app._online_microstep('test')
    second = app._online_microstep('test')
    current = next(l for l in app.dialogue.lessons if l['id'] == lesson['id'])
    assert first['accepted'] is False
    assert current['status'] == 'needs_help'
    assert current['remaining'] == 0
    assert second.get('stopped_identical_proposal') is True
