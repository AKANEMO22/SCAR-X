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


def test_before_probe_failure_never_falls_back_to_live_inference_or_teaching(app,monkeypatch):
    from scarx.neural.unified import UnifiedCore
    before=app._pack();calls={'loads':0,'live_generate':0};original=UnifiedCore.loads
    monkeypatch.setattr(app,'_guard_pairs',lambda _p:[])
    def fail_first(payload):
        calls['loads']+=1
        if calls['loads']==1:raise RuntimeError('Injected observation clone failure')
        return original(payload)
    def live_generate(*args,**kwargs):calls['live_generate']+=1;raise AssertionError('Live observation forbidden')
    # A rejected teach restores the model by clearing its instance dict. Put the
    # original method on the instance first so pytest puts it back with setattr.
    # Deleting it on undo fails once rollback has already removed that entry.
    app.model.generate = UnifiedCore.generate.__get__(app.model, UnifiedCore)
    monkeypatch.setattr(UnifiedCore,'loads',staticmethod(fail_first));monkeypatch.setattr(app.model,'generate',live_generate)
    with pytest.raises(RuntimeError,match='observation clone failure'):app.teach({'prompt':'hi','answer':'Hello.','repeats':1,'confirmation':True})
    assert calls['live_generate']==0 and app._pack()==before


def test_rejected_trace_cursor_never_goes_below_retained_event_and_manual_rejection_persists(app,monkeypatch):
    import itertools,json
    verdict=itertools.cycle([[1],[0]])
    monkeypatch.setattr(app,'_guard_pairs',lambda _p:[('oldskill','target')]);monkeypatch.setattr(app,'_guard_scores',lambda *_a:next(verdict))
    cursor=app.sequence;before=app.model.dumps()
    with pytest.raises(LearningGuardRejected):app.teach({'prompt':'hi','answer':'Hello.','repeats':1,'confirmation':True})
    status=app.status(cursor);assert app.model.dumps()==before
    rejected=[e for e in status['trace'] if e['kind']=='learning_rejected'];assert rejected and max(e['id'] for e in rejected)<=status['trace_cursor']
    assert not app.status(status['trace_cursor'])['trace']
    row=app.db.execute("SELECT details FROM events WHERE action='teach-reject' ORDER BY id DESC LIMIT 1").fetchone();record=json.loads(row[0])
    assert record['accepted'] is False and record['rolled_back'] is True and record['weight_delta']['w_out']==0
    from scarx.app.observatory import feed
    observed=feed(app,app.directory)['learning'];assert any(r.get('action')=='teach-reject' and r['rolled_back'] is True for r in observed)


def test_automatic_rejection_also_keeps_trace_highwatermark(app,monkeypatch):
    import itertools
    app.dialogue.enqueue('test','hi','Hello.','test');app.dialogue.lessons[-1]['remaining']=1
    verdict=itertools.cycle([[1],[0]]);monkeypatch.setattr(app,'_guard_pairs',lambda _p:[('oldskill','target')]);monkeypatch.setattr(app,'_guard_scores',lambda *_a:next(verdict))
    result=app._online_microstep('test');assert result['accepted'] is False
    status=app.status();assert max(e['id'] for e in status['trace'])<=status['trace_cursor']
