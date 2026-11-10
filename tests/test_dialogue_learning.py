import json
import numpy as np
import pytest
from scarx.app.generation import Generation
from scarx.learning.dialogue import DialogueLearning
from scarx.learning.verifiers import parse_integer_expression,integer_answer


@pytest.fixture
def app(tmp_path):
    session=Generation(tmp_path,background=False)
    yield session
    session.close()


def test_unknown_input_asks_and_pending_question_survives_restart(app,tmp_path):
    before=app.model.core.fingerprint()
    result=app.chat('2+3 bằng mấy?',session_id='qa-human')
    q=result['learning_question']
    assert result['source'].startswith('learning controller') and 'Hỏi để học' in result['text']
    assert app.model.core.fingerprint()==before
    assert app.pending_question('other-chat')['question'] is None
    payload=app._pack();app.close()
    restarted=Generation(tmp_path,background=False)
    try:
        assert restarted._pack()==payload and restarted.pending_question('qa-human')['question']['id']==q['id']
    finally:restarted.close()


def test_false_numeric_feedback_rejected_and_skip_not_learned(app):
    result=app.chat('1+1 bằng mấy?',session_id='qa');q=result['learning_question'];before=app._pack()
    with pytest.raises(ValueError,match='mâu thuẫn'):
        app.answer_question({'session_id':'qa','question_id':q['id'],'answer':'3'})
    assert app._pack()==before
    result=app.answer_question({'session_id':'qa','question_id':q['id'],'answer':'bỏ qua'})
    assert not result['learned_from_human_label'] and app.dialogue.accepted_updates==0


def test_human_label_causes_local_updates_and_idle_review_without_self_labels(app):
    q=app.chat('1+1 bằng mấy?',session_id='qa')['learning_question'];before=app.model.core.fingerprint()
    result=app.answer_question({'session_id':'qa','question_id':q['id'],'answer':'2'})
    assert result['independently_verified'] and result['learned_from_human_label']
    assert app.dialogue.accepted_updates>0 and app.model.core.fingerprint()!=before
    for _ in range(120):app.idle_once()
    assert app.dialogue.review_count>0 or any(l['status']=='queued' for l in app.dialogue.lessons)
    assert all(l['target']=='2' for l in app.dialogue.lessons)
    assert all(l['kind']=='integer_arithmetic_checked' for l in app.dialogue.lessons)
    assert app.pending_question('qa')['question'] is None


def test_explicit_dialogue_correction_is_scoped_and_recorded(app):
    app.chat('hi',session_id='qa')
    app.dialogue.episodes[-1]['neural_output']='Hello.'
    result=app.chat('không được hello phải là chào',session_id='qa')
    assert result['source'].startswith('learning controller acknowledgement')
    assert any(l['prompt']=='hi' and l['target']=='chào' for l in app.dialogue.lessons)
    assert not any(l['independently_verified'] for l in app.dialogue.lessons)


def test_idle_without_approved_labels_does_not_train_itself(app):
    app.chat('một sự kiện chưa có nguồn',session_id='qa');before=app.model.core.fingerprint();updates=app.model.core.updates
    for _ in range(180):app.idle_once()
    assert app.model.core.fingerprint()==before and app.model.core.updates==updates
    assert app.dialogue.lessons==[] and app.dialogue.review_count>0


def test_online_updates_do_not_replace_foreground_request_load(app):
    response=app.chat('1+1 bằng mấy?',request_id='load-identity',session_id='qa')
    q=response['learning_question']
    app.answer_question({'session_id':'qa','question_id':q['id'],'answer':'2'})
    assert app.workload['request_id']=='load-identity' and app.workload['input_characters']==len('1+1 bằng mấy?')


def test_guard_rejection_restores_model_and_counts_rejection(app,monkeypatch):
    app.dialogue.enqueue('qa','1+1','2','operator test');before=app.model.dumps();calls=iter(([True],[False]))
    identities=(id(app.model),id(app.model.core))
    monkeypatch.setattr(app,'_guard_pairs',lambda _: [('hi','Hello.')])
    monkeypatch.setattr(app,'_guard_scores',lambda *args:next(calls))
    result=app._online_microstep('test')
    assert result['accepted'] is False and app.model.dumps()==before and app.dialogue.rejected_updates==1
    assert (id(app.model),id(app.model.core))==identities


def test_paused_learning_queues_feedback_without_weight_change(app):
    q=app.chat('1+1 bằng mấy?',session_id='qa')['learning_question']
    app.configure_learning({'enabled':False});before=app.model.core.fingerprint()
    app.answer_question({'session_id':'qa','question_id':q['id'],'answer':'2'})
    for _ in range(60):app.idle_once()
    assert app.model.core.fingerprint()==before
    assert app.dialogue.lessons[0]['status']=='queued'


def test_verifier_checks_visible_expression_and_does_not_trust_flag():
    assert integer_answer(parse_integer_expression('4 cộng 1'))==5
    assert integer_answer(parse_integer_expression('Có 6 bông hoa, bớt đi 2 bông thì còn mấy?'))==4
    assert parse_integer_expression('không thêm 3 vào 2') is None
    controller=DialogueLearning()
    lesson=controller.enqueue('qa','không có toán ở đây','lời dạy','synthetic',kind='integer_arithmetic_checked')
    assert lesson['independently_verified'] is False
    bad=controller.dumps();bad['lessons'][0]['independently_verified']=True
    with pytest.raises(ValueError,match='proof'):DialogueLearning.loads(bad)
    with pytest.raises(ValueError,match='mâu thuẫn'):controller.enqueue('qa','1+1','3.','explicit')
    good=controller.enqueue('qa','1+1','2.','explicit')
    assert good['independently_verified'] and DialogueLearning.loads(controller.dumps())


def test_online_commit_failure_preserves_whole_state(app,monkeypatch):
    import sqlite3
    app.dialogue.enqueue('qa','1+1','2','public arithmetic fixture')
    before=app._pack()
    def fail():raise sqlite3.OperationalError('fault injection')
    monkeypatch.setattr(app,'_persist',fail)
    with pytest.raises(sqlite3.OperationalError):app._online_microstep('test')
    assert app._pack()==before


def test_skipped_prompt_does_not_nag_during_idle_review_but_new_turn_can_ask():
    c=DialogueLearning();q=c.ask('qa','unknown','idle review')
    c.feedback('qa',q['id'],'bỏ qua')
    assert c.ask('qa','unknown','idle review again') is None
    assert c.observe('qa','unknown',{'text':'x','tokens':['x'],'stopped':True}) is not None


def test_review_and_learning_policy_changes_rollback_on_persistence_failure(app,monkeypatch):
    import sqlite3
    app.chat('unknown thing',session_id='qa');before=app._pack()
    def fail():raise sqlite3.OperationalError('fault injection')
    monkeypatch.setattr(app,'_persist',fail)
    with pytest.raises(sqlite3.OperationalError):app._review_experience()
    assert app._pack()==before
    with pytest.raises(sqlite3.OperationalError):app.configure_learning({'enabled':False,'ask_enabled':False})
    assert app._pack()==before
