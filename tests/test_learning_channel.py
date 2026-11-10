import pytest
from scarx.app.generation import Generation
from scarx.app.learning_channel import feed,post,append,LIMIT


def test_two_way_feedback_is_verified_idempotent_and_survives_restart(tmp_path):
    app=Generation(tmp_path,background=False)
    try:
        view=post(app,{'client_id':'question-001','kind':'chat','text':'Hãy tính 1 + 1.'})['feed']
        q=next(x for x in view['messages'] if x['kind']=='question' and x['question_status']=='open')
        wrong={'client_id':'reply-wrong-001','kind':'reply','question_id':q['question_id'],'session_id':q['session_id'],'answer':'3'}
        with pytest.raises(ValueError):post(app,wrong)
        assert feed(app)['pending_questions']==1 and not app.dialogue.lessons
        body={**wrong,'client_id':'reply-correct-001','answer':'2'}
        result=post(app,body);assert result['feed']['pending_questions']==0
        assert any(x.get('independently_verified') for x in result['feed']['messages'])
        updates=app.model.core.updates
        assert post(app,body)['deduplicated'] and app.model.core.updates==updates
        with pytest.raises(ValueError):post(app,{**body,'answer':'4'})
    finally:app.close()
    app=Generation(tmp_path,background=False)
    try:
        view=feed(app);assert view['pending_questions']==0
        assert any(x['id']=='u:reply-correct-001' for x in view['messages'])
        assert post(app,body)['deduplicated']
    finally:app.close()


def test_bounded_history_keeps_old_unanswered_question_visible_and_chat_is_not_label(tmp_path):
    app=Generation(tmp_path,background=False)
    try:
        before=app.model.core.fingerprint();post(app,{'client_id':'unknown-001','kind':'chat','text':'Tôi muốn bạn tìm hiểu một điều chưa biết.'})
        assert app.model.core.fingerprint()==before and not app.dialogue.lessons
        with app.db:
            for i in range(LIMIT+20):append(app,'test:'+str(i),'user','ordinary text',kind='chat')
        view=feed(app)
        assert app.db.execute('SELECT COUNT(*) FROM learning_messages').fetchone()[0]==LIMIT
        assert any(x.get('question_status')=='open' for x in view['messages'])
        assert len(view['messages'])<=216
    finally:app.close()
