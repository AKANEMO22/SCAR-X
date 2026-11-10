import weakref
from agent import stateful_questions as questions


class Agent:
    session_id='test-session';base_url='http://127.0.0.1:8765/v1'
    _stateful_chat_backend=True;_stateful_questions_enabled=True


def one_poll(monkeypatch,agent):
    counter=iter((False,True))
    monkeypatch.setattr(questions,'_wake',type('Wake',(),{'wait':lambda self,_:next(counter)})())
    monkeypatch.setattr(questions,'_agent_ref',weakref.ref(agent))


def test_clarification_answer_is_linked_to_exact_question_and_session(monkeypatch):
    agent=Agent();seen=[]
    agent.clarify_callback=lambda prompt,choices:'2'
    def request(_,path,data=None):
        seen.append((path,data))
        return {'question':{'id':'q-public','question':'1+1?'}} if data is None else {'recorded':True}
    monkeypatch.setattr(questions,'_request',request);one_poll(monkeypatch,agent)
    questions._watch()
    assert seen[0][0]=='/questions?session_id=test-session'
    assert seen[1]==('/feedback',{'session_id':'test-session','question_id':'q-public','answer':'2'})


def test_empty_or_cancelled_clarification_never_becomes_a_learning_label(monkeypatch):
    agent=Agent();agent.clarify_callback=lambda prompt,choices:'';posted=[]
    def request(_,path,data=None):
        if data is None:return {'question':{'id':'q-public','question':'unknown'}}
        posted.append(data);return {'recorded':True}
    monkeypatch.setattr(questions,'_request',request);one_poll(monkeypatch,agent)
    questions._watch()
    assert posted==[{'session_id':'test-session','question_id':'q-public','answer':'bỏ qua'}]
