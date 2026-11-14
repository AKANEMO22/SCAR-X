import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scarx_session_transport_fixture',ROOT/'integrations/hermes/scarx_session_transport.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def isolated_Hermes_host_note(monkeypatch):
    monkeypatch.setitem(sys.modules,'agent.prompt_builder',SimpleNamespace(STEER_CHANNEL_NOTE='## Mid-turn user steering\nEXACT OWNED HOST NOTE'))
    monkeypatch.setitem(sys.modules,'gateway.session_submission',SimpleNamespace(current_submission_id=lambda:None))


def test_actual_transcript_replaces_ambient_affinity_without_dropping_headers():
    old={'extra_headers':{'x-scarx-session':'host-affinity-alias','Other':'kept'},'messages':['kept']}
    agent=SimpleNamespace(model='scarx-hybrid',base_url='http://127.0.0.1:8775/v1',session_id='local-real-transcript')
    got=module.bind_transcript_session(agent,old)
    assert got['extra_headers']=={'X-SCARX-Session':'local-real-transcript','Other':'kept'}
    assert old['extra_headers']['x-scarx-session']=='host-affinity-alias' and got['messages'] is old['messages']


@pytest.mark.parametrize('field,value',[('model','other'),('base_url','https://example.com/v1'),
    ('base_url','http://127.0.0.1:8888/v1'),('base_url','http://127.0.0.1:8775/wrong'),
    ('session_id',''),('session_id','local'),('session_id','x\r\ninjected: yes')])
def test_nonlocal_routes_auxiliary_and_invalid_session_unchanged(field,value):
    agent=SimpleNamespace(model='scarx-hybrid',base_url='http://127.0.0.1:8775/v1',session_id='local-real-transcript')
    setattr(agent,field,value);old={'extra_headers':{'Other':'kept'}}
    assert module.bind_transcript_session(agent,old) is old


@pytest.mark.parametrize('query',['Chào bạn! Trả lời ngắn bằng tiếng Việt nhé.','hello','xin chào','Hi! Reply briefly in English.'])
def test_pure_greeting_uses_model_without_irrelevant_schemas(query):
    agent=SimpleNamespace(model='scarx-hybrid',base_url='http://localhost:8775/v1',session_id='local-real-transcript')
    old={'messages':[{'role':'system','content':'KEEP ALL USER AND WORKSPACE RULES'},{'role':'user','content':query}],
         'tools':[{'type':'function','function':{'name':'terminal'}}],'tool_choice':'auto'}
    got=module.bind_transcript_session(agent,old)
    assert 'tools' not in got and 'tool_choice' not in got and got['messages'] is old['messages']
    assert old['tools'] and len(got['messages'])==2


@pytest.mark.parametrize('content',['Chào bạn, đọc README giúp tôi','hello, search the web',
    'Hi, summarize the video','Chào, mở terminal', [{'type':'text','text':'hello'},{'type':'image_url','image_url':{'url':'data:image/png;base64,fixture'}}]])
def test_actual_tasks_and_images_keep_all_tools(content):
    assert module.social_greeting_only([{'role':'user','content':content}]) is False


def test_explicit_tool_selection_and_followup_results_never_stripped():
    agent=SimpleNamespace(model='scarx-hybrid',base_url='http://localhost:8775/v1',session_id='local-real-transcript')
    old={'messages':[{'role':'user','content':'hello'}],'tools':[{'type':'function'}],'tool_choice':'required'}
    assert module.bind_transcript_session(agent,old)['tools'] is old['tools']
    assert not module.social_greeting_only([{'role':'user','content':'hello'},{'role':'tool','content':'real result'}])


def test_social_turn_omits_only_exact_generated_host_note_without_hiding_output():
    note=sys.modules['agent.prompt_builder'].STEER_CHANNEL_NOTE
    agent=SimpleNamespace(model='scarx-hybrid',base_url='http://127.0.0.1:8775/v1',session_id='owned')
    old={'messages':[{'role':'system','content':'KEEP RULE A\n'+note+'\nKEEP WORKSPACE B'},
                     {'role':'assistant','content':'Earlier answer retained'}, {'role':'user','content':'chào bạn'}],
         'tools':[{'type':'function'}]}
    got=module.bind_transcript_session(agent,old)
    assert got['messages'][0]['content']=='KEEP RULE A\n\nKEEP WORKSPACE B'
    assert note in old['messages'][0]['content']
    assert got['messages'][1:] == old['messages'][1:]
    assert 'answer' not in got and 'content' not in got


@pytest.mark.parametrize('query',['read README.md','[OUT-OF-BAND USER MESSAGE — hello]','Explain the phrase OUT-OF-BAND USER MESSAGE'])
def test_real_tasks_and_actual_steer_messages_keep_host_note_and_tools(query):
    note=sys.modules['agent.prompt_builder'].STEER_CHANNEL_NOTE
    agent=SimpleNamespace(model='scarx-hybrid',base_url='http://127.0.0.1:8775/v1',session_id='owned')
    old={'messages':[{'role':'system','content':note},{'role':'user','content':query}],'tools':[{'type':'function'}]}
    got=module.bind_transcript_session(agent,old)
    assert got['messages'] is old['messages'] and got['tools'] is old['tools']


def test_headers_use_actual_admitted_request_not_text_or_cached_agent(monkeypatch):
    monkeypatch.setitem(sys.modules,'gateway.session_submission',SimpleNamespace(current_submission_id=lambda:'actual-request-A'))
    agent=SimpleNamespace(model='scarx-hybrid',base_url='http://127.0.0.1:8775/v1',session_id='owned')
    old={'messages':[{'role':'user','content':'xem giúp'}],'extra_headers':{'x-scarx-submission':'stale-request'}}
    result=module.bind_transcript_session(agent,old)
    assert result['extra_headers']['X-SCARX-Submission']=='actual-request-A'
    assert 'x-scarx-submission' not in result['extra_headers']
    assert old['extra_headers']['x-scarx-submission']=='stale-request'
