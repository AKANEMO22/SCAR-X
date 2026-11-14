import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scarx_session_transport_fixture',ROOT/'integrations/hermes/scarx_session_transport.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


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
