"""Source/task binding and persistence, not a source-truth or intelligence test."""

import copy
import ast
import json
import sqlite3
import re
import os
from pathlib import Path
from typing import Any
import time
from types import SimpleNamespace

import pytest
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.gateway import Gateway
from scarx.hybrid.store import Store
from scarx.hybrid.video_context import VideoContext
from scarx.hybrid.web_evidence import supplied_web_sources, persist_web_sources, task_interpretation, task_user_message


QUERY = 'jojo ra tập mới nhất là tập bn'


def test_quantity_interpretation_preserves_original_text_and_is_never_evidence():
    context=task_interpretation(QUERY)
    assert context['original_question']==QUERY and context['expanded_question'].endswith('tập bao nhiêu')
    assert context['requested_value']=='quantity_or_index'
    assert context['interpretation_is_not_factual_evidence']
    assert task_interpretation('Giá BN mới nhất là gì?')['expanded_question']=='Giá BN mới nhất là gì?'
    assert task_interpretation('Latest May news about JoJo')['requested_value']=='answer_the_original_question'
    long=task_interpretation('A long user task. '*1000)
    assert long['question_is_excerpt'] and len(long['original_question'])==1000


def turn(query=QUERY, call='owned-web-call', url='https://example.org/official-release'):
    return [{'role':'user','content':query},
            {'role':'assistant','tool_calls':[{'id':call,'function':{'name':'web_search',
                'arguments':json.dumps({'query':query})}}]},
            {'role':'tool','tool_call_id':call,'content':json.dumps({'success':True,'verified':True,'data':{'web':[
                {'title':'Owned source fixture','url':url,'description':'A launch date is recorded; no current episode index supplied.'}]}})}]


@pytest.fixture
def system(tmp_path):
    store=Store(tmp_path)
    backend=SimpleNamespace(calls=[])
    def generate(messages,**options):
        backend.calls.append(copy.deepcopy(messages))
        return {'text':json.dumps({'question':'Which current episode is supported?',
                                  'reason':'Need evidence for the requested current index.',
                                  'public_topic_id':None})}
    backend.generate=generate
    engine=HybridEngine(store,backend,SimpleNamespace(sample=lambda:{'rss_bytes':0},check=lambda:None))
    gateway=Gateway.__new__(Gateway)
    gateway.store=store;gateway.engine=engine;gateway.video_context=VideoContext(store)
    gateway.active_screen_streams={}
    try:yield gateway,engine,store
    finally:store.close()


def test_only_current_turn_matching_web_call_results_are_observations():
    original=turn();before=copy.deepcopy(original)
    entries,coverage=supplied_web_sources(original)
    assert len(entries)==1 and entries[0]['query']==QUERY and not coverage['partial']
    assert original==before
    assert supplied_web_sources(original+[{'role':'user','content':'Different task'}])[0]==[]
    changed=copy.deepcopy(original);changed[1]['tool_calls'][0]['function']['name']='search_files'
    assert supplied_web_sources(changed)[0]==[]


def host_wrap(raw):
    source=Path(os.environ['LOCALAPPDATA'])/'hermes/hermes-agent/agent/tool_dispatch_helpers.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in ('_neutralize_delimiters','_maybe_wrap_untrusted')]
    assert len(functions)==2 and not any(isinstance(n,(ast.Import,ast.ImportFrom)) for f in functions for n in ast.walk(f))
    scope={'Any':Any,'_DELIMITER_TOKEN_RE':re.compile('untrusted_tool_result',re.IGNORECASE),
           '_is_untrusted_tool':lambda name:name=='web_search','_UNTRUSTED_WRAP_MIN_CHARS':0}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),scope)
    return scope['_maybe_wrap_untrusted']('web_search',raw)


def test_actual_installed_Hermes_wrapper_is_decoded_only_for_storage(system):
    gateway,_,_=system;original=turn()
    original[2]['content']=host_wrap(original[2]['content']);before=copy.deepcopy(original)
    sources,coverage=supplied_web_sources(original)
    assert len(sources)==1 and coverage['paired_results']==1 and not coverage['partial']
    body={'messages':original,'tools':[{'function':{'name':'web_search'}}]}
    prepared=gateway.prepare_agent_messages(body)
    assert original==before and prepared[2]['content']==before[2]['content']
    assert prepared[2]['content'].startswith('<untrusted_tool_result source="web_search">')
    assert body['_scarx_evidence'][0]['status']=='observed'


def test_wrong_wrapper_and_arbitrary_prefix_remain_unparsed_unverified():
    for content in (host_wrap(turn()[2]['content']).replace('source="web_search"','source="read_file"'),
                    'Arbitrary prefix\n'+turn()[2]['content']):
        original=turn();original[2]['content']=content
        sources,coverage=supplied_web_sources(original)
        assert sources==[] and coverage['partial']


def test_host_tool_continuation_is_not_saved_as_the_user_question(system):
    gateway,_,_=system
    original=turn()+[{'role':'user','content':'You just executed tool calls but returned an empty response. Please process the tool results above and continue with the task.'}]
    assert task_user_message(original)['content']==QUERY
    body={'messages':original,'tools':[{'function':{'name':'web_search'}}]}
    prepared=gateway.prepare_agent_messages(body)
    assert body['_scarx_task_question']==QUERY and len(body['_scarx_evidence'])==1
    assert task_user_message(prepared)['content'][0]['text']==QUERY
    changed=copy.deepcopy(original);changed[2]['tool_call_id']='unmatched'
    assert supplied_web_sources(changed)[0]==[]


def test_failed_or_invalid_source_results_never_become_truth_or_unbounded_inputs():
    for url in ('file:///private', 'javascript:alert(1)', 'http://127.0.0.1/private',
                'https://user:password@example.org/a', 'https://localhost/a'):
        assert supplied_web_sources(turn(url=url))[0]==[]
    original=turn();original[2]['content']=json.dumps({'success':False,'verified':True,'error':'failure'})
    sources,coverage=supplied_web_sources(original)
    assert not sources and coverage['partial']


def test_snapshot_is_atomic_and_repeatable_without_changing_evidence_status(system):
    _,engine,store=system;sources,_=supplied_web_sources(turn())
    before=engine.model.dumps()
    first=persist_web_sources(store,sources,'owned-session')
    second=persist_web_sources(store,sources,'owned-session')
    assert first==second and first[0]['status']=='observed'
    assert first[0]['modality']=='web_search_snippet' and engine.model.dumps()==before
    assert store.db.execute('SELECT count(*) FROM evidence').fetchone()[0]==1
    assert store.get('web_source_provenance',first[0]['id'])['factual_verification'] is False
    cold=Store(store.directory)
    try:
        assert cold.evidence(first[0]['id'])==first[0]
        assert persist_web_sources(cold,sources,'owned-session')==first
    finally:cold.close()
    changed=[dict(sources[0],text='Shorter view after host compression')]
    newer=persist_web_sources(store,changed,'owned-session')
    assert newer[0]['id']!=first[0]['id'] and newer[0]['status']=='observed'
    assert store.evidence(first[0]['id'])['text']==sources[0]['text']


def test_source_batch_database_failure_rolls_back_evidence_fts_and_provenance(system):
    _,_,store=system;sources,_=supplied_web_sources(turn())
    sources.append(dict(sources[0],source='https://example.org/second-source'))
    store.db.execute("""CREATE TRIGGER source_fault BEFORE INSERT ON objects
        WHEN NEW.kind='web_source_provenance' BEGIN SELECT RAISE(ABORT,'source fault'); END""")
    store.db.commit()
    with pytest.raises(sqlite3.IntegrityError):persist_web_sources(store,sources,'owned-session')
    for table in ('evidence','evidence_fts'):
        assert store.db.execute('SELECT count(*) FROM '+table).fetchone()[0]==0
    assert not store.list('web_source_provenance')


def test_public_web_context_excludes_unrelated_global_personal_memory(system):
    gateway,engine,store=system
    unrelated=store.observe('là một đoạn trí nhớ cá nhân không liên quan',source='owned human fixture',
                            stream='conversation',status='human_statement')
    body={'messages':turn(),'tools':[{'function':{'name':'web_search'}}],
          'scarx_session_id':'owned-session','_scarx_task_question':'forged task'}
    before=copy.deepcopy(body['messages']);prepared=gateway.prepare_agent_messages(body)
    assert body['messages']==before and body['_scarx_task_question']==QUERY
    assert [e['modality'] for e in body['_scarx_evidence']]==['web_search_snippet']
    assert unrelated['id'] not in body['_scarx_candidates']
    assert unrelated['text'] not in str(prepared)
    assert 'original_question' in str(prepared) and engine.model.updates==0


def test_followup_keeps_actual_user_message_but_review_uses_original_task_and_sources(system):
    gateway,engine,store=system
    original=turn()+[{'role':'assistant','content':'Prior answer'},*turn('tìm kiếm đi',call='second-call')]
    # The actual second search query must still name the original subject.
    original[-2]['tool_calls'][0]['function']['arguments']=json.dumps({'query':QUERY})
    body={'messages':original,'tools':[{'function':{'name':'web_search'}}],'scarx_session_id':'owned-session'}
    gateway.prepare_agent_messages(body)
    engine.record_agent_answer('owned-answer','tìm kiếm đi','A launch date, not the requested index.',
        evidence=body['_scarx_evidence'],candidates=body['_scarx_candidates'],
        task_question=body['_scarx_task_question'],web_coverage=body['_scarx_web_coverage'],
        task_interpretation=body['_scarx_task_interpretation'])
    engine.state['cognition']['last_activity']=time.time()-45
    result=engine.review_once()
    content=json.loads(engine.backend.calls[-1][1]['content'])
    assert content['question']==QUERY and content['user_message']=='tìm kiếm đi'
    assert content['answer_time_utc'].endswith('+00:00') and content['review_time_utc'].endswith('+00:00')
    assert 'planned total' in content['time_reference']
    assert content['task_interpretation']['expanded_question'].endswith('tập bao nhiêu')
    assert content['evidence'][0]['modality']=='web_search_snippet'
    assert content['evidence'][0]['status']=='observed'
    assert not result['weights_updated'] and engine.model.updates==0
    assert store.get('answer','owned-answer')['question']=='tìm kiếm đi'
    body['tools']=None
    gateway.prepare_agent_messages(body)
    assert body['_scarx_task_question']==QUERY and Gateway.native_tools(body) is None


def test_explicit_relevance_feedback_uses_bound_task_not_bare_followup(system,monkeypatch):
    gateway,engine,store=system;captured=[]
    observations=persist_web_sources(store,supplied_web_sources(turn())[0],'owned-session')
    engine.record_agent_answer('owned-answer','tìm kiếm đi','Unverified answer.',evidence=observations,
                              candidates=[e['id'] for e in observations],task_question=QUERY)
    def capture(sequence,*args,**kwargs):
        captured.append(sequence);return {'accepted':False,'reason':'capture-only fixture, no update'}
    monkeypatch.setattr(engine,'_guarded_learn',capture)
    engine.feedback('owned-answer',positive_ids=[observations[0]['id']])
    from scarx.hybrid.learner import features
    import numpy as np
    expected=features(QUERY+' [event] '+observations[0]['text'],
                      (min(len(observations[0]['text'])/4000,1.0),))
    np.testing.assert_array_equal(captured[0],expected)
    assert engine.model.updates==0
