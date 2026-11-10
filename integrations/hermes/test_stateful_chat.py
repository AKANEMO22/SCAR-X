"""Provider opt-in and real agent-init floor invariants."""
from types import SimpleNamespace
import pytest
from agent.stateful_chat import configure_stateful_chat, prepare_stateful_chat_turn
from agent.agent_init import _enforce_minimum_context


def test_provider_mode_binds_endpoint_and_model_and_preserves_default_guard():
    config = {'providers': {'example': {'api': 'http://127.0.0.1:8765/v1',
        'models': {'stateful-test': {}}, 'interaction_mode': 'stateful_chat', 'max_input_characters': 20}}}
    def candidate(model='stateful-test', url='http://127.0.0.1:8765/v1'):
        return SimpleNamespace(model=model,base_url=url,provider='custom',tools=[{'function':{'name':'old'}}],
            _config_context_length=4096,context_compressor=SimpleNamespace(context_length=4096))
    opted=candidate();assert configure_stateful_chat(opted,config)
    assert opted.tools==[] and opted._fallback_chain==[]
    _enforce_minimum_context(opted)
    from agent.context_breakdown import context_usage_fields, compute_session_context_breakdown
    opted.context_compressor._stateful_input_limit_characters=20
    fields=context_usage_fields(opted.context_compressor)
    assert fields['context_mode']=='backend_state' and fields['context_max']==0
    assert compute_session_context_breakdown(opted,[{'role':'user','content':'old transcript'}])['categories']==[]
    for other in (candidate('other'),candidate(url='http://127.0.0.1:9999/v1')):
        assert configure_stateful_chat(other,config) is False
        with pytest.raises(ValueError,match='minimum'):_enforce_minimum_context(other)
    opted.tools=[{'function':{'name':'injected'}}]
    with pytest.raises(ValueError):_enforce_minimum_context(opted)


def test_stateful_turn_keeps_caller_content_and_enforces_actual_input_limit():
    agent=SimpleNamespace(tools=[],_stateful_chat_max_input_characters=12)
    content=[{'type':'text','text':'hello'},{'type':'image_url','image_url':{'url':'data:image/png;base64,AA=='}}]
    copy=prepare_stateful_chat_turn(agent,content)
    assert agent._stateful_request_load['input_characters']==5 and agent._stateful_request_load['image_count']==1
    assert copy==content and copy is not content
    copy[0]['text']='changed';assert content[0]['text']=='hello'
    from agent.turn_request_assembly import assemble_api_request
    agent._stateful_chat_backend=True
    scaffold=[{'role':'system','content':'LLM tools scaffold'},{'role':'user','content':'injected task state'}]
    assembled=assemble_api_request(agent,messages=scaffold,current_turn_user_idx=1,_ext_prefetch_cache=None,
        _plugin_user_context='injected policy',moa_config=None,active_system_prompt='LLM system',
        original_user_message=content,pending_moa_prepared_request=None,request_logger=None)
    assert assembled.api_messages==[{'role':'user','content':content}] and assembled.tools_for_api==[]
    assert scaffold[1]['content']=='injected task state'
    with pytest.raises(ValueError,match='input limit'):prepare_stateful_chat_turn(agent,'x'*13)
    agent.tools=[{}]
    with pytest.raises(ValueError,match='tools'):prepare_stateful_chat_turn(agent,'hi')
