"""Explicit provider-owned stateful chat, separate from prompt-based agents.

The transcript is a UI ledger; the backend owns working state and learning.
No assumption that its input buffer describes neural memory or intelligence.
"""
from copy import deepcopy
import json
import uuid
from urllib.request import Request,urlopen
from urllib.error import URLError


def configure_stateful_chat(agent, config):
    agent._stateful_chat_backend = False
    endpoint = str(getattr(agent, 'base_url', '') or '').rstrip('/')
    model = str(getattr(agent, 'model', '') or '')
    providers = config.get('providers', {})
    if not endpoint or not isinstance(providers, dict):
        return False
    profile = next((p for p in providers.values() if isinstance(p, dict)
                    and p.get('interaction_mode') == 'stateful_chat'
                    and str(p.get('api', '')).rstrip('/') == endpoint
                    and isinstance(p.get('models'), (dict, list, tuple))
                    and model in p['models']), None)
    if profile is None:
        return False
    maximum = profile.get('max_input_characters')
    if type(maximum) is not int or maximum <= 0:
        raise ValueError('stateful_chat requires a positive max_input_characters')
    agent._stateful_chat_backend = True
    agent._stateful_chat_max_input_characters = maximum
    agent._stateful_workload_url = endpoint+'/workload'
    agent._stateful_question_surface = (profile.get('extra_body') or {}).get('scarx_question_surface','inline')
    agent._stateful_questions_enabled = profile.get('proactive_questions') is True and agent._stateful_question_surface!='learning_tab'
    agent.tools = []
    agent.valid_tool_names = set()
    agent._tool_capability_names = frozenset()
    agent._tool_snapshot_generation = 0
    agent._kanban_worker_guidance = ''
    agent.enabled_toolsets = []
    agent.skip_context_files = True
    agent.skip_background_review = True
    agent.disable_persistent_search = True
    agent.ephemeral_system_prompt = None
    # A failed request must stay on the selected stateful service, rather than
    # falling back to a prompt-model with different state and learning semantics.
    agent._fallback_chain = []
    agent._fallback_model = None
    return True


def prepare_stateful_chat_turn(agent, user_message):
    if getattr(agent, 'tools', None):
        raise ValueError('stateful_chat does not support agent tools')
    maximum = agent._stateful_chat_max_input_characters
    if isinstance(user_message, str):
        text = user_message
    elif isinstance(user_message, list):
        text = '\n'.join(p.get('text', '') for p in user_message
                         if isinstance(p, dict) and p.get('type') == 'text')
    else:
        raise ValueError('stateful_chat needs text or multimodal content parts')
    if len(text) > maximum:
        raise ValueError(f'Stateful backend input limit: {maximum} characters per message; this is not a memory/context-capacity score.')
    if not text.strip() and not isinstance(user_message, list):
        raise ValueError('User message is empty')
    # Keep the actual caller data and persisted transcript immutable.
    agent._stateful_chat_raw_user_message = deepcopy(user_message)
    agent._stateful_request_id=uuid.uuid4().hex
    images=sum(1 for p in user_message if isinstance(p,dict) and p.get('type')=='image_url') if isinstance(user_message,list) else 0
    agent._stateful_request_load={'request_id':agent._stateful_request_id,'phase':'prepared','modality':'image_and_text' if images else 'text',
        'input_characters':len(text),'input_utf8_bytes':len(text.encode('utf-8')),'image_count':images,
        'replayed_messages':0,'retrieved_episodes':0,'tool_schemas':0,'source':'actual outgoing user payload'}
    compressor=getattr(agent,'context_compressor',None)
    if compressor is not None:compressor._stateful_request_load=agent._stateful_request_load
    from agent.stateful_questions import activate_question_watcher
    activate_question_watcher(agent)
    return deepcopy(user_message)


def stateful_workload_display(agent):
    """Only adopt backend metrics for THIS request; never another chat's load."""
    load=dict(getattr(agent,'_stateful_request_load',{}))
    if not load:return None
    try:
        headers={}
        if getattr(agent,'api_key',None):headers['Authorization']='Bearer '+agent.api_key
        with urlopen(Request(agent._stateful_workload_url,headers=headers),timeout=.3) as response:
            data=json.loads(response.read(16384))
        backend=data.get('workload')
        if data.get('model')==agent.model and isinstance(backend,dict) and backend.get('request_id')==load['request_id']:
            fields=('phase','modality','input_characters','input_utf8_bytes','image_count','neural_image_pixels','prompt_features_nonzero',
                'neural_input_channels','neural_input_buffer_bytes','active_neurons','allocated_neurons','max_neurons',
                'output_connected_neurons','neural_arrays_bytes','process_rss_bytes','allocation_reason','note')
            load.update({k:backend[k] for k in fields if k in backend})
            load['source']='measured backend workset matched by request ID'
    except (URLError,TimeoutError,OSError,ValueError,KeyError):
        # A previous backend may have no telemetry endpoint. Keep measured host
        # payload sizes and leave core allocation unknown rather than guessing.
        pass
    agent._stateful_request_load=load
    agent.context_compressor._stateful_request_load=load
    return load
