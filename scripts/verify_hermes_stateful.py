"""Real Hermes AIAgent -> SCAR-X, isolated home, public input only."""
import os
from pathlib import Path
import sys
import uuid
import json
import time
import copy
import argparse
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--stream',action='store_true');args=parser.parse_args()
hermes=Path(os.environ['LOCALAPPDATA'])/'hermes/hermes-agent'
home=root/'local'/('hermes-stateful-qa-'+uuid.uuid4().hex[:8]);home.mkdir(parents=True)
os.environ['HERMES_HOME']=str(home)
for key in list(os.environ):
    if key.endswith(('_API_KEY','_AUTH_TOKEN')):os.environ.pop(key)
sys.path.insert(0,str(hermes))
import yaml
snippet=yaml.safe_load((root/'integrations/hermes/provider.yaml').read_text(encoding='utf-8'))
config={**snippet,'model':{'default':'scarx-generation1','provider':'custom:scarx-local','base_url':'http://127.0.0.1:8765/v1','context_length':4096,'streaming':args.stream},
 'agent':{'environment_probe':False,'stall_guards':False,'task_completion_guidance':False},
 'compression':{'enabled':False},'platform_toolsets':{'cli':[],'web':[]},'auxiliary':{},'fallback_models':[]}
(home/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
from run_agent import AIAgent
from hermes_state import SessionDB
db=SessionDB(); session='stateful-qa-'+uuid.uuid4().hex[:8]
agent=AIAgent(model='scarx-generation1',base_url='http://127.0.0.1:8765/v1',api_key='no-key-required',provider='custom:scarx-local',api_mode='chat_completions',
    enabled_toolsets=[],quiet_mode=True,max_iterations=1,max_tokens=32,skip_memory=True,skip_context_files=True,
    skip_background_review=True,session_id=session,session_db=db)
assert agent._stateful_chat_backend and agent.tools==[]
assert agent.context_compressor.context_length==4096
from agent.context_breakdown import compute_session_context_breakdown
memory_display=compute_session_context_breakdown(agent,[])
assert memory_display['context_mode']=='backend_state' and memory_display['context_max']==0
assert memory_display['input_limit_characters']==2048
import httpx
original=httpx.HTTPTransport.handle_request;observed=[]
def trace_request(transport,request):
    if request.url.port==8765 and request.url.path.endswith('/chat/completions'):
        data=json.loads(request.read())
        observed.append({k:data.get(k) for k in ('messages','tools','model','stream')})
    return original(transport,request)
httpx.HTTPTransport.handle_request=trace_request
start=time.monotonic()
first=agent.run_conversation('hi')
assert first.get('final_response'),{k:v for k,v in first.items() if k in ('error','status','final_response')}
second=agent.run_conversation('chào bạn',conversation_history=copy.deepcopy(first.get('messages',[])))
assert second.get('final_response'),{k:v for k,v in second.items() if k in ('error','status','final_response')}
after_load=compute_session_context_breakdown(agent,[]).get('request_load')
assert after_load and after_load['input_characters']==8,after_load
assert 48<=after_load['active_neurons']<=after_load['allocated_neurons']<=128,after_load
assert after_load.get('source')=='measured backend workset matched by request ID'
ledger=db.get_messages(session)
users=[row.get('content') for row in ledger if row.get('role')=='user']
assert 'hi' in users and 'chào bạn' in users,users
assert len(observed)==2,{'calls':observed,'responses':[first.get('final_response'),second.get('final_response')],'home':str(home)}
for expected,request in zip(('hi','chào bạn'),observed):
    assert request['tools'] in (None,[]) and request['model']=='scarx-generation1'
    assert bool(request['stream']) is args.stream
    texts=[m.get('content') for m in request['messages'] if m.get('role')=='user']
    assert texts==[expected],request
    assert not any(m.get('role')=='system' and m.get('content') for m in request['messages']),request
record={'real_agent_initialized':True,'stream_requested':args.stream,'host_context_buffer':agent.context_compressor.context_length,'memory_display':memory_display,'last_request_load':after_load,
 'neural_context_capacity_claim':None,'stateful_chat':True,'tools':len(agent.tools),'calls':observed,
 'responses':[first['final_response'],second['final_response']],'ledger_user_messages':users,
 'elapsed_seconds':time.monotonic()-start,'model_weights_updated_by_this_check':False,'neural_code_modified_by_hermes_adapter':False,'created_utc':datetime.now(timezone.utc).isoformat()}
(root/('artifacts/hermes-stateful-stream-check.json' if args.stream else 'artifacts/hermes-stateful-check.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('STATEFUL_RESULT '+json.dumps(record,ensure_ascii=False))
