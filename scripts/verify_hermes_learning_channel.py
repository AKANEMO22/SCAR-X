"""Real Hermes plugin discovery/auth -> live isolated SCAR-X HTTP routes."""
from pathlib import Path
import os,sys,json,shutil,threading,uuid,urllib.request
root=Path(__file__).resolve().parents[1];hermes=Path(os.environ['LOCALAPPDATA'])/'hermes/hermes-agent'
home=root/'local'/('hermes-learning-qa-'+uuid.uuid4().hex[:8]);home.mkdir(parents=True)
os.environ['HERMES_HOME']=str(home)
for key in list(os.environ):
    if key.endswith(('_API_KEY','_AUTH_TOKEN')):os.environ.pop(key)
sys.path.insert(0,str(hermes));sys.path.insert(0,str(root/'src'))
import yaml
from scarx.app.generation import Generation
from scarx.app.generation_server import make_server
from scarx.app.learning_channel import feed
from scarx.runtime.limits import apply_memory_limit
from scarx.evaluation.runs import new_run,write_json
from fastapi.testclient import TestClient

if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
cap=apply_memory_limit();app=Generation(home/'scarx',background=False)
server=make_server(app,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
config=yaml.safe_load((root/'integrations/hermes/provider.yaml').read_text(encoding='utf-8'))
config['providers']['scarx-local']['api']=f'http://127.0.0.1:{server.server_port}/v1'
config['plugins']={'enabled':['scarx-learning']}
(home/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
source=root/'integrations/hermes/learning_plugin/dashboard';target=home/'plugins/scarx-learning/dashboard';target.mkdir(parents=True)
for name in ('manifest.json','plugin_api.py'):shutil.copyfile(source/name,target/name)
run=new_run(root,'hermes-learning-channel-acceptance',{'protocol':'experiments/learning-channel-acceptance.md','private_session_used':False},data_description='Public deterministic arithmetic teaching and raw hi fixture only')
try:
    from hermes_cli import web_server
    client=TestClient(web_server.app,base_url='http://127.0.0.1:8888',client=('127.0.0.1',50000))
    path='/api/plugins/scarx-learning'
    rejected=client.get(path+'/feed');assert rejected.status_code in (401,403),rejected.status_code
    headers={'X-Hermes-Session-Token':web_server._SESSION_TOKEN}
    response=client.get(path+'/feed',headers=headers);assert response.status_code==200,response.text[:500]
    before=app.model.core.fingerprint()
    response=client.post(path+'/message',headers=headers,json={'kind':'chat','client_id':'qa-question-0001','text':'QA: 1 + 1 bằng mấy?'})
    assert response.status_code==200,response.text[:500]
    assert app.model.core.fingerprint()==before and not app.dialogue.lessons
    question=next(m for m in response.json()['feed']['messages'] if m['kind']=='question' and m['question_status']=='open')
    reply={'kind':'reply','client_id':'qa-reply-0001','question_id':question['question_id'],'session_id':question['session_id'],'answer':'3'}
    wrong=client.post(path+'/message',headers=headers,json=reply);assert wrong.status_code==400,wrong.text
    assert not app.dialogue.lessons
    reply.update(client_id='qa-reply-0002',answer='2')
    good=client.post(path+'/message',headers=headers,json=reply);assert good.status_code==200,good.text[:500]
    updates=app.model.core.updates
    duplicate=client.post(path+'/message',headers=headers,json=reply)
    assert duplicate.status_code==200 and duplicate.json()['deduplicated'] and app.model.core.updates==updates
    assert any(m.get('independently_verified') for m in feed(app)['messages'])
    payload={'model':'scarx-generation1','scarx_question_surface':'learning_tab','scarx_session_id':'qa-main',
      'messages':[{'role':'user','content':'QA: 2 + 2 bằng mấy?'}]}
    req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    result=json.load(urllib.request.urlopen(req,timeout=10));text=result['choices'][0]['message']['content']
    assert 'chuyển sang tab Hỏi để học' in text and not text.startswith('[Hỏi để học]')
    saved=app.export();messages=len(feed(app)['messages']);app._unpack(saved)
    assert len(feed(app)['messages'])==messages
    report={'real_plugin_discovery_and_auth':True,'unauthenticated_rejected':True,'real_scarx_http':True,
      'ordinary_message_not_training_label':True,'false_numeric_feedback_rejected':True,'correct_feedback_verified':True,
      'same_client_id_exactly_once':True,'main_chat_nonblocking_routing_notice':True,'history_retained_across_neural_restore':True,
      'controller_questions_separate_from_neural_output':True,'ram_bytes':app.process.memory_info().rss,'memory_cap':cap,
      'native_visual_verification':'recorded separately; this is API acceptance','learning_quality_claim':False}
    write_json(run/'result.json',report);write_json(root/'artifacts/hermes-learning-channel-check.json',{'run':run.relative_to(root).as_posix(),**report})
    (run/'report.md').write_text('# Learning channel engineering acceptance\n\n'+json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('RESULT '+json.dumps(report),flush=True)
finally:server.shutdown();server.server_close();thread.join();app.close()
