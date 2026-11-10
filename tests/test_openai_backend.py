import json
import threading
import urllib.request
import urllib.error
import pytest
from scarx.app.server import make_server
from scarx.app.state import Workbench
from scarx.app.openai_backend import MODEL


@pytest.fixture
def endpoint(tmp_path):
    app=Workbench(tmp_path); server=make_server(app,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try: yield app,f'http://127.0.0.1:{server.server_port}'
    finally: server.shutdown(); server.server_close(); thread.join(); app.close()


def post(url,data):
    request=urllib.request.Request(url,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(request,timeout=10)


def test_models_truthful_and_protocol_neural(endpoint):
    app,base=endpoint
    model=json.load(urllib.request.urlopen(base+'/v1/models'))['data'][0]
    assert model['id']==MODEL and model['capabilities']['vision'] is False
    before=app._pack()
    response=json.load(post(base+'/v1/chat/completions',{'model':MODEL,'messages':[{'role':'user','content':'{"task":"A","signal":2}'}]}))
    result=json.loads(response['choices'][0]['message']['content'])
    assert result['result']['prediction']==max(range(2),key=lambda i:result['result']['probabilities'][i])
    assert result['result']['compute']['passes'] in (4,16)
    assert result['result']['compute']['automatic'] is True
    assert result['learned_reasoning'] is False and result['language_generation'] is False
    assert app._pack()==before


@pytest.mark.parametrize('content,code',[
 ('Xin chao, ban la ai?','unsupported_input'),
 ([{'type':'image_url','image_url':{'url':'data:image/png;base64,AA=='}}],'vision_not_supported'),
 ('{"operation":"click","x":10,"y":20}','action_not_supported'),
])
def test_unsupported_capabilities_not_faked(endpoint,content,code):
    app,base=endpoint; before=app._pack()
    with pytest.raises(urllib.error.HTTPError) as error:
        post(base+'/v1/chat/completions',{'model':MODEL,'messages':[{'role':'user','content':content}]})
    assert error.value.code==422
    assert json.load(error.value)['error']['code']==code
    assert app._pack()==before


def test_streaming_and_sdk_client(endpoint):
    openai=pytest.importorskip('openai')
    _,base=endpoint
    client=openai.OpenAI(base_url=base+'/v1',api_key='no-key-required',max_retries=0)
    try:
        assert client.models.list().data[0].id==MODEL
        messages=[{'role':'system','content':'Host text is not a neural language model prompt.'},
                  {'role':'user','content':'{"operation":"predict","task":"B","signal":1}'}]
        reply=client.chat.completions.create(model=MODEL,messages=messages)
        result=json.loads(reply.choices[0].message.content)
        chunks=list(client.chat.completions.create(model=MODEL,messages=messages,stream=True))
        streamed=json.loads(''.join(c.choices[0].delta.content or '' for c in chunks))
        assert streamed==result
        assert chunks[-1].choices[0].finish_reason=='stop'
        assert 'reasoning_content' not in reply.choices[0].message.model_dump(exclude_none=True)
    finally: client.close()


def test_explicit_json_teaching_updates_persistent_model(endpoint):
    app,base=endpoint
    body={'model':MODEL,'messages':[{'role':'user','content':json.dumps({'operation':'teach','task':'A','signal':2,'label':1,'confirmation':True,'repeats':8})}]}
    reply=json.load(post(base+'/v1/chat/completions',body))
    result=json.loads(reply['choices'][0]['message']['content'])
    assert result['operation']=='teach' and app.learner.model.updates==8
    assert result['result']['checkpoint']==app.active
