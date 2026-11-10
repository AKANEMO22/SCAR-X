"""Read-only neural protocol check against the running local backend."""
from pathlib import Path
import json
from urllib.request import urlopen
from openai import OpenAI

root=Path(__file__).resolve().parents[1]
base='http://127.0.0.1:8765'
before=json.load(urlopen(base+'/api/status'))
client=OpenAI(base_url=base+'/v1',api_key='no-key-required',max_retries=0)
try:
    model=client.models.list().data[0]
    message=[{'role':'user','content':'{"operation":"predict","task":"A","signal":2}'}]
    plain=client.chat.completions.create(model=model.id,messages=message)
    chunks=list(client.chat.completions.create(model=model.id,messages=message,stream=True))
    result=json.loads(plain.choices[0].message.content)
    assert result==json.loads(''.join(c.choices[0].delta.content or '' for c in chunks))
    assert result['result']['retrieval_used'] is False and result['learned_reasoning'] is False
    after=json.load(urlopen(base+'/api/status'))
    assert before['updates']==after['updates'] and before['checkpoint']==after['checkpoint']
    evidence={'base_url':base+'/v1','model':model.id,'openai_sdk_nonstream':True,'openai_sdk_stream':True,
              'native_neural_prediction':result['result']['prediction'],'automatic_compute':result['result']['compute'],
              'learned_weights_unchanged_by_predict':True,'vision':False,'tool_actions':False,'natural_language_generation':False,
              'framework':'Python stdlib HTTP + NumPy PlasticCore; not llama.cpp/GGUF',
              'resources':after['resources'],'hermes_config':'Named provider added; previous selected Qwen route preserved. Actual Hermes GUI selection is left to user.'}
    (root/'artifacts/hermes-backend-check.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=True,indent=2))
finally: client.close()
