"""OpenAI Chat Completions wire format for the real, structured SCAR-X core.

This is a model protocol adapter, not an LLM or a language-template chatbot.
Unsupported vision/tool/reasoning requests receive explicit capability errors.
"""
import json
import time
import uuid
from .pilots import LANGUAGE,VISION

MODEL='scarx-signal-v0.1'


class BackendError(ValueError):
    def __init__(self,message,code='unsupported_input',status=422):
        super().__init__(message); self.code=code; self.status=status

    def response(self):
        return {'error':{'message':str(self),'type':'invalid_request_error','param':None,'code':self.code}}


def models(workbench=None):
    result={'object':'list','data':[{'id':MODEL,'object':'model','created':0,'owned_by':'local-scarx',
        'name':'SCAR-X V0.1 · structured neural backend',
        'capabilities':{'vision':False,'tool_calling':False,'reasoning':False,'structured_signal':True},
        'description':'ALIF neural classifier with local learning. JSON tasks only; no natural-language generation.'}]}
    available=workbench.pilots.available() if workbench is not None else {}
    if 'language' in available:
        result['data'].append({'id':LANGUAGE,'object':'model','created':0,'owned_by':'local-scarx',
          'name':'SCAR-X Language LAB · experimental / unstable',
          'capabilities':{'vision':False,'tool_calling':False,'reasoning':False,'word_generation':True},
          'description':'Trained48-ALIF autoregressive word pilot. Quality gate FAILED; can blend/repeat sentences. No answer lookup, no general-chat claim.'})
    if 'vision' in available:
        result['data'].append({'id':VISION,'object':'model','created':0,'owned_by':'local-scarx',
          'name':'SCAR-X Digits LAB · digits only',
          'capabilities':{'vision':True,'tool_calling':False,'reasoning':False},
          'description':'Trained48-ALIF handwritten digit classifier,89.2% reused heldoutdiagnostic. NOT screenshot/OCR/visualQA. Returns formatted classification data.'})
    return result


def chat(workbench,data):
    if data.get('model') in (LANGUAGE,VISION):return chat_pilot(workbench,data)
    if data.get('model')!=MODEL: raise BackendError('Model không tồn tại. Dùng scarx-signal-v0.1.','model_not_found',404)
    if data.get('n',1)!=1: raise BackendError('Chỉ hỗ trợ n=1.','unsupported_parameter',400)
    if type(data.get('stream',False)) is not bool: raise BackendError('stream phải là boolean.','invalid_parameter',400)
    if data.get('tool_choice') not in (None,'none','auto'): raise BackendError('SCAR-X chưa có tool/action policy.','tools_not_supported')
    messages=data.get('messages')
    if not isinstance(messages,list) or not 1<=len(messages)<=256: raise BackendError('messages phải có 1–256 message.','invalid_messages',400)
    users=[]
    for message in messages:
        if not isinstance(message,dict): raise BackendError('Message không hợp lệ.','invalid_messages',400)
        content=message.get('content')
        if isinstance(content,list):
            if any(not isinstance(part,dict) or part.get('type')!='text' for part in content):
                raise BackendError('Vision/audio chưa được huấn luyện trong SCAR-X V0.1. Không gửi ảnh vào backend này.','vision_not_supported')
            content='\n'.join(part.get('text','') for part in content)
        if message.get('role')=='user':
            if not isinstance(content,str): raise BackendError('User content phải là text JSON.','invalid_messages',400)
            users.append(content)
    if not users: raise BackendError('Cần user message chứa JSON tác vụ.','invalid_messages',400)
    text=users[-1].strip()
    if text.startswith('```json') and text.endswith('```'): text=text[7:-3].strip()
    try: command=json.loads(text,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    except (ValueError,TypeError) as exc:
        raise BackendError('SCAR-X hiện chưa hiểu chat tự do. Nhập JSON như {"operation":"predict","task":"A","signal":2}.') from exc
    if not isinstance(command,dict): raise BackendError('Nội dung phải là một JSON object.')
    operation=command.pop('operation','predict')
    if operation=='predict':
        raw=workbench.predict(command,automatic=True)
        result={k:raw[k] for k in ('prediction','probabilities','expected','sample','checkpoint','retrieval_used','prediction_source','expected_source','compute')}
    elif operation=='evaluate': result=workbench.evaluate()
    elif operation=='teach': result=workbench.teach(command)
    elif operation=='checkpoint': result=workbench.checkpoint()
    else: raise BackendError('Chỉ hỗ trợ predict, teach, evaluate và checkpoint; chưa có computer actions.','action_not_supported')
    content=json.dumps({'source':'SCAR-X structured neural backend','operation':operation,'result':result,
                        'language_generation':False,'vision':False,'tool_actions':False,'learned_reasoning':False},ensure_ascii=False,allow_nan=False)
    return format_completion(content,data,MODEL,{'structured_only':True,'history_interpretation':'Only the latest user JSON command is processed; no neural language context claim.'})


def chat_pilot(workbench,data):
    if type(data.get('stream',False)) is not bool or data.get('n',1)!=1:raise BackendError('Chỉ n=1, streamboolean.','invalid_parameter',400)
    if data.get('tool_choice') not in (None,'none','auto'):raise BackendError('Pilot chưa có action policy.','tools_not_supported')
    messages=data.get('messages')
    if not isinstance(messages,list) or not 1<=len(messages)<=256:raise BackendError('messages không hợp lệ.','invalid_messages',400)
    users=[m for m in messages if isinstance(m,dict) and m.get('role')=='user']
    if not users:raise BackendError('Cần user message.','invalid_messages',400)
    content=users[-1].get('content');parts=content if isinstance(content,list) else [{'type':'text','text':content}]
    if any(not isinstance(p,dict) for p in parts):raise BackendError('Content part không hợp lệ.','invalid_messages',400)
    if data['model']==LANGUAGE:
        if any(p.get('type')!='text' for p in parts):raise BackendError('Language LAB chưa có image encoder chung. Chọn Digits LAB chỉ khi thử chữ số.','vision_not_supported')
        prompt='\n'.join(p.get('text','') for p in parts)
        raw=workbench.pilots.language(prompt)
        if not raw['text']:raise BackendError('Neural pilot dừng sớm, chưa tạo được câu trả lời.','pilot_empty_generation')
        return format_completion(raw['text'],data,LANGUAGE,{'experimental':True,'quality_gate_passed':False,'source':raw['source'],'vocabulary_size':raw['vocabulary_size'],'stopped':raw['stopped'],'checkpoint':raw['checkpoint']})
    images=[p for p in parts if p.get('type')=='image_url']
    if len(images)!=1:raise BackendError('Digits LAB cần đúng một ảnhPNG/JPEG dạngdataURI. Chỉ nhận dạng chữ số, không mô tả giao diện.','image_required')
    value=images[0].get('image_url')
    uri=value.get('url') if isinstance(value,dict) else value
    raw=workbench.pilots.vision({'image':uri})
    return format_completion(json.dumps(raw,ensure_ascii=False),data,VISION,{'vision_scope':'handwritten_digits_only','language_generation':False,'presentation':'server-formatted neural classification; no general visual QA'})


def format_completion(content,data,model,metadata):
    identifier='chatcmpl-scarx-'+uuid.uuid4().hex
    completion={'id':identifier,'object':'chat.completion','created':int(time.time()),'model':model,
                'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop','logprobs':None}],
                'scarx':metadata}
    if not data.get('stream'): return completion,None
    # Framed after neural inference; not a fabricated live thinking trace.
    events=[]
    def event(delta,finish=None):
        return {'id':identifier,'object':'chat.completion.chunk','created':completion['created'],'model':model,
                'choices':[{'index':0,'delta':delta,'finish_reason':finish,'logprobs':None}]}
    events.append(event({'role':'assistant','content':''}))
    events.extend(event({'content':content[i:i+160]}) for i in range(0,len(content),160))
    events.append(event({},'stop'))
    payload=''.join('data: '+json.dumps(e,ensure_ascii=False)+'\n\n' for e in events)+'data: [DONE]\n\n'
    return completion,payload.encode('utf-8')
