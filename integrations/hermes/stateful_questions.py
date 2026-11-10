"""One opt-in watcher for questions from the stateful neural backend.

Uses Hermes's existing human clarification callback. No model tool call,
auxiliary LLM, or invented thinking trace. A timeout never becomes a label.
"""
import json
import threading
import weakref
import inspect
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from urllib.error import URLError

_agent_ref=None
_thread=None
_lock=threading.Lock()
_wake=threading.Event()


def _request(agent,path,data=None):
    headers={'Content-Type':'application/json'}
    if getattr(agent,'api_key',None):headers['Authorization']='Bearer '+str(agent.api_key)
    request=Request(str(agent.base_url).rstrip('/')+path,
        None if data is None else json.dumps(data,ensure_ascii=False).encode('utf-8'),headers=headers)
    with urlopen(request,timeout=10 if data is not None else 2) as response:return json.loads(response.read(65536))


def activate_question_watcher(agent):
    global _agent_ref,_thread
    if not getattr(agent,'_stateful_questions_enabled',False) or not callable(getattr(agent,'clarify_callback',None)):return
    with _lock:
        _agent_ref=weakref.ref(agent)
        if _thread is None or not _thread.is_alive():
            _thread=threading.Thread(target=_watch,name='stateful-human-questions',daemon=True);_thread.start()


def _watch():
    last=None
    while not _wake.wait(2):
        with _lock:agent=_agent_ref() if _agent_ref else None
        if agent is None or not getattr(agent,'_stateful_chat_backend',False) or not getattr(agent,'_stateful_questions_enabled',False):
            agent=None;continue
        session=getattr(agent,'session_id',None)
        if not session:agent=None;continue
        stage='poll'
        try:
            result=_request(agent,'/questions?'+urlencode({'session_id':session}))
            question=result.get('question')
            if not question:
                last=None;continue
            key=(session,question['id'])
            if key==last:continue
            last=key
            callback=agent.clarify_callback
            parameters=inspect.signature(callback).parameters
            standalone='standalone' in parameters or any(p.kind==inspect.Parameter.VAR_KEYWORD for p in parameters.values())
            kwargs={'standalone':True} if standalone else {}
            answer=callback('[SCAR-X · hỏi để học] '+question['question'],None,**kwargs)
            if answer is None or not str(answer).strip() or str(answer).startswith('The user did not provide a response'):
                answer='bỏ qua'
            elif not isinstance(answer,str):answer=str(answer)
            stage='feedback'
            _request(agent,'/feedback',{'session_id':session,'question_id':question['id'],'answer':answer})
            if callable(getattr(agent,'_emit_status',None)):
                agent._emit_status('SCAR-X đã ghi phản hồi. Local learning và idle review có guard; chưa bảo đảm giữ mọi kỹ năng.')
        except (URLError,TimeoutError,OSError,ValueError,KeyError,RuntimeError):
            # Endpoint/callback may disappear on a model/session switch. No
            # fallback answer is fabricated and no failed response is learned.
            if stage=='feedback':
                last=None
                if callable(getattr(agent,'_emit_status',None)):
                    agent._emit_status('SCAR-X chưa chấp nhận được phản hồi; câu hỏi vẫn chờ, chưa dùng phản hồi lỗi làm nhãn học.')
            continue
        finally:
            agent=None
