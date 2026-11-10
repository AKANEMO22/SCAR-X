"""Authenticated Hermes plugin edge to the configured loopback SCAR-X service."""
import asyncio,json
from urllib.parse import urlsplit
from urllib.request import Request,build_opener,HTTPRedirectHandler
from urllib.error import HTTPError,URLError
from fastapi import APIRouter,HTTPException,Request as WebRequest
from hermes_cli.config import load_config_readonly

router=APIRouter()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def configured_endpoint():
    config=load_config_readonly();profile=config.get('providers',{}).get('scarx-local',{})
    base=str(profile.get('api','')).rstrip('/');url=urlsplit(base)
    if profile.get('interaction_mode')!='stateful_chat' or 'scarx-generation1' not in profile.get('models',{}):
        raise HTTPException(409,'Chưa cấu hình provider SCAR-X có trạng thái cho profile này.')
    if url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost') or url.path!='/v1' or url.username or url.password or url.query or url.fragment:
        raise HTTPException(409,'Tab này chỉ kết nối provider SCAR-X loopback đã cấu hình.')
    try:port=url.port
    except ValueError:raise HTTPException(409,'Port SCAR-X không hợp lệ.')
    if port is None:raise HTTPException(409,'Cần port SCAR-X rõ ràng.')
    return base


def forward(path,data=None):
    endpoint=configured_endpoint();encoded=None if data is None else json.dumps(data,ensure_ascii=False).encode()
    if encoded is not None and len(encoded)>20000:raise HTTPException(413,'Tin nhắn quá lớn.')
    try:
        headers={'Content-Type':'application/json'}
        if encoded is not None:
            # Keep SCAR-X's mutation token in this backend, never in renderer
            # state or a log. The new channel retains the existing CSRF guard.
            with build_opener(NoRedirect()).open(endpoint[:-3]+'/api/status',timeout=3) as status_response:
                status=json.loads(status_response.read(1_000_000))
            if not isinstance(status.get('token'),str):raise HTTPException(502,'SCAR-X chưa cấp phiên ghi dữ liệu.')
            headers['X-SCARX-Token']=status['token']
        request=Request(endpoint+path,encoded,headers=headers)
        with build_opener(NoRedirect()).open(request,timeout=12 if data is not None else 3) as response:
            raw=response.read(1_000_001)
        if len(raw)>1_000_000:raise HTTPException(502,'Phản hồi SCAR-X vượt giới hạn.')
        return json.loads(raw)
    except HTTPError as exc:
        try:
            value=json.loads(exc.read(8000));error=value.get('error','SCAR-X từ chối yêu cầu.')
            message=error.get('message') if isinstance(error,dict) else str(error)
        except (ValueError,UnicodeError):message='SCAR-X chưa hỗ trợ yêu cầu này.'
        raise HTTPException(exc.code if exc.code in (400,403,409,413,422,507) else 502,message[:500])
    except (URLError,TimeoutError,OSError):raise HTTPException(503,'Chưa kết nối được SCAR-X. Kiểm tra server localhost rồi thử lại.')
    except (ValueError,UnicodeError):raise HTTPException(502,'Phản hồi SCAR-X không hợp lệ.')


@router.get('/feed')
async def learning_feed():return await asyncio.to_thread(forward,'/learning/feed')


@router.post('/message')
async def learning_message(request:WebRequest):
    raw=await request.body()
    if len(raw)>20000:raise HTTPException(413,'Tin nhắn quá lớn.')
    try:body=json.loads(raw)
    except ValueError:raise HTTPException(400,'Cần JSON hợp lệ.')
    if not isinstance(body,dict):raise HTTPException(400,'Cần JSON object.')
    return await asyncio.to_thread(forward,'/learning/message',body)
