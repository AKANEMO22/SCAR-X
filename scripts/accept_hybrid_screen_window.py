"""Real selected-window capture on an owned, silent FFplay fixture.

No desktop-wide capture, microphone, speaker loopback, user window control,
training or long endurance. Stop the owned stream and player in finally.
"""

from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import uuid
import psutil

ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8775'


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run=ROOT/'local/screen-window-acceptance'/stamp;run.mkdir(parents=True)
    token=None;stream=None;player=None
    def api(path,body=None,session=None):
        headers={'X-SCARX-Backend':'hermes'}
        if body is not None:headers.update({'Content-Type':'application/json','X-SCARX-Token':token or ''})
        if session:headers['X-SCARX-Session']=session
        with urlopen(Request(BASE+path,data=json.dumps(body,ensure_ascii=False).encode() if body is not None else None,headers=headers),timeout=90 if path.startswith('/v1/') else 15) as response:
            return json.load(response)
    status=api('/api/status?after=2147483647');token=status['token']
    if not status['ready'] or status.get('cognition',{}).get('phase') in ('serving','observing'):
        raise RuntimeError('User work or another capture active; no fixture started')
    if any(s.get('kind')=='screen' and s.get('status') in ('starting','sharing') for s in status['streams']):
        raise RuntimeError('Another screen share is active; no fixture started')
    sample=ROOT/'local/hybrid-v1/samples/engineering-vi-video.mkv'
    assert hashlib.sha256(sample.read_bytes()).hexdigest()=='c69a1d1b5d873895aa682294a5a5220cb09e424ef749738c3726275d2bcc211b'
    ffplay=shutil.which('ffplay');assert ffplay
    session='screen-acceptance-'+uuid.uuid4().hex;title='SCAR-X owned screen fixture '+uuid.uuid4().hex[:10]
    report={'scope':'owned silent video window; actual gateway + installed Hermes dispatcher',
            'fixture_session':session,'system_audio':False,'microphone_used':False,'new_training':False,
            'source_sample_sha256':hashlib.sha256(sample.read_bytes()).hexdigest(),'gateway_pid':status['pid']}
    def save():
        (run/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # Use the registered Hermes runtime for its published tool-dispatch path.
    sys.path.insert(0,str(ROOT/'src'))
    from scarx.evaluation.hermes_resource_roots import verified_gateway_pid
    hermes=psutil.Process(verified_gateway_pid(Path.home()/'AppData/Local/hermes'))
    env=hermes.environ();env['PYTHONIOENCODING']='utf-8'
    code='''import json,sys
from model_tools import get_tool_definitions,handle_function_call
from hermes_cli.lifecycle import invoke_hook
from agent.api_request_hooks import ApiRequestHooksMixin
get_tool_definitions(quiet_mode=True)
sid=sys.argv[1];args=json.loads(sys.argv[2]);task='owned-screen-'+sid
invoke_hook('pre_api_request',model='scarx-hybrid',base_url='http://127.0.0.1:8775/v1',session_id=sid,turn_id='capture',task_id=task,request=ApiRequestHooksMixin()._api_request_payload_for_hook({'tools':[{'function':{'name':'scarx_video'}}]}),user_message='Owned window capture acceptance')
value=handle_function_call('scarx_video',args,task_id=task,session_id=sid,turn_id='capture',tool_call_id='owned-'+args['action'])
print('SCARX_RESULT '+value)
'''
    (run/'Hermes-dispatch-driver.py').write_text(code,encoding='utf-8')
    def tool(args):
        result=subprocess.run([hermes.exe(),str(run/'Hermes-dispatch-driver.py'),session,json.dumps(args)],cwd=hermes.cwd(),env=env,capture_output=True,text=True,encoding='utf-8',timeout=30)
        (run/(args['action']+'.stdout.log')).write_text(result.stdout,encoding='utf-8');(run/(args['action']+'.stderr.log')).write_text(result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError('Installed Hermes dispatcher failed: '+args['action'])
        line=next((line for line in reversed(result.stdout.splitlines()) if line.startswith('SCARX_RESULT ')),None)
        if not line:raise RuntimeError('Missing actual dispatcher reply')
        reply=json.loads(line.removeprefix('SCARX_RESULT '))
        if reply.get('error'):raise RuntimeError('Installed Hermes tool returned an error: '+str(reply['error'])[:300])
        return reply
    try:
        with (run/'player.log').open('wb') as log:
            player=subprocess.Popen([ffplay,'-hide_banner','-loglevel','error','-an','-autoexit','-window_title',title,'-x','640','-y','420',str(sample)],stdout=log,stderr=log,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        created=psutil.Process(player.pid).create_time();report['owned_player']={'pid':player.pid,'created':created,'title':title};save()
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            sources=api('/api/screen/sources')['sources'];matches=[s for s in sources if s['title']==title]
            if len(matches)==1:break
            if player.poll() is not None:raise RuntimeError('Owned player exited before selection')
            time.sleep(.2)
        else:raise TimeoutError('No uniquely identified owned window')
        hwnd=matches[0]['id'];report['selected_window']={'id':hwnd,'title':title};save()
        try:api('/api/screen/watch-request',{'hwnd':hwnd,'system_audio':False,'scarx_session_id':session+'-ungranted'})
        except HTTPError as exc:
            denial=json.loads(exc.read(8000));report['ungranted_request_HTTP_status']=exc.code
            report['ungranted_request_denial']=denial
            assert exc.code in (400,403) and 'recent explicit user request' in json.dumps(denial)
        else:raise AssertionError('An ungranted request unexpectedly started capture')
        system='This is an owned local screen-capture engineering check. Only acknowledge the request and wait for window selection. Do not call tools.'
        user='Cho xem cửa sổ màn hình thử do Codex tạo. Chỉ xác nhận và đợi chọn cửa sổ.'
        acknowledgement=api('/v1/chat/completions',{'model':'scarx-hybrid','messages':[{'role':'system','content':system},{'role':'user','content':user}],'max_tokens':16,'tools':[],'tool_choice':'none'},session)
        report['chat_acknowledgement']=acknowledgement['choices'][0]['message'].get('content');save()
        watched=tool({'action':'watch','hwnd':hwnd,'system_audio':False});stream=watched['id'];report['stream_id']=stream;report['watch_reply']=watched;save()
        deadline=time.monotonic()+26;states=[]
        while time.monotonic()<deadline:
            state=next(s for s in api('/api/status?after=2147483647')['streams'] if s['id']==stream)
            states.append({k:state.get(k) for k in ('status','capture_frontier','vision_frontier','error')})
            report['capture_states']=states;save()
            if state['status']=='failed':raise RuntimeError('Actual capture worker failed; retained local log')
            if state['vision_frontier']>=8 or state['status']=='stopped':break
            time.sleep(.5)
        report['stop_reply']=tool({'action':'stop','stream_id':stream});save()
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            state=next(s for s in api('/api/status?after=2147483647')['streams'] if s['id']==stream)
            if state['status'] in ('stopped','failed'):break
            time.sleep(.25)
        report['final_stream']=state;save()
        question='Trong cửa sổ vừa xem, hình chữ nhật bên trái lúc đầu có màu gì? Chỉ trả lời nếu có bằng chứng từ lần xem đó.'
        answer=api('/v1/chat/completions',{'model':'scarx-hybrid','messages':[{'role':'system','content':'Answer in Vietnamese using evidence from the requested screen window. If unavailable say so.'},{'role':'user','content':question}],'max_tokens':128,'tools':[],'tool_choice':'none'},session)
        report['post_stop_answer']=answer['choices'][0]['message'].get('content')
        with sqlite3.connect((ROOT/'local/hybrid-v1/hybrid.sqlite').as_uri()+'?mode=ro',uri=True) as db:
            rows=db.execute("SELECT payload FROM objects WHERE kind='answer' ORDER BY rowid DESC LIMIT 20").fetchall()
            record=next(json.loads(row[0]) for row in rows if json.loads(row[0]).get('question')==question)
            evidence=[dict(id=i,start=start,end=end,text=text,source=source) for i,start,end,text,source in db.execute("SELECT id,start,end,text,source FROM evidence WHERE stream=? AND modality='sampled_frame' ORDER BY start",(stream,))]
        report['post_stop_selected_stream']=record.get('stream');report['post_stop_evidence_count']=len(record['evidence'])
        report['sampled_frame_evidence']=evidence;report['capture_completed']=True;save()
        print(json.dumps({k:report[k] for k in ('stream_id','ungranted_request_HTTP_status','final_stream','post_stop_answer','post_stop_selected_stream','post_stop_evidence_count')},ensure_ascii=False),flush=True)
    except Exception as exc:
        report['failure_type']=type(exc).__name__;report['failure']=str(exc)[:600];save()
        print(json.dumps({'private_run':str(run),'failure_type':type(exc).__name__,'stream_id':stream}),flush=True)
        raise
    finally:
        if stream:
            try:report['finally_stop']=api('/api/screen/stop',{'id':stream})
            except Exception as exc:report['cleanup_error']=type(exc).__name__
        if player and player.poll() is None:
            process=psutil.Process(player.pid)
            if abs(process.create_time()-report['owned_player']['created'])<.01:
                player.terminate();player.wait(timeout=5)
        report['owned_player_terminal']=player is None or player.poll() is not None;save()
    print(json.dumps({'private_run':str(run),'capture_completed':report.get('capture_completed'),'sampled_frames':len(report.get('sampled_frame_evidence',[])),'owned_player_terminal':report['owned_player_terminal']}),flush=True)


if __name__=='__main__':main()
