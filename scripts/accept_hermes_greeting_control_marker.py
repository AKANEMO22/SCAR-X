"""Real Hermes greeting acceptance; no output masking or generated training label."""

import asyncio
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]


async def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run=ROOT/'local/greeting-marker-acceptance'/stamp
    project=run/'project';project.mkdir(parents=True)
    subprocess.run(['git','init','--quiet',str(project)],check=True,capture_output=True,timeout=10)
    (project/'README.md').write_text('Owned greeting integration fixture. No instructions or model answers.\n',encoding='utf-8')
    home=Path(os.environ['LOCALAPPDATA'])/'hermes';sys.path.insert(0,str(home/'hermes-agent'))
    from websockets.asyncio.client import connect
    from hermes_cli.gateway_client import GatewayClient
    grant=json.loads(subprocess.check_output([str(home/'bin/hermes.exe'),'--profile','default','gateway','attach','--json'],text=True,encoding='utf-8',timeout=20))
    report={'scope':'real canonical Hermes/SCAR-X greetings with the installed integration','queries':[],'no_answer_substitution':True,'new_training':False}
    def save():
        (run/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    async with connect(grant['url'],subprotocols=grant['protocols'],proxy=None,max_size=8*1024*1024) as ws:
        async with GatewayClient(ws) as client:
            for index,query in enumerate(('chào bạn','xin chào','hello')):
                snapshot=await client.rpc('session.create',request_id=uuid.uuid4().hex,profile='default',source='gui',
                    cwd=str(project),cwd_explicit=True,model='scarx-hybrid',provider='custom:scarx-local',
                    base_url='http://127.0.0.1:8775/v1',title='SCAR-X greeting marker check '+stamp+' '+str(index),
                    hidden=True,max_turns=4,run_budget=30,client={'name':'scarx-release-acceptance','version':'1'})
                sid=snapshot['session_id'];assert Path(snapshot['info']['cwd']).resolve()==project
                row={'query':query,'session_id':sid};report['queries'].append(row);save()
                row['admission']=await client.rpc('prompt.submit',session_id=sid,profile='default',submission_id=uuid.uuid4().hex,text=query);save()
                deadline=time.monotonic()+60
                while time.monotonic()<deadline:
                    try:frame=await asyncio.wait_for(client.events.get(),10)
                    except asyncio.TimeoutError:continue
                    if isinstance(frame,Exception):raise frame
                    params=frame.get('params',{})
                    if params.get('session_id',sid)!=sid:continue
                    if params.get('type')=='message.complete':row['completion']=params['payload'];save();break
                    if params.get('type')=='error':row['error']=params.get('payload');save();raise RuntimeError('Owned Hermes acceptance turn failed')
                else:raise TimeoutError('Admission outcome unknown; do not resend')
                text=row['completion'].get('text','')
                row['control_marker_present']=any(marker in text for marker in ('[OUT-OF-BAND','[/OUT-OF-BAND','[CONTEXT COMPACTION','[PRIOR CONTEXT'))
                save()
                assert row['completion']['status']=='complete' and text.strip() and not row['control_marker_present']
                print(json.dumps({'private_run':str(run),'query':query,'answer':text,'control_marker_present':False},ensure_ascii=False),flush=True)
    report['completed']=True;save()


if __name__=='__main__':asyncio.run(main())
