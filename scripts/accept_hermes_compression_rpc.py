"""Short real-session context acceptance via Desktop's canonical mutation wire.

Only the explicitly owned Jade fixture session is resumed. No user history is
rewound, imported or labelled as training data; no eight-hour test is opened.
"""

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
    original=ROOT/'local/release-session-acceptance/20261114T141928208486Z'
    old=json.loads((original/'result.json').read_text(encoding='utf-8'))
    project=Path(old['project']).resolve();assert project.is_relative_to(original)
    assert (project/'README.md').read_text(encoding='utf-8')=='# Jade Inventory\nAn inventory application for a shop.\nReceipt: JADE-472.\nDevelopment port: 9237.\n'
    run=ROOT/'local/release-context-acceptance'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run.mkdir(parents=True)
    home=Path(os.environ['LOCALAPPDATA'])/'hermes';sys.path.insert(0,str(home/'hermes-agent'))
    from websockets.asyncio.client import connect
    from hermes_cli.gateway_client import GatewayClient
    grant=json.loads(subprocess.check_output([str(home/'bin/hermes.exe'),'--profile','default','gateway','attach','--json'],text=True,encoding='utf-8',timeout=20))
    report={'scope':'actual canonical Hermes/SCAR-X on an owned project session; no quality benchmark',
            'parent_run':str(original),'turns':[],'eight_hour_run':False,'new_training':False};events=[]
    def save():
        (run/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (run/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        async with connect(grant['url'],subprotocols=grant['protocols'],proxy=None,max_size=8*1024*1024) as ws:
            async with GatewayClient(ws) as client:
                snapshot=await client.rpc('session.resume',session_id=old['session_id'],profile='default',display_projection=True)
                assert Path(snapshot['info']['cwd']).resolve()==project and not snapshot['running']
                sid=snapshot['session_id'];report['session_id']=sid;save()
                for i in range(5):
                    prompt=(('Unrelated archived note '+str(i)+': no project settings changed. ')*125
                        +'\n\nAnswer briefly with the receipt code and current port already read from the project. Do not edit files or browse.')
                    row={'input_characters':len(prompt)};report['turns'].append(row);save()
                    admission=await client.rpc('prompt.submit',session_id=sid,profile='default',submission_id=uuid.uuid4().hex,text=prompt)
                    row['admission']=admission;save();deadline=time.monotonic()+90
                    while time.monotonic()<deadline:
                        try:frame=await asyncio.wait_for(client.events.get(),min(10,max(.1,deadline-time.monotonic())))
                        except asyncio.TimeoutError:continue
                        if isinstance(frame,Exception):raise frame
                        params=frame.get('params',{})
                        if params.get('session_id',sid)!=sid:continue
                        events.append(frame);save()
                        kind=params.get('type');payload=params.get('payload',{})
                        if kind=='message.complete':row['completion']=payload;save();break
                        if kind=='error':row['error']=payload;save();break
                    else:raise TimeoutError('Owned admission outcome unknown; do not resend')
                    print(json.dumps({'run':run.name,'step':i,'completion':row.get('completion'),'error':row.get('error')},ensure_ascii=False),flush=True)
                    if row.get('error') or row.get('completion',{}).get('status')=='error':break
                snapshot=await client.rpc('session.resume',session_id=sid,profile='default',display_projection=True)
                assert not snapshot['running']
                request={'session_id':snapshot['session_id'],'request_id':uuid.uuid4().hex,
                         'expected_revision':snapshot['revision'],'expected_generation':snapshot['execution_generation'],
                         'operation':'compress','payload':{'focus':'Keep project receipt code and current port'}}
                report['compress_request']=request;save()
                report['compression']=await client.rpc('session.mutate',_timeout=120,**request);save()
                final=await client.rpc('session.resume',session_id=sid,profile='default',display_projection=True)
                (run/'final-snapshot.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
                markers=('[PRIOR CONTEXT','[CONTEXT COMPACTION','[END OF PRIOR CONTEXT')
                report['display_handoff_rows']=sum(any(m in json.dumps(row,ensure_ascii=False) for m in markers) for row in final.get('display_messages',[]))
                report['display_projection_present']=isinstance(final.get('display_messages'),list)
                report['final_running']=final['running'];report['transport_completed']=True;save()
    except Exception as exc:
        report['failure_type']=type(exc).__name__;report['failure']=str(exc)[:500];report['refusal_data']=getattr(exc,'data',None);save()
        print(json.dumps({'run':str(run),'failure_type':type(exc).__name__}),flush=True);raise
    print(json.dumps({'private_run':str(run),'display_handoff_rows':report.get('display_handoff_rows'),
                     'compression':report.get('compression'),'turns':len(report['turns'])},ensure_ascii=False),flush=True)


if __name__=='__main__':asyncio.run(main())
