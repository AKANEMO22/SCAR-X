"""Bounded real canonical Hermes session on an owned project, never private tasks.

Session tickets stay in memory. Persist only this fixture's RPC/events/results.
This is operational acceptance, not a research benchmark or a learning label.
"""

import argparse
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
    parser=argparse.ArgumentParser();parser.add_argument('--long',action='store_true');parser.add_argument('--compress',action='store_true');args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run=ROOT/'local/release-session-acceptance'/stamp
    project=run/'project';project.mkdir(parents=True)
    # A neutral project root avoids inheriting the research checkout's AGENTS.
    subprocess.run(['git','init','--quiet',str(project)],check=True,capture_output=True,timeout=10)
    (project/'README.md').write_text('# Jade Inventory\nAn inventory application for a shop.\nReceipt: JADE-472.\nDevelopment port: 9237.\n',encoding='utf-8')
    home=Path(os.environ['LOCALAPPDATA'])/'hermes';sys.path.insert(0,str(home/'hermes-agent'))
    from websockets.asyncio.client import connect
    from hermes_cli.gateway_client import GatewayClient
    grant=json.loads(subprocess.check_output([str(home/'bin/hermes.exe'),'--profile','default','gateway','attach','--json'],text=True,encoding='utf-8',timeout=20))
    report={'started_epoch':time.time(),'project':str(project),'model':'scarx-hybrid','turns':[],
            'scope':'actual canonical Hermes gateway, owned source fixture; not neural research','eight_hour_run':False}
    events=[];session=None
    def save():
        (run/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (run/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        async with connect(grant['url'],subprotocols=grant['protocols'],proxy=None,max_size=8*1024*1024) as ws:
            async with GatewayClient(ws) as client:
                snapshot=await client.rpc('session.create',request_id=uuid.uuid4().hex,profile='default',source='gui',cwd=str(project),cwd_explicit=True,
                    model='scarx-hybrid',provider='custom:scarx-local',base_url='http://127.0.0.1:8775/v1',
                    title='SCAR-X owned release acceptance '+stamp,hidden=True,toolsets=['terminal','file','web'],max_turns=8,run_budget=60,
                    client={'name':'scarx-release-acceptance','version':'1'})
                session=snapshot['session_id'];report['session_id']=session;report['create_info']=snapshot.get('info',{});save()
                if Path(snapshot['info']['cwd']).resolve()!=project:raise RuntimeError('Created session has a different workspace; no prompt submitted')
                prompts=['Read README.md in the current project. State its application purpose, receipt code and development port. Do not edit files or search the web.']
                if args.long:
                    prompts.append(('Archived unrelated observation; no project configuration changes. '*105)
                        +'\n\nKeep the current project unchanged. Reply briefly with the receipt code and port from README.md.')
                for prompt in prompts:
                    started=time.monotonic();record={'prompt':prompt,'characters':len(prompt)};report['turns'].append(record);save()
                    admission=await client.rpc('prompt.submit',session_id=session,profile='default',text=prompt,
                        submission_id=uuid.uuid4().hex)
                    record['admission']=admission;save()
                    deadline=time.monotonic()+90
                    while time.monotonic()<deadline:
                        try:frame=await asyncio.wait_for(client.events.get(),min(10,max(.1,deadline-time.monotonic())))
                        except asyncio.TimeoutError:continue
                        if isinstance(frame,Exception):raise frame
                        params=frame.get('params',{});payload=params.get('payload',params)
                        if params.get('session_id',session)!=session:continue
                        events.append(frame);save()
                        kind=params.get('type',frame.get('method'))
                        if kind=='message.complete':
                            record['completion']=payload;record['seconds']=time.monotonic()-started;save();break
                        if kind=='error':
                            record['error']=payload;save();break
                    else:raise TimeoutError('Owned turn has no terminal event; do not resend')
                    print(json.dumps({'private_run':str(run),'characters':len(prompt),'completion':record.get('completion'),'error':record.get('error')},ensure_ascii=False),flush=True)
                    if record.get('error') or record.get('completion',{}).get('status')=='error':break
                if args.compress:
                    latest=await client.rpc('session.resume',session_id=session,profile='default',display_projection=True)
                    compacted=await client.rpc('session.mutate',session_id=latest['session_id'],request_id=uuid.uuid4().hex,
                        expected_revision=latest['revision'],expected_generation=latest['execution_generation'],
                        operation='compress',payload={'focus':'Keep project receipt and port'},_timeout=120)
                    report['manual_compression']=compacted;save()
                final=await client.rpc('session.resume',session_id=session,profile='default',display_projection=True)
                (run/'final-snapshot.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
                report['display_projection_present']=isinstance(final.get('display_messages'),list)
                markers=('[PRIOR CONTEXT','[CONTEXT COMPACTION','[END OF PRIOR CONTEXT')
                report['display_handoff_rows']=sum(any(marker in json.dumps(row,ensure_ascii=False) for marker in markers) for row in final.get('display_messages',[]))
                report['final_running']=final.get('running');report['transport_completed']=True;save()
    except Exception as exc:
        report['transport_completed']=False;report['failure_type']=type(exc).__name__;report['failure']=str(exc)[:500];save()
        report['refusal_data']=getattr(exc,'data',None);save()
        print(json.dumps({'private_run':str(run),'failure_type':type(exc).__name__,'session_created':session is not None}),flush=True)
        raise
    print(json.dumps({'private_run':str(run),'session_id':session,'turns':len(report['turns']),
        'display_handoff_rows':report.get('display_handoff_rows'),'final_running':report.get('final_running')}),flush=True)


if __name__=='__main__':asyncio.run(main())
