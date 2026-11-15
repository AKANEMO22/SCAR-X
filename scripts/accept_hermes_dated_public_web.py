"""Exact reported dated public question through real canonical Hermes/SearXNG."""
import asyncio
from datetime import datetime, timezone
import json, os
from pathlib import Path
import subprocess, sys, time, uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.store import Store


async def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=ROOT/'local/dated-public-web-acceptance'/stamp
    project=run/'project';project.mkdir(parents=True)
    subprocess.run(['git','init','--quiet',str(project)],check=True,capture_output=True,timeout=10)
    (project/'README.md').write_text('Owned empty integration project. No private files or reference answers.\n',encoding='utf-8')
    (ROOT/'local/current-dated-public-web-acceptance.txt').write_text(str(run),encoding='utf-8')
    home=Path(os.environ['LOCALAPPDATA'])/'hermes';sys.path.insert(0,str(home/'hermes-agent'))
    from websockets.asyncio.client import connect
    from hermes_cli.gateway_client import GatewayClient
    grant=json.loads(subprocess.check_output([str(home/'bin/hermes.exe'),'--profile','default','gateway','attach','--json'],
        text=True,encoding='utf-8',timeout=20,creationflags=subprocess.CREATE_NO_WINDOW))
    trace=Store(ROOT/'local/hybrid-v1')
    report={'scope':'owned real canonical Hermes chat; public query only; not a learning study',
        'model':'scarx-hybrid','turns':[],'completed':False,'new_training':False,'source_fact_verified_by_harness':False}
    events=[]
    def save():
        (run/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (run/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    try:
        async with connect(grant['url'],subprotocols=grant['protocols'],proxy=None,max_size=8*1024*1024) as ws:
            async with GatewayClient(ws) as client:
                snapshot=await client.rpc('session.create',request_id=uuid.uuid4().hex,profile='default',source='gui',
                    cwd=str(project),cwd_explicit=True,model='scarx-hybrid',provider='custom:scarx-local',
                    base_url='http://127.0.0.1:8775/v1',title='SCAR-X dated public-web acceptance '+stamp,
                    hidden=True,toolsets=['terminal','file','web'],max_turns=6,run_budget=60,
                    client={'name':'scarx-dated-web-acceptance','version':'1'})
                sid=snapshot['session_id'];report['session_id']=sid
                assert Path(snapshot['info']['cwd']).resolve()==project.resolve()
                for query in ('việt nam có bao nhiêu tỉnh thành 2026','là'):
                    row={'query':query,'tool_calls':[],'tool_results':[]}
                    report['turns'].append(row)
                    started=time.monotonic()
                    trace.event('work_trace.activity',{'title':'Kiểm Hermes thật: dated public web','run_id':stamp,
                        'question':query,'new_training':False},'orchestrator')
                    row['admission']=await client.rpc('prompt.submit',session_id=sid,profile='default',text=query,submission_id=uuid.uuid4().hex)
                    save();deadline=time.monotonic()+90
                    while time.monotonic()<deadline:
                        try:frame=await asyncio.wait_for(client.events.get(),min(10,max(.1,deadline-time.monotonic())))
                        except asyncio.TimeoutError:continue
                        if isinstance(frame,Exception):raise frame
                        params=frame.get('params',{});payload=params.get('payload',params)
                        if params.get('session_id',sid)!=sid:continue
                        events.append(frame)
                        kind=params.get('type',frame.get('method'))
                        if kind=='tool.start':
                            call={'name':payload.get('tool_name',payload.get('name')),'tool_call_id':payload.get('tool_call_id'),'args':payload.get('args')}
                            row['tool_calls'].append(call)
                            trace.event('work_trace.activity',{'title':'Hermes gọi tool thật','run_id':stamp,**call},'orchestrator')
                        if kind=='tool.complete':row['tool_results'].append(payload)
                        if kind=='message.complete':
                            row['completion']=payload;row['seconds']=time.monotonic()-started
                            save();break
                        if kind=='error':row['error']=payload;save();break
                        save()
                    else:raise TimeoutError('No terminal Hermes event; never resend same admitted prompt')
                    trace.event('work_trace.activity',{'title':'Hermes dated web: câu trả lời thật','run_id':stamp,
                        'question':query,'actual_answer':row.get('completion',{}).get('text'),
                        'tool_calls':row['tool_calls'],'error':row.get('error'),'model_weights_changed':False},'orchestrator')
                    print(json.dumps({'query':query,'tool_calls':row['tool_calls'],'completion':row.get('completion'),'error':row.get('error')},ensure_ascii=False),flush=True)
                    if row.get('error') or row.get('completion',{}).get('status')=='error':break
                report['completed']=len(report['turns'])==2 and all(t.get('completion') for t in report['turns'])
                save()
    except Exception as exc:
        report.update(completed=False,failure=type(exc).__name__+': '+str(exc));save();raise
    finally:trace.close()
    print(json.dumps({'private_receipts':str(run),'completed':report['completed']},ensure_ascii=False),flush=True)


if __name__=='__main__':asyncio.run(main())
