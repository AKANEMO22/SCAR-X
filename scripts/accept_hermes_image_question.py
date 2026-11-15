"""Real Hermes image attachment and grounded reply, private local receipts."""
import asyncio
from datetime import datetime, timezone
import hashlib,json,os
from pathlib import Path
import subprocess,sys,time,uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.store import Store


async def main():
    image=Path(sys.argv[1]).resolve();raw=image.read_bytes()
    if not image.is_file() or len(raw)>2_000_000:raise ValueError('Bounded owned image required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=ROOT/'local/image-question-acceptance'/stamp;run.mkdir(parents=True)
    (ROOT/'local/current-Hermes-image-question-acceptance.txt').write_text(str(run),encoding='utf-8')
    home=Path(os.environ['LOCALAPPDATA'])/'hermes';sys.path.insert(0,str(home/'hermes-agent'))
    # Hermes accepts upload staging paths only under this profile's image cache,
    # never a retained input path from another admission. Copy exact bytes to a
    # fresh owned staging file; do not weaken the native-input ownership check.
    staging_root=(home/'cache/images').resolve()
    if not staging_root.is_dir() or staging_root!=home/'cache/images':
        raise ValueError('Expected existing canonical profile staging cache')
    staged=staging_root/('scarx-image-acceptance-'+uuid.uuid4().hex+'.png')
    with staged.open('xb') as f:f.write(raw)
    from websockets.asyncio.client import connect
    from hermes_cli.gateway_client import GatewayClient
    grant=json.loads(subprocess.check_output([str(home/'bin/hermes.exe'),'--profile','default','gateway','attach','--json'],
        text=True,encoding='utf-8',timeout=20,creationflags=subprocess.CREATE_NO_WINDOW))
    trace=Store(ROOT/'local/hybrid-v1')
    with trace.lock:
        start_cursor=trace.db.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
    report={'scope':'actual canonical Hermes attachment, original reported image; not general vision benchmark',
        'image_sha256':hashlib.sha256(raw).hexdigest(),'new_training':False,
        'reference_answer_in_user_prompt':False,'turns':[],'completed':False,'trace_start_cursor':start_cursor}
    events=[]
    def save():
        (run/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (run/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    try:
        async with connect(grant['url'],subprotocols=grant['protocols'],proxy=None,max_size=8*1024*1024) as ws:
            async with GatewayClient(ws) as client:
                snapshot=await client.rpc('session.create',request_id=uuid.uuid4().hex,profile='default',source='gui',
                    cwd=str(Path.home()),cwd_explicit=True,model='scarx-hybrid',provider='custom:scarx-local',
                    base_url='http://127.0.0.1:8775/v1',title='SCAR-X image question acceptance '+stamp,
                    hidden=True,toolsets=['terminal','file','web'],max_turns=4,run_budget=60,
                    client={'name':'scarx-image-question-acceptance','version':'1'})
                sid=snapshot['session_id'];report['session_id']=sid
                for query,attachments in [('câu này làm  ntn giải thích chi tiết',[{'path':str(staged),'mime':'image/png'}]),('chào bạn',[])]:
                    row={'query':query,'attachments':len(attachments),'tool_calls':[]}
                    report['turns'].append(row);began=time.monotonic()
                    trace.event('work_trace.activity',{'title':'Hermes nhận câu hỏi kèm ảnh thật','run_id':stamp,
                        'question':query,'image_sha256':report['image_sha256'] if attachments else None,'new_training':False},'orchestrator')
                    row['admission']=await client.rpc('prompt.submit',session_id=sid,profile='default',text=query,
                        **({'attachments':attachments} if attachments else {}),submission_id=uuid.uuid4().hex)
                    save();deadline=time.monotonic()+90
                    while time.monotonic()<deadline:
                        try:frame=await asyncio.wait_for(client.events.get(),min(10,max(.1,deadline-time.monotonic())))
                        except asyncio.TimeoutError:continue
                        if isinstance(frame,Exception):raise frame
                        params=frame.get('params',{});payload=params.get('payload',params)
                        if params.get('session_id',sid)!=sid:continue
                        events.append(frame);kind=params.get('type',frame.get('method'))
                        if kind=='tool.start':row['tool_calls'].append({'name':payload.get('tool_name',payload.get('name')),'args':payload.get('args')})
                        if kind=='message.complete':
                            row['completion']=payload;row['seconds']=time.monotonic()-began;save();break
                        if kind=='error':row['error']=payload;save();break
                        save()
                    else:raise TimeoutError('No terminal Hermes event; do not resend admitted image')
                    trace.event('work_trace.activity',{'title':'Hermes image: đầu ra thực','run_id':stamp,'question':query,
                        'actual_answer':row.get('completion',{}).get('text'),'tool_calls':row['tool_calls'],
                        'documentation_grounding_is_not_SNN_learning':True},'orchestrator')
                    print(json.dumps({'query':query,'tool_calls':row['tool_calls'],'completion':row.get('completion'),'error':row.get('error')},ensure_ascii=False),flush=True)
                    if row.get('error'):break
                report['completed']=len(report['turns'])==2 and all(t.get('completion',{}).get('outcome')=='completed' for t in report['turns'])
                save()
    except Exception as exc:
        report.update(completed=False,failure=type(exc).__name__+': '+str(exc));save();raise
    finally:trace.close()
    print(json.dumps({'private_receipts':str(run),'completed':report['completed']},ensure_ascii=False),flush=True)


if __name__=='__main__':asyncio.run(main())
