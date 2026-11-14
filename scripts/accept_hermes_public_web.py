"""Actual canonical Hermes chat for the reported public query and its follow-up.

Owned hidden session, existing GPU model and real SearXNG. No private workspace
or model teaching; no assertion that SERP snippets prove the latest episode.
"""

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


async def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = ROOT / 'local/public-web-acceptance' / stamp
    project = run / 'project'
    project.mkdir(parents=True)
    subprocess.run(['git', 'init', '--quiet', str(project)], check=True, capture_output=True)
    (project / 'README.md').write_text('Owned public-web acceptance project; no private files.\n', encoding='utf-8')
    (ROOT / 'local/current-public-web-acceptance.txt').write_text(str(run), encoding='utf-8')
    home = Path(os.environ['LOCALAPPDATA']) / 'hermes'
    sys.path.insert(0, str(home / 'hermes-agent'))
    from websockets.asyncio.client import connect
    from hermes_cli.gateway_client import GatewayClient
    grant = json.loads(subprocess.check_output(
        [str(home / 'bin/hermes.exe'), '--profile', 'default', 'gateway', 'attach', '--json'],
        text=True, encoding='utf-8', timeout=20,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
    report = {'scope': 'actual canonical Hermes/SearXNG on owned project; not a research benchmark',
              'model': 'scarx-hybrid', 'turns': [], 'completed': False}
    events = []
    def save():
        (run / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (run / 'events.json').write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    try:
        async with connect(grant['url'], subprotocols=grant['protocols'], proxy=None, max_size=8*1024*1024) as ws:
            async with GatewayClient(ws) as client:
                snapshot = await client.rpc('session.create', request_id=uuid.uuid4().hex,
                    profile='default', source='gui', cwd=str(project), cwd_explicit=True,
                    model='scarx-hybrid', provider='custom:scarx-local', base_url='http://127.0.0.1:8775/v1',
                    title='SCAR-X owned public-web acceptance '+stamp, hidden=True,
                    toolsets=['terminal', 'file', 'web'], max_turns=6, run_budget=60,
                    client={'name':'scarx-web-route-acceptance','version':'1'})
                session = snapshot['session_id']
                report['session_id'] = session
                assert Path(snapshot['info']['cwd']).resolve() == project.resolve()
                for query in ('jojo ra tập mới nhất là tập bn', 'tìm kiếm đi'):
                    turn = {'query':query, 'tool_calls': [], 'tool_results': []}
                    report['turns'].append(turn)
                    started = time.monotonic()
                    admission = await client.rpc('prompt.submit', session_id=session, profile='default',
                        text=query, submission_id=uuid.uuid4().hex)
                    turn['admission'] = admission
                    save()
                    deadline = time.monotonic()+90
                    while time.monotonic() < deadline:
                        try:
                            frame = await asyncio.wait_for(client.events.get(), min(10, max(.1, deadline-time.monotonic())))
                        except asyncio.TimeoutError:
                            continue
                        if isinstance(frame, Exception):
                            raise frame
                        params=frame.get('params', {});payload=params.get('payload', params)
                        if params.get('session_id', session) != session:
                            continue
                        events.append(frame)
                        kind=params.get('type', frame.get('method'))
                        if kind == 'tool.start':
                            turn['tool_calls'].append({'name':payload.get('tool_name', payload.get('name')),
                                'tool_call_id':payload.get('tool_call_id'), 'args':payload.get('args')})
                        if kind == 'tool.complete':
                            turn['tool_results'].append(payload)
                        if kind == 'message.complete':
                            turn['completion']=payload;turn['seconds']=time.monotonic()-started
                            save();break
                        if kind == 'error':
                            turn['error']=payload;save();break
                        save()
                    else:
                        raise TimeoutError('No terminal event; do not resend the admitted request')
                    print(json.dumps({'query':query,'tool_calls':turn['tool_calls'],
                                      'completion':turn.get('completion'),'error':turn.get('error')},ensure_ascii=False),flush=True)
                    if turn.get('error') or turn.get('completion', {}).get('status')=='error':
                        break
                report['completed'] = len(report['turns']) == 2 and all(t.get('completion') for t in report['turns'])
                save()
    except Exception as exc:
        report.update(failure=type(exc).__name__+': '+str(exc), completed=False)
        save()
        raise
    print(json.dumps({'private_receipts':str(run),'completed':report['completed']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    asyncio.run(main())
