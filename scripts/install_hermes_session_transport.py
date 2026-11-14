"""Scoped, backed-up Hermes transport repair; never change SCAR-X weights."""

from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def patch(original):
    before='''    return merge_session_affinity_headers(
        kwargs,
        getattr(agent, "provider", None),
        getattr(agent, "base_url", None),
        getattr(agent, "session_id", None),
    )'''
    after='''    kwargs = merge_session_affinity_headers(
        kwargs,
        getattr(agent, "provider", None),
        getattr(agent, "base_url", None),
        getattr(agent, "session_id", None),
    )
    from agent.scarx_session_transport import bind_transcript_session
    return bind_transcript_session(agent, kwargs)'''
    text=original.decode('utf-8').replace('\r\n','\n')
    if after in text:return original
    if text.count(before)!=1:raise ValueError('Hermes request-builder anchor changed')
    raw=text.replace(before,after,1).encode()
    if b'\r\n' in original:raw=raw.replace(b'\n',b'\r\n')
    compile(raw,'chat_completion_helpers.py','exec');return raw


def main():
    home=Path(os.environ['LOCALAPPDATA'])/'hermes'
    agent=home/'hermes-agent/agent';target=agent/'chat_completion_helpers.py'
    original=target.read_bytes();candidate=patch(original)
    helper=ROOT/'integrations/hermes/scarx_session_transport.py'
    mapping={target:candidate,agent/helper.name:helper.read_bytes()}
    backup=ROOT/'local/hybrid-v1/hermes-backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup.mkdir(parents=True);records=[]
    for path,raw in mapping.items():
        before=path.read_bytes() if path.is_file() else None
        if before==raw:continue
        if before is not None:(backup/path.name).write_bytes(before)
        fd,temp=tempfile.mkstemp(dir=path.parent,prefix='.scarx-transport-')
        try:
            with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            if (path.read_bytes() if path.exists() else None)!=before:raise RuntimeError('Concurrent Hermes edit')
            os.replace(temp,path)
        finally:
            if os.path.exists(temp):os.unlink(temp)
        records.append({'file':path.name,'before_sha256':hashlib.sha256(before).hexdigest() if before else None,'after_sha256':hashlib.sha256(raw).hexdigest()})
    (backup/'receipt.json').write_text(json.dumps({'files':records,'runtime_reload_required':True},indent=2))
    print(json.dumps({'installed_files':records,'private_backup':str(backup),'needs_Hermes_gateway_restart':True}))


if __name__=='__main__':main()
