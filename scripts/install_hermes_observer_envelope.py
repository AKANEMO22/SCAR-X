"""One-file CAS update, preserving all model workers and private memories."""

from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
BASELINE='7dbef5a4747ff867b1af4235194b8c831a4383d62f53a4adaf0760e6f9f91c52'


def main():
    source=ROOT/'integrations/hermes/hybrid_plugin/learning_hooks.py'
    target=Path(os.environ['LOCALAPPDATA'])/'hermes/plugins/scarx-hybrid/learning_hooks.py'
    candidate=source.read_bytes();compile(candidate,str(source),'exec')
    before=target.read_bytes()
    if before==candidate:print(json.dumps({'already_installed':True}));return
    if hashlib.sha256(before).hexdigest()!=BASELINE:raise ValueError('Installed observer changed; refusing overwrite')
    private=ROOT/'local/hybrid-v1/hermes-backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    private.mkdir(parents=True);(private/'learning_hooks.py').write_bytes(before)
    fd,name=tempfile.mkstemp(dir=target.parent,prefix='.scarx-observer-',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as out:out.write(candidate);out.flush();os.fsync(out.fileno())
        if target.read_bytes()!=before:raise RuntimeError('Concurrent observer edit')
        os.replace(name,target)
    finally:
        if os.path.exists(name):os.unlink(name)
    receipt={'before_sha256':BASELINE,'after_sha256':hashlib.sha256(candidate).hexdigest(),
             'code_files_changed':1,'private_state_or_model_workers_changed':False,
             'Hermes_gateway_reload_required':True}
    (private/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(dict(receipt,private_backup=str(private))))


if __name__=='__main__':main()
