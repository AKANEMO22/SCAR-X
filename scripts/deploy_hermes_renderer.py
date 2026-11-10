"""Install the locally built renderer, keeping Electron main/preload unchanged.

Back up every overwritten byte; retain old hashed assets so restoring index.html
recovers the previous renderer. No recursive deletion or package replacement.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,default=root/'local/hermes-renderer-stateful-20261109');args=parser.parse_args()
build=args.build_dir.resolve()
hermes=(Path(os.environ['LOCALAPPDATA'])/'hermes/hermes-agent').resolve()
target=(hermes/'apps/desktop/release-graph-chat-scroll-production/win-unpacked/resources/app.asar.unpacked/dist').resolve()
if not build.is_relative_to((root/'local').resolve()) or not target.is_relative_to(hermes):raise ValueError('Unexpected build/installation path')
if not (build/'index.html').is_file() or not (target/'electron-main.mjs').is_file():raise ValueError('Missing renderer or installed app')
files=sorted(p for p in build.rglob('*') if p.is_file())
if any(p.name in ('electron-main.mjs','electron-preload.js') for p in files):raise ValueError('Only renderer output may be installed')
if not any(b'active_neurons' in p.read_bytes() and b'input_characters' in p.read_bytes() for p in files if p.suffix=='.js'):raise ValueError('Renderer lacks per-request load/neuron display')
backup=root/'local/hermes-renderer-backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');backup.mkdir(parents=True)
electron={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (target/'electron-main.mjs',target/'electron-preload.js')}
receipt={'target':str(target),'build':str(build),'electron_unchanged':electron,'files':{}}
# Write all assets before swapping entrypoint, so live reload cannot see an
# index that refers to files not installed yet.
files.sort(key=lambda p:p.name=='index.html')
for source in files:
    rel=source.relative_to(build);destination=(target/rel).resolve()
    if not destination.is_relative_to(target):raise ValueError('Invalid build member')
    data=source.read_bytes();before=destination.read_bytes() if destination.exists() else None
    if before==data:continue
    if before is not None:
        copy=backup/rel;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(before)
    receipt['files'][rel.as_posix()]={'before':hashlib.sha256(before).hexdigest() if before is not None else None,'after':hashlib.sha256(data).hexdigest()}
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
for name,digest in electron.items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest
(backup/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
(root/'artifacts/hermes-renderer-deployment.json').write_text(json.dumps({'renderer_built':True,'renderer_files_installed':len(receipt['files']),'electron_main_preload_unchanged':True,'rollback':'restore backed-up index.html; old hashed assets retained','backup':str(backup)},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'installed_renderer_files':len(receipt['files']),'electron_unchanged':True,'backup':str(backup)}))
