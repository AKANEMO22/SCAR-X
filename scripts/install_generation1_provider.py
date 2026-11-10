"""Replace retired SCAR-X entries, select G1, preserve other providers/secrets."""
from pathlib import Path
from datetime import datetime,timezone
import copy
import json
import os
import tempfile
import yaml

root=Path(__file__).resolve().parents[1]
path=Path(os.environ['LOCALAPPDATA'])/'hermes/config.yaml'
original=path.read_bytes();text=original.decode('utf-8-sig');current=yaml.safe_load(text)
if not isinstance(current,dict):raise ValueError('Invalid Hermes config')
expected=copy.deepcopy(current)
snippet=yaml.safe_load((root/'integrations/hermes/provider.yaml').read_text(encoding='utf-8'))
profile=snippet['providers']['scarx-local']
expected.setdefault('providers',{})['scarx-local']=profile
expected.setdefault('model',{}).update({'default':'scarx-generation1','provider':'custom:scarx-local','supports_vision':True,'context_length':4096})
for name,section in snippet['model_overrides'].items():
    expected.setdefault('model_overrides',{}).setdefault(name,{}).update(section)
assert {k:v for k,v in expected.items() if k not in ('providers','model','model_overrides')}=={k:v for k,v in current.items() if k not in ('providers','model','model_overrides')}
document=yaml.compose(text);pairs=document.value;replacements=[]
for target in ('providers','model','model_overrides'):
    index=next((i for i,(key,_) in enumerate(pairs) if key.value==target),None)
    block=yaml.safe_dump({target:expected[target]},sort_keys=False,allow_unicode=True)
    if index is None:replacements.append((len(text),len(text),'\n'+block))
    else:
        start=text.rfind('\n',0,pairs[index][0].start_mark.index)+1
        end=text.rfind('\n',0,pairs[index+1][0].start_mark.index)+1 if index+1<len(pairs) else len(text)
        replacements.append((start,end,block+'\n'))
for start,end,block in sorted(replacements,reverse=True):text=text[:start]+block+text[end:]
if yaml.safe_load(text)!=expected:raise ValueError('Config mismatch; original preserved')
if expected==current:print(json.dumps({'already_configured':True,'model':'scarx-generation1'}));raise SystemExit(0)
backup=path.parent/'backups'/('config-before-generation1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.yaml')
backup.parent.mkdir(exist_ok=True);backup.write_bytes(original)
fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.scarx.tmp')
try:
    with os.fdopen(fd,'wb') as stream:stream.write(text.encode('utf-8'));stream.flush();os.fsync(stream.fileno())
    if path.read_bytes()!=original:raise RuntimeError('Hermes config changed concurrently')
    os.replace(tmp,path)
finally:
    if os.path.exists(tmp):os.unlink(tmp)
print(json.dumps({'model':'scarx-generation1','provider':'custom:scarx-local','scarx_choices':1,'backup':str(backup),'llamacpp_enabled':expected.get('local_runtime',{}).get('enabled')},indent=2))
