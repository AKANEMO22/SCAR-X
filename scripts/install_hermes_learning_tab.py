"""Install a scoped disk plugin using Hermes's existing desktop/backend doors."""
from pathlib import Path
from datetime import datetime,timezone
import os,hashlib,json,copy,tempfile
import yaml

root=Path(__file__).resolve().parents[1];home=Path(os.environ['LOCALAPPDATA'])/'hermes'
source=root/'integrations/hermes/learning_plugin';stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
backup=root/'local/hermes-learning-plugin-backups'/stamp;backup.mkdir(parents=True)
mapping={home/'desktop-plugins/scarx-learning/plugin.js':source/'plugin.js',
 home/'plugins/scarx-learning/dashboard/plugin_api.py':source/'dashboard/plugin_api.py',
 home/'plugins/scarx-learning/dashboard/manifest.json':source/'dashboard/manifest.json'}
receipt={'home':str(home),'files':{}}
for target,origin in mapping.items():
    target=target.resolve();assert target.is_relative_to(home.resolve())
    data=origin.read_bytes();old=target.read_bytes() if target.exists() else None
    if old==data:continue
    rel=target.relative_to(home.resolve())
    if old is not None:
        known=[json.loads(p.read_text(encoding='utf-8')).get('files',{}).get(rel.as_posix(),{}).get('after') for p in (root/'local/hermes-learning-plugin-backups').glob('*/receipt.json')]
        if hashlib.sha256(old).hexdigest() not in known:raise ValueError('Existing plugin differs: '+str(rel))
        copy_path=backup/rel;copy_path.parent.mkdir(parents=True,exist_ok=True);copy_path.write_bytes(old)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    receipt['files'][rel.as_posix()]={'before':hashlib.sha256(old).hexdigest() if old is not None else None,'after':hashlib.sha256(data).hexdigest()}
config_path=home/'config.yaml';original=config_path.read_bytes();text=original.decode('utf-8-sig');config=yaml.safe_load(text);expected=copy.deepcopy(config)
plugins=expected.setdefault('plugins',{});enabled=plugins.get('enabled',[])
if not isinstance(enabled,list):raise ValueError('plugins.enabled must be a list; preserve config for review')
if 'scarx-learning' not in enabled:plugins['enabled']=enabled+['scarx-learning']
if isinstance(plugins.get('disabled'),list):plugins['disabled']=[p for p in plugins['disabled'] if p!='scarx-learning']
if expected!=config:
    (backup/'config.yaml').write_bytes(original)
    doc=yaml.compose(text);pairs=doc.value;index=next((i for i,(key,_) in enumerate(pairs) if key.value=='plugins'),None)
    block=yaml.safe_dump({'plugins':plugins},sort_keys=False,allow_unicode=True)
    if index is None:text+='\n'+block
    else:
        start=text.rfind('\n',0,pairs[index][0].start_mark.index)+1
        end=text.rfind('\n',0,pairs[index+1][0].start_mark.index)+1 if index+1<len(pairs) else len(text)
        text=text[:start]+block+'\n'+text[end:]
    assert yaml.safe_load(text)==expected
    fd,tmp=tempfile.mkstemp(dir=home,suffix='.learning-tab.tmp')
    try:
        with os.fdopen(fd,'wb') as f:f.write(text.encode());f.flush();os.fsync(f.fileno())
        if config_path.read_bytes()!=original:raise RuntimeError('Concurrent Hermes config edit')
        os.replace(tmp,config_path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
(backup/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'plugin':'scarx-learning','files_updated':list(receipt['files']),'config_enabled':True,'backup':str(backup),'renderer_rebuild_required':False}))
