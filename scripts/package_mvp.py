"""Package only allowlisted project files; never include local/user/Hermes data."""
from pathlib import Path
import hashlib
import json
import zipfile
import sqlite3

root=Path(__file__).resolve().parents[1]
output=root/'artifacts/SCAR-X-MVP-v0.1.zip'
catalog=json.loads((root/'artifacts/mvp-v01-summary.json').read_text(encoding='utf-8'))
acceptance=root/catalog['run']
measured=json.loads((acceptance/'result.json').read_text(encoding='utf-8'))
database=root/catalog['data_dir']/'scarx.sqlite'
with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
    payload=db.execute('SELECT payload FROM mvp_checkpoints WHERE id=?',(measured['learned_checkpoint'],)).fetchone()[0]
assert hashlib.sha256(payload).hexdigest()==measured['restart']['sha256']
demo=root/'artifacts/scarx-demo-v01.checkpoint.zip'
demo.write_bytes(payload)
allowed=[]
for directory in ('src','docs','configs','scripts','tests','experiments','runs','native','integrations'):
    allowed.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json','.jsonl','.txt','.html','.js','.css','.cpp','.yaml'))
allowed.extend(root/name for name in ('README.md','pyproject.toml','run_app.py','Start-SCARX.ps1','Stop-SCARX.ps1','capabilities.json','project.json','AGENTS.md','.gitignore','.ignore'))
allowed.extend(root/'artifacts'/name for name in ('README.md','phase21-summary.json','mvp-v01-summary.json','scarx-demo-v01.checkpoint.zip'))
allowed.extend(p for p in (root/'artifacts/phase21-figures').glob('*') if p.suffix in ('.png','.svg'))
allowed.extend(acceptance/name for name in ('report.md','result.json','manifest.json'))
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(allowed))}
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(allowed)):
        archive.writestr(zipfile.ZipInfo('scar-x/'+path.relative_to(root).as_posix()),path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
    archive.writestr(zipfile.ZipInfo('scar-x/PACKAGE_MANIFEST.json'),json.dumps({'format':'scarx-source-package-v1','sha256':manifest,'excludes':['local/','user data','.env','personal Hermes config','paid API keys','external models'],'capability_note':'OpenAI Chat Completions wire adapter for structured neural commands; not GGUF, language generation, vision or action model; see capabilities.json'},indent=2),compress_type=zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read('scar-x/'+name)).hexdigest()==digest for name,digest in manifest.items())
digest=hashlib.sha256(output.read_bytes()).hexdigest()
(root/'artifacts/SCAR-X-MVP-v0.1.sha256').write_text(digest+'  '+output.name+'\n',encoding='ascii')
print(json.dumps({'package':str(output),'files':len(manifest),'bytes':output.stat().st_size,'sha256':digest},indent=2))
