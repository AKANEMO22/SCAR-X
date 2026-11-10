"""V0.2 experimental release; preserve the V0.1 ZIP and never include local/ data."""
from pathlib import Path
import hashlib
import json
import zipfile

root=Path(__file__).resolve().parents[1];target=root/'artifacts/SCAR-X-V0.2-LAB.zip';files=[]
for base in ('src','docs','scripts','tests','configs','experiments','data','native','integrations','runs'):
    files.extend(p for p in (root/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json','.jsonl','.yaml','.toml','.txt','.css','.html','.js','.cpp'))
files.extend(root/name for name in ('README.md','pyproject.toml','run_app.py','Start-SCARX.ps1','Stop-SCARX.ps1','AGENTS.md','project.json','capabilities.json','.gitignore','.ignore'))
release=json.loads((root/'artifacts/pilot-release.json').read_text(encoding='utf-8'))
for kind,entry in release.items():
    path=(root/entry['checkpoint']).resolve()
    assert path.is_relative_to((root/'artifacts').resolve())
    files.extend(sorted(path.rglob('*')) if path.is_dir() else [path])
files.extend(root/'artifacts'/name for name in ('pilot-release.json','phase21-summary.json','mvp-v01-summary.json','pilot-product-check.json','language-vision-pilot-summary.json','scarx-demo-v01.checkpoint.zip'))
files.append(root/'artifacts/pilots/vision-examples.json')
files.extend((root/'artifacts/phase21-figures').glob('*'))
files=sorted({p.resolve() for p in files if p.is_file()})
manifest={p.relative_to(root.resolve()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(target,'w') as archive:
    for path in files:archive.writestr(zipfile.ZipInfo('scar-x/'+path.relative_to(root.resolve()).as_posix()),path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
    archive.writestr(zipfile.ZipInfo('scar-x/PACKAGE_MANIFEST.json'),json.dumps({'version':'0.2-LAB','experimental':True,'language_quality_gate_passed':False,'vision_scope':'digits-only','sha256':manifest,'excludes':['local/','personal Hermes config','private data','.env','external models']},indent=2),compress_type=zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(target) as archive:assert all(hashlib.sha256(archive.read('scar-x/'+name)).hexdigest()==value for name,value in manifest.items())
digest=hashlib.sha256(target.read_bytes()).hexdigest()
(root/'artifacts/SCAR-X-V0.2-LAB.sha256').write_text(digest+'  '+target.name+'\n',encoding='ascii')
print(json.dumps({'package':str(target),'bytes':target.stat().st_size,'files':len(files),'sha256':digest},indent=2))
