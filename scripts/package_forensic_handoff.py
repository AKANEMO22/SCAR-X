"""Package public audit evidence for user-controlled sharing; no network upload."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    out=ROOT/'artifacts/forensic-r6/SCARX_CODEX_FORENSIC_REVIEW_2026-11-11.zip'
    if out.exists():raise FileExistsError('Preserve existing handoff package')
    selected=[]
    selected+=list((ROOT/'docs/research/forensic-r6-20261111').rglob('*'))
    selected+=list((ROOT/'src/scarx/evaluation/forensic').glob('*.py'))
    selected+=list((ROOT/'scripts').glob('forensic_*.py'))
    selected+=[ROOT/'scripts/package_forensic_handoff.py',ROOT/'tests/test_forensic_evaluator.py',ROOT/'tests/test_forensic_mutations.py']
    for run in ROOT.glob('runs/*forensic*'):selected+=list(run.rglob('*'))
    selected+=list((ROOT/'artifacts/forensic-r6').glob('*-receipt.json'))
    selected+=[ROOT/'artifacts/forensic-r6/independent-tests.xml']
    files=sorted({p for p in selected if p.is_file() and '__pycache__' not in p.parts})
    if any('local' in p.relative_to(ROOT).parts or '.git' in p.relative_to(ROOT).parts for p in files):raise ValueError('Private/worktree data must not enter package')
    manifest={'source_commit':commit,'branch':'codex/scarx-forensic-r6-20261111','network_upload_performed':False,
        'main_merged':False,'production_ready':False,'files':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise ValueError('Package CRC verification failed')
        for name,digest in manifest['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Package hash mismatch: '+name)
    print(json.dumps({'package':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_commit':commit,'files':len(files)},ensure_ascii=False))


if __name__=='__main__':main()
