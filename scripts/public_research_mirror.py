"""Mirror public run files to the existing observer; never write a live SQLite.

Source code remains in its research worktree. This bridge only publishes actual
file-backed evidence; it cannot run a model, start a job, or fabricate status.
"""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,zipfile

ALLOWED=('manifest.json','report.md','result.json','source.zip','baseline-audit.json','baseline-tests.log')


def main_checkout(research):
    research=Path(research).resolve()
    if research.parent.name!='worktrees' or research.parent.parent.name!='local':raise ValueError('Expected isolated nested worktree')
    original=research.parents[2]
    if not (original/'.git').is_dir() or not (original/'run_app.py').is_file():raise ValueError('Original checkout not verified')
    return original


def _atomic(target,raw):
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=target.parent,prefix='public-mirror-',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(name,target)
    finally:
        if os.path.exists(name):os.unlink(name)


def publish(run):
    run=Path(run).resolve();research=run.parents[1];original=main_checkout(research)
    if run.parent!=research/'runs' or not run.name.replace('-','').replace('_','').isalnum():raise ValueError('Registered publicrun required')
    branch=subprocess.check_output(['git','-C',str(original),'branch','--show-current'],text=True).strip()
    if branch=='main':
        return {'mirror':None,'sqlite_written':False,'skipped':'Original checkout is on main; keep new evidence in isolated research worktree',
            'research_run':str(run),'public_manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest()}
    target=(original/'runs'/run.name).resolve()
    if target.parent!=(original/'runs').resolve():raise ValueError('Mirror target boundary')
    owner=target/'PUBLIC_WORKTREE_MIRROR.json'
    if target.exists() and not owner.is_file():raise ValueError('Refuse to overwrite an unowned run')
    expected={'source_worktree':str(research),'source_run':run.name,'scope':'Public file mirror only; no liveSQLite writes, training or deployment'}
    if owner.exists() and json.loads(owner.read_text(encoding='utf-8'))!=expected:raise ValueError('Different mirror owner')
    _atomic(owner,json.dumps(expected,indent=2).encode())
    for filename in ALLOWED:
        source=run/filename
        if not source.is_file():continue
        if source.stat().st_size>(12_000_000 if filename=='source.zip' else 2_000_000):raise ValueError('Public evidence size bound')
        if filename=='source.zip':
            with zipfile.ZipFile(source) as z:
                if any('local' in Path(n).parts or '.env'==Path(n).name for n in z.namelist()):raise ValueError('Private source path rejected')
        _atomic(target/filename,source.read_bytes())
    status=research/'local/experiment-status'/(run.name+'.json')
    if status.is_file():_atomic(original/'local/experiment-status'/status.name,status.read_bytes())
    return {'mirror':str(target),'sqlite_written':False,'public_manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest()}
