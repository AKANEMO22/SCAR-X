"""Preserve completed branch/worktree evidence before cleanup.

The active R13.1 checkout, branch refs, caches and process state are excluded.
Default operation ONLY archives. Removal is a separate explicit phase and
requires every archive tag to be present on origin at the recorded SHA.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,tempfile,uuid,zipfile
ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.research_workspace import main_repository

DATE='2026-11-13'
def sha(b):return hashlib.sha256(b).hexdigest()
def git(repo,*args,input=None,env=None):
    return subprocess.check_output(['git','-C',str(repo),*args],input=input,env=env)
def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def tag(repo,name,commit,message):
    existing=subprocess.run(['git','-C',str(repo),'rev-parse','--verify','refs/tags/'+name+'^{}'],capture_output=True,text=True)
    if existing.returncode==0:
        if existing.stdout.strip()!=commit:raise ValueError('Never move an archive tag: '+name)
        return
    subprocess.run(['git','-C',str(repo),'tag','-a',name,commit,'-m',message],check=True)
def safe_target(repo,path,protected):
    repo=repo.resolve();workspace=repo.parent;path=Path(path).resolve()
    if path==repo or path in protected or path==ROOT.resolve():raise ValueError('Protected/primary checkout: '+str(path))
    inside=path.is_relative_to(repo/'local/worktrees')
    sibling=path.parent==workspace and path.name.startswith('scar-x-')
    if not (inside or sibling) or not(path/'.git').is_file():raise ValueError('Not an explicitly bounded registered worktree: '+str(path))
    return path
def cache(name):return any(p in ('__pycache__','.pytest_cache','.venv','build') or p.endswith('.egg-info') for p in Path(name).parts)

def archive(repo,maintenance):
    inventory=json.loads((maintenance/'worktree-inventory-before.json').read_text(encoding='utf-8'))
    protection=json.loads((maintenance/'protected.json').read_text(encoding='utf-8'))
    protected={Path(p).resolve() for p in protection['worktrees']}
    protected|={(repo/'local/worktrees'/p).resolve() for p in protection['keep_dependency_worktrees']}
    protected.add(ROOT.resolve());protected.add(repo.resolve())
    # Protect both aliases of the live campaign even if their current tips match.
    active=set(protection['local_refs'])
    archived_refs=[];worktrees=[];out=maintenance/'worktree-backups';out.mkdir(exist_ok=True)
    before=json.loads((maintenance/'refs-before.json').read_text(encoding='utf-8'))
    for entry in before:
        ref=entry['ref'];commit=entry['sha']
        if ref.startswith('refs/remotes/origin/'):
            short=ref[len('refs/remotes/origin/'):]
        elif ref.startswith('refs/heads/'):
            short=ref[len('refs/heads/'):]
        else:continue
        if short in active or any(short.startswith(prefix) for prefix in protection.get('protected_ref_prefixes',[])):continue
        name='archive/'+DATE+'/branches/'+short
        tag(repo,name,commit,'Historical branch snapshot; not production approval: '+short)
        archived_refs.append({'ref':ref,'sha':commit,'tag':name,'tag_ref':'refs/tags/'+name})
    for item in inventory:
        path=Path(item['path']).resolve()
        if path in protected:continue
        path=safe_target(repo,path,protected)
        head=git(path,'rev-parse','HEAD').decode().strip()
        if head!=item['head']:raise ValueError('Checkout moved after inventory; preserve instead: '+str(path))
        cleanenv=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
        status=git(path,'status','--porcelain=v1','-uall',env=cleanenv).decode('utf-8')
        if status.splitlines()!=item['dirty_sample'] and len(status.splitlines())<=15:
            # Full inventory retains only a sample; exact larger states are
            # captured below. Unexpected new changes are never silently erased.
            raise ValueError('Worktree changed since inventory: '+str(path))
        if len(status.splitlines())!=item['dirty_count']:raise ValueError('Dirty file count changed: '+str(path))
        slug=path.name+'-'+sha(str(path).encode())[:8]
        names=set()
        for args in [('diff','--name-only','-z'),('diff','--cached','--name-only','-z'),('ls-files','--others','--exclude-standard','-z')]:
            names.update(os.fsdecode(p) for p in git(path,*args).split(b'\0') if p)
        snapshot=head
        if names:
            index=maintenance/('index-'+uuid.uuid4().hex);env=dict(os.environ,GIT_INDEX_FILE=str(index))
            git(path,'read-tree',head,env=env)
            git(path,'add','-A','--pathspec-from-file=-','--pathspec-file-nul',input=b'\0'.join(os.fsencode(n) for n in sorted(names))+b'\0',env=env)
            tree=git(path,'write-tree',env=env).decode().strip()
            snapshot=git(path,'commit-tree',tree,'-p',head,input=('Archive completed worktree state before cleanup: '+path.name+'\n').encode(),env=env).decode().strip()
            index.unlink(missing_ok=True)
        archive_tag='archive/'+DATE+'/worktrees/'+slug
        tag(repo,archive_tag,snapshot,'Recoverable worktree snapshot, including previously uncommitted research files. Original HEAD '+head)
        ignored=[os.fsdecode(n) for n in git(path,'ls-files','--others','--ignored','--exclude-standard','-z').split(b'\0') if n]
        keep=[n for n in ignored if not cache(n)]
        # Store untracked and ignored research outputs locally, never publish
        # private local sessions/cache through a GitHub archive tag.
        recovery=out/(slug+'.zip');hashes={}
        with zipfile.ZipFile(recovery,'x',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('WORKTREE_STATUS.txt',status)
            z.writestr('TRACKED_UNSTAGED.patch',git(path,'diff','--binary'))
            z.writestr('TRACKED_STAGED.patch',git(path,'diff','--cached','--binary'))
            for name in sorted(set(keep)|names):
                p=path/name
                if not p.exists() or not p.is_file():continue
                resolved=p.resolve()
                if not resolved.is_relative_to(path):raise ValueError('Refuse outside-worktree link: '+str(p))
                raw=p.read_bytes();hashes[name]=sha(raw);z.writestr('files/'+name.replace('\\','/'),raw)
            z.writestr('FILE_SHA256.json',json.dumps(hashes,sort_keys=True))
        with zipfile.ZipFile(recovery) as z:
            for name,h in hashes.items():
                if sha(z.read('files/'+name.replace('\\','/')))!=h:raise ValueError('Archive verification failed')
        worktrees.append({'path':str(path),'head':head,'snapshot_sha':snapshot,'tag':archive_tag,'dirty_count':len(status.splitlines()),
            'dirty_paths':sorted(names),'local_recovery_zip':str(recovery),'local_recovery_sha256':sha(recovery.read_bytes()),
            'preserved_file_sha256':hashes,'ignored_cache_files_discardable':sum(cache(n) for n in ignored)})
    receipt={'date':DATE,'primary_repository':str(repo),'protected':protection,'archived_refs':archived_refs,'worktrees':worktrees,
       'scope':'Archive only; no worktree/ref removal yet. Protected R13.1 and source dependencies untouched.'}
    write_new(maintenance/'archive-receipt.json',receipt)
    print(json.dumps({'archived_ref_entries':len(archived_refs),'archived_worktrees':len(worktrees),'unique_tags':len({r['tag'] for r in archived_refs}|{r['tag'] for r in worktrees})}))

def remove(repo,maintenance):
    receipt=json.loads((maintenance/'archive-receipt.json').read_text(encoding='utf-8'))
    remote={}
    for line in git(repo,'ls-remote','--tags','origin','refs/tags/archive/'+DATE+'/*').decode().splitlines():
        commit,ref=line.split('\t');remote[ref]=commit
    for entry in receipt['archived_refs']+receipt['worktrees']:
        name='refs/tags/'+entry['tag']+'^{}';expected=entry.get('snapshot_sha',entry.get('sha'))
        if remote.get(name)!=expected:raise ValueError('Archive not verified on origin: '+entry['tag'])
    protection=receipt['protected'];protected={Path(p).resolve() for p in protection['worktrees']}
    protected|={(repo/'local/worktrees'/p).resolve() for p in protection['keep_dependency_worktrees']};protected|={repo.resolve(),ROOT.resolve()}
    journal=maintenance/'worktree-removal.jsonl'
    for entry in receipt['worktrees']:
        path=Path(entry['path'])
        if not path.exists():continue
        path=safe_target(repo,path,protected)
        if git(path,'rev-parse','HEAD').decode().strip()!=entry['head']:raise ValueError('Checkout changed; not removed: '+str(path))
        recovery=Path(entry['local_recovery_zip'])
        if sha(recovery.read_bytes())!=entry['local_recovery_sha256']:raise ValueError('Local recovery archive changed')
        # Verify every currently untracked/ignored non-cache file is already in
        # the recovery ZIP. Any new live work stops cleanup instead of losing it.
        current=set()
        for args in [('ls-files','--others','--exclude-standard','-z'),('ls-files','--others','--ignored','--exclude-standard','-z')]:
            current|={os.fsdecode(n) for n in git(path,*args).split(b'\0') if n and not cache(os.fsdecode(n))}
        for name in current:
            p=path/name
            if p.is_file() and sha(p.read_bytes())!=entry['preserved_file_sha256'].get(name):raise ValueError('New/unarchived file; checkout retained: '+str(p))
        for name in entry['dirty_paths']:
            p=path/name
            if p.is_file() and sha(p.read_bytes())!=entry['preserved_file_sha256'].get(name):raise ValueError('Dirty file changed; checkout retained')
        subprocess.run(['git','-C',str(repo),'worktree','remove','--force',str(path)],check=True)
        with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'removed':str(path),'recover_from_tag':entry['tag'],'snapshot_sha':entry['snapshot_sha']})+'\n')
        print('Removed archived worktree: '+path.name,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('archive','remove'),required=True);a=p.parse_args()
    repo=main_repository(ROOT);maintenance=repo/'local/maintenance/20261113-cleanup'
    if a.phase=='archive':archive(repo,maintenance)
    else:remove(repo,maintenance)
if __name__=='__main__':main()
