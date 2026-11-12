"""Remove ONLY archived completed refs, after consolidation is on origin/main.

Unknown/new refs and ALL live R13-family refs are always retained. Every
remote deletion is guarded by an exact expected-SHA lease. No force-push main.
"""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.research_workspace import main_repository
PREFIX='archive/2026-11-13/branches/'
def git(repo,*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True).strip()
def protected(name,rules):return name=='main' or name in rules['local_refs'] or any(name.startswith(p) for p in rules.get('protected_ref_prefixes',['research/gemini/r13']))
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();repo=main_repository(ROOT)
    maintenance=repo/'local/maintenance/20261113-cleanup'
    before=json.loads((maintenance/'refs-before.json').read_text(encoding='utf-8'))
    rules=json.loads((maintenance/'protected.json').read_text(encoding='utf-8'))
    remote={line.split('\t')[1]:line.split('\t')[0] for line in git(repo,'ls-remote','--heads','origin').splitlines()}
    tags={line.split('\t')[1]:line.split('\t')[0] for line in git(repo,'ls-remote','--tags','origin','refs/tags/archive/2026-11-13/*').splitlines()}
    currentmain=git(repo,'rev-parse','main')
    if remote.get('refs/heads/main')!=currentmain:raise ValueError('Push and verify consolidated main before ref cleanup')
    if not(repo/'docs/research/experiment-catalog.json').exists():raise ValueError('Consolidated catalog missing from main checkout')
    deletion=[];retained=[]
    for entry in before:
        ref=entry['ref']
        if not ref.startswith('refs/remotes/origin/'):continue
        name=ref[len('refs/remotes/origin/'):]
        if protected(name,rules):continue
        target='refs/heads/'+name;actual=remote.get(target)
        if actual is None:continue
        if actual!=entry['sha']:
            retained.append({'branch':name,'reason':'Changed after inventory; no deletion','sha':actual});continue
        if tags.get('refs/tags/'+PREFIX+name+'^{}')!=actual:raise ValueError('Remote archive not exact: '+name)
        deletion.append({'branch':name,'sha':actual,'archive_tag':PREFIX+name})
    print(json.dumps({'remote_deletions_planned':len(deletion),'changed_refs_retained':retained,'protected_prefixes':rules.get('protected_ref_prefixes',[])},ensure_ascii=False),flush=True)
    if not a.execute:return
    journal=maintenance/'ref-cleanup.jsonl'
    for entry in deletion:
        name=entry['branch'];expected=entry['sha']
        subprocess.run(['git','-C',str(repo),'push','--force-with-lease=refs/heads/'+name+':'+expected,'origin',':refs/heads/'+name],check=True)
        with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'remote_deleted':name,'sha':expected,'archive_tag':entry['archive_tag']})+'\n')
    # Retain every currently checked-out branch, including source dependencies.
    attached={line[len('branch refs/heads/'):] for line in git(repo,'worktree','list','--porcelain').splitlines() if line.startswith('branch refs/heads/')}
    oldlocal={e['ref']:e['sha'] for e in before if e['ref'].startswith('refs/heads/')}
    for line in git(repo,'for-each-ref','--format=%(refname) %(objectname)','refs/heads').splitlines():
        ref,actual=line.split(' ');name=ref[len('refs/heads/'):]
        if protected(name,rules) or name in attached:continue
        # Unknown branches created by another agent during cleanup are retained.
        if ref not in oldlocal:continue
        archive=PREFIX+name
        if name=='codex/consolidate-research-20261113':
            if subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',actual,'main']).returncode:continue
            archive='main (consolidation commit is an ancestor)'
        elif actual!=oldlocal[ref] or tags.get('refs/tags/'+archive+'^{}')!=actual:continue
        subprocess.run(['git','-C',str(repo),'update-ref','-d',ref,actual],check=True)
        with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'local_deleted':name,'sha':actual,'archive_tag':archive})+'\n')
    subprocess.run(['git','-C',str(repo),'fetch','origin','--prune'],check=True)
    after=git(repo,'ls-remote','--heads','origin')
    (maintenance/'remote-heads-after.txt').write_text(after+'\n',encoding='utf-8')
    print('Archived completed refs removed; active R13 and unknown refs retained',flush=True)
if __name__=='__main__':main()
