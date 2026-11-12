"""Pin branch/source/run provenance without importing another agent's code."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.metadata,json,platform,subprocess,sys,zipfile,io

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[2]
OUT=ROOT/'docs/research/forensic-r6-20261111'


def git(*args):return subprocess.check_output(['git','-C',str(PROJECT),*args])
def text(*args):return git(*args).decode('utf-8').strip()
def digest(raw):return hashlib.sha256(raw).hexdigest()


def main():
    remote={r.split()[1].removeprefix('refs/heads/'):r.split()[0] for r in text('ls-remote','--heads','origin').splitlines()}
    local={r.split()[0]:r.split()[1] for r in text('for-each-ref','--format=%(refname:short) %(objectname)','refs/heads').splitlines()}
    branches={**{k:v for k,v in local.items() if k.startswith('research/')},**remote}
    branches={k:v for k,v in branches.items() if k in ('main','cursor/scarx-release-candidate') or k.startswith('research/')}
    records=[];custody=[]
    for branch,sha in sorted(branches.items()):
        tree={row.split('\t',1)[1]:row.split()[2] for row in text('ls-tree','-r',sha).splitlines()}
        reports=[p for p in tree if p.startswith('docs/research/collab/') and (Path(p).name.startswith(('report','audit','argument','lessons')) and p.endswith('.md'))]
        changed=text('diff','--name-only','main',sha).splitlines()
        relevant=[p for p in changed if p in tree and (p.endswith(('.py','.md','.json')) and (p.startswith(('scripts/','src/','tests/','experiments/','docs/research/collab/','artifacts/')) or p.endswith(('result.json','manifest.json'))))]
        files=[]
        for p in relevant:
            raw=git('show',f'{sha}:{p}');files.append({'path':p,'git_blob':tree[p],'sha256':digest(raw),'bytes':len(raw)})
        protocols=[f for f in files if f['path'].startswith('experiments/')]
        results=[f for f in files if f['path'].endswith('/result.json')]
        records.append({'branch':branch,'sha':sha,'pushed_at_inventory':remote.get(branch)==sha,'base_main':text('merge-base','main',sha),
            'reports':reports,'protocols':protocols,'result_files':results,'changed_files':files})
        if branch in ('research/deepseek/r6d-audit','research/grok/r6-causal-readout-repair','research/deepseek/r5-independent-relational-binding'):
            for f in results:
                p=f['path'];d=json.loads(git('show',f'{sha}:{p}'))
                runpath=str(Path(p).parent).replace('\\','/')
                entry={'branch':branch,'sha':sha,'result':p,'result_sha256':f['sha256'],'top_keys':list(d),
                       'saved_predictions_present':any(k in str(list(d)).casefold() for k in ('predictions','prediction_file'))}
                mp=runpath+'/manifest.json';sp=runpath+'/source.zip'
                if mp in tree:
                    manifest=json.loads(git('show',f'{sha}:{mp}'));entry['manifest']=manifest
                    if sp in tree:
                        snap=git('show',f'{sha}:{sp}');checks=[]
                        with zipfile.ZipFile(io.BytesIO(snap)) as z:
                            for name,expected in manifest.get('source_sha256',{}).items():
                                raw=z.read(name);match=digest(raw)==expected
                                head=git('show',f'{sha}:{name}') if name in tree else None
                                same=head==raw
                                newline_only=head is not None and head.replace(b'\r\n',b'\n')==raw.replace(b'\r\n',b'\n')
                                checks.append({'path':name,'manifest_hash_matches_snapshot':match,'head_bytes_equal':same,'head_text_newline_equivalent':newline_only})
                        entry['source_zip_sha256']=digest(snap);entry['snapshot_checks']=checks
                custody.append(entry)
    worktrees=[]
    for block in text('worktree','list','--porcelain').split('\n\n'):
        lines=block.splitlines()
        if not lines:continue
        p=Path(lines[0].removeprefix('worktree '))
        status=subprocess.check_output(['git','-C',str(p),'status','--short'],text=True,encoding='utf-8')
        worktrees.append({'path':str(p),'identity':lines[1:],'status':status.splitlines()})
    inventory={'created_utc':datetime.now(timezone.utc).isoformat(),'audit_branch':text('-C',str(ROOT),'branch','--show-current'),
      'audit_base_sha':text('-C',str(ROOT),'rev-parse','HEAD'),'main_sha':text('rev-parse','main'),'remote_main_sha':remote['main'],
      'ponytail_available':False,'branches':records,'worktrees':worktrees,'run_custody':custody,
      'environment':{'python':sys.version,'platform':platform.platform(),'packages':{name:importlib.metadata.version(name) for name in ('numpy','psutil','pytest','Pillow','scikit-learn','torch')}}}
    target=OUT/'BASELINE_INVENTORY.json';target.write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# P0 baseline inventory','','Read-only snapshot. Branch existence/report is not approval, reproduction or a measured skill.','',
           f"Audit base/main: `{inventory['audit_base_sha']}`. Main remote: `{remote['main']}`.",'',
           '| Branch | Pinned SHA | Pushed | Changed protocol files | Changed raw result files |','|---|---|---|---:|---:|']
    for r in records:lines.append(f"| `{r['branch']}` | `{r['sha']}` | {r['pushed_at_inventory']} | {len(r['protocols'])} | {len(r['result_files'])} |")
    lines+=['','Full source hashes, run manifests, source-zip comparison, package versions and dirty worktrees: `BASELINE_INVENTORY.json`.','',
            'Gemini R6 is a local working branch; its uncommitted candidate/results are not treated as a frozen final handoff. No source files or data in peer worktrees were edited.',
            'Source-zip/head CRLF-only differences are recorded separately from exact byte identity. Missing predictions are not synthesized. `ponytail` was absent from enabled tool inventory.']
    (OUT/'BASELINE_INVENTORY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'branches':len(records),'custody_runs':len(custody),'main':remote['main'],'output':str(target)}))


if __name__=='__main__':main()
