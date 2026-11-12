"""Fetch exact upstream commits OUTSIDE the production source tree.

No dependencies are installed and no upstream module is imported. Existing
checkouts are read-only: refuse wrong/dirty paths rather than resetting them.
"""
from pathlib import Path
import json,subprocess

ROOT=Path(__file__).resolve().parents[1]
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def main():
    manifest=json.loads((ROOT/'docs/research/literature/sources/upstream-pins-20261112.json').read_text(encoding='utf-8'))
    refs=ROOT.parent.parent/'references';refs.mkdir(parents=True,exist_ok=True)
    for entry in manifest['entries']:
        if not entry.get('commit'):continue
        path=refs/entry['name']
        if not path.exists():
            subprocess.run(['git','clone','--no-checkout',entry['url'],str(path)],check=True)
            subprocess.run(['git','-C',str(path),'checkout','--detach',entry['commit']],check=True)
        if git(path,'rev-parse','HEAD')!=entry['commit'] or git(path,'status','--porcelain','--untracked-files=no'):
            raise ValueError('Existing reference differs; no automatic overwrite: '+str(path))
        import hashlib
        for filename,expected in entry.get('source_sha256',{}).items():
            if hashlib.sha256((path/filename).read_bytes()).hexdigest()!=expected:raise ValueError('Pinned body mismatch: '+filename)
        print(entry['name']+' '+entry['commit'])
if __name__=='__main__':main()
