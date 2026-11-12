"""Restore detached, exact historical source inputs without touching live R13.1."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.research_workspace import main_repository
PINS={'review-gemini-r8-607ffca':'607ffca7c9222c5e3fd9741c3ebdf5939e10bedf',
      'review-gemini-r8-fe472b3':'fe472b3b1257cdb264805bf79fd53e5647e3568d'}
def main():
    repo=main_repository(ROOT)
    for name,sha in PINS.items():
        path=repo/'local/worktrees'/name
        if not path.exists():subprocess.run(['git','-C',str(repo),'worktree','add','--detach',str(path),sha],check=True)
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip()
        if actual!=sha or dirty:raise ValueError('Existing input differs; no automatic overwrite: '+str(path))
        print(name+' '+sha)
if __name__=='__main__':main()
