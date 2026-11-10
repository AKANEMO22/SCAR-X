"""Windows/local launcher. Set numeric threading before importing NumPy."""
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'src'))
from scarx.app.generation_server import main

if __name__=='__main__':
    os.chdir(root)
    main()
