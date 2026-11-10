import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
from scarx.evaluation.binding_audit21 import run

run(Path(__file__).resolve().parents[1])
