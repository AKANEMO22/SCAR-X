import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
from scarx.evaluation.controls import run_phase21_control

print(run_phase21_control(Path(__file__).resolve().parents[1]))
