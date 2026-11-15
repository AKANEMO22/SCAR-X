"""Proposed v2 telemetry helper; not invoked by the stopped v1 runner."""
import hashlib
from pathlib import Path
import subprocess
import time

GPU_QUERY_ARGS=('--id=0','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits')


def parse_gpu_memory(text):
    rows=text.strip().splitlines()
    if len(rows)!=1:raise ValueError('Expected one dedicated GPU reading')
    values=rows[0].split(',')
    if len(values)!=2:raise ValueError('Malformed GPU reading')
    free,total=(int(v.strip()) for v in values)
    if not 0<=free<=total or total<=0:raise ValueError('Invalid GPU memory range')
    return {'free_bytes':free*1024**2,'total_bytes':total*1024**2,
            'source':'nvidia-smi dedicated GPU memory query; point sample, not continuous peak'}


def sample_physical_gpu(executable,expected_sha256):
    path=Path(executable).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('GPU telemetry executable changed')
    result=subprocess.run([str(path),*GPU_QUERY_ARGS],capture_output=True,text=True,
        timeout=3,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return {**parse_gpu_memory(result.stdout),'epoch':time.time()}
