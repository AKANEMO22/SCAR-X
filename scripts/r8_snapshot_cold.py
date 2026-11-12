"""Cold process for an owned R8 adapter engineering fixture."""
from pathlib import Path
import json,os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.runtime.limits import apply_memory_limit
from scarx.evaluation.forensic import r8_snapshot as S
from scarx.evaluation.forensic.r8_snapshot_fixture import fixture_model,continuation


def main():
    kind,checkpoint,target=sys.argv[1:];checkpoint=Path(checkpoint).resolve();target=Path(target).resolve()
    if not checkpoint.is_relative_to(ROOT/'runs') or checkpoint.parent!=target.parent or target.exists():raise ValueError('Owned fresh run paths required')
    cap=apply_memory_limit();model,controller=S.load(fixture_model(kind),checkpoint)
    result=continuation(model,controller);target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cap':cap,'state_sha256':result['state_sha256']}))


if __name__=='__main__':main()
