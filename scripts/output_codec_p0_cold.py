"""Cold-process continuation of an auditor-owned codec fixture checkpoint."""
from pathlib import Path
import json,os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.runtime.limits import apply_memory_limit
from scarx.evaluation.output_codec_p0 import CodecFixtureSession
from scarx.evaluation.output_codec_p0_validation import continuation_fingerprint


def main():
    if len(sys.argv)!=3:raise ValueError('Fixture checkpoint/output required')
    source=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
    if not source.is_relative_to(ROOT/'runs') or source.parent!=target.parent or target.exists():raise ValueError('Only fresh owned P0 run paths supported')
    cap=apply_memory_limit();session=CodecFixtureSession.load(source)
    result=continuation_fingerprint(session)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state_sha256':result['state_sha256'],'cap':cap,'fixture_updates':result['fixture_updates']}))


if __name__=='__main__':main()
