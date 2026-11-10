"""Separate fresh process for the registered P0 token-boundary continuation."""
import os,sys,json,hashlib,argparse
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.readout_usefulness import UsefulnessPrototype
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    source=Path(args.input).resolve();output=Path(args.output).resolve()
    assert source.parent==output.parent==(root/'artifacts').resolve() and source.name.startswith('usefulness-p0-') and output.name.startswith('usefulness-p0-') and source!=output
    cap=apply_memory_limit()
    with ResourceMonitor(interval=.01) as monitor:
        model=UsefulnessPrototype.load(source);source_sha=hashlib.sha256(model.dumps()).hexdigest()
        for _ in range(3):model.draw_fixture();model.permuted_g()
        while model.controller['pending']:model.advance()
        for identifier in ('p0-gamma','p0-alpha'):model.teach_fixture(identifier)
        model.save(output);monitor.check()
        print(json.dumps({'scope':'P0 fixture continuation only, not pilot','source_sha256':source_sha,'final_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'endpoints':model.core.updates,'cap':cap,'resources':monitor.report()}))


if __name__=='__main__':main()
