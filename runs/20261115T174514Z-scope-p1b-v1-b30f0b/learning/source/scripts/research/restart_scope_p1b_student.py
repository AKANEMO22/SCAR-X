"""Fresh S-only inference process. Receives X only; teacher artifact opens denied."""
import argparse
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"
import json
from pathlib import Path
import sys
import time
import numpy as np
from scope_p1b_common import ALIF,sha


def main():
    start=time.perf_counter();ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--inputs",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--study-root",type=Path,required=True)
    args=ap.parse_args();allowed={args.checkpoint.resolve(),args.inputs.resolve(),args.output.resolve()};blocked=[];opened=[]
    if args.output.exists():raise FileExistsError(args.output)
    def hook(event,values):
        if event!="open" or not isinstance(values[0],(str,bytes)):return
        p=Path(values[0]).resolve()
        if p.is_relative_to(args.study_root.resolve()):
            if p not in allowed:
                blocked.append(str(p));raise PermissionError("Teacher or unregistered study artifact access denied")
            opened.append(str(p))
    sys.addaudithook(hook)
    blob=args.checkpoint.read_bytes();model=ALIF.loads(blob)
    if model.core.config.hidden!=64 or model.kind!="student":raise ValueError("S64 student checkpoint only")
    with np.load(args.inputs,allow_pickle=False) as z:
        if set(z.files)!={"X","ids"}:raise ValueError("X-only contract, no labels/hints")
        X=z["X"];ids=z["ids"]
    before=model.forwards
    prob=[]
    for x in X:
        old=bool(np.any(x[:,2]!=0))
        prob.append(model.forward(x,old=old)[0])
    np.savez(args.output,p=np.asarray(prob),ids=ids,
             metadata=np.frombuffer(json.dumps({"checkpoint_sha256":sha(blob),"neural_forwards":model.forwards-before,
                 "opened_study_files":opened,"blocked_attempts":blocked,"labels_or_hints_loaded":False,
                 "wall_seconds":time.perf_counter()-start}).encode(),np.uint8))


if __name__=="__main__":main()
