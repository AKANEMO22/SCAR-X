"""Fixed-workload forward/eligibility/proposed-update backend profile, no fit."""
import time
START=time.perf_counter()
import argparse
import json
from pathlib import Path
import statistics
import sys
from scope_p1b_common import ALIF, cases, cuda, ROOT, np
from scope_p1b_resources import gpu, memory
sys.path.insert(0,str(ROOT/"src"))
from scarx.runtime.limits import apply_memory_limit


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--backend",choices=("cpu","gpu"),required=True)
    ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cap=apply_memory_limit(2_000_000_000);before=gpu();setup=time.perf_counter()
    handles=[]
    if args.backend=="gpu":
        if not before["available"] or before["free_MiB"]<1024:raise MemoryError("GPU reserve before context")
        xp,handles=cuda()
        props=xp.cuda.runtime.getDeviceProperties(0)
        device={"compute_capability":[props["major"],props["minor"]],
                "driver_api_version":xp.cuda.runtime.driverGetVersion(),"runtime_version":xp.cuda.runtime.runtimeGetVersion(),
                "cupy_version":xp.__version__}
    else:xp=np;device={"numpy_version":np.__version__}
    setup_seconds=time.perf_counter()-setup
    samples=cases(261161,"qual")[:2]
    reference=ALIF(261161,kind="teacher")
    hints=[]
    for case in samples:
        _,_,_,history,_=reference.forward(case["sequence"],tape=True)
        hints.append(np.asarray(history,np.float32)[:,64:120])
    model=ALIF(261161,sign=-1,xp=xp)
    initial_weights={name:model.host(getattr(model.core,name)).copy() for name in ("w_in","w_rec","w_out","bias")}
    def sync():
        if xp is not np:xp.cuda.get_current_stream().synchronize()
    p,bank,_,_,_=model.forward(samples[0]["sequence"],track=True,hint=hints[0])
    model.proposal(samples[0]["label"],p,bank,"teacher");sync()
    durations=[];outputs=[];free=[]
    for rep in range(3):
        sync();t0=time.perf_counter()
        for case,hint in zip(samples,hints):
            p,bank,q,_,_=model.forward(case["sequence"],track=True,hint=hint)
            _,_,_,delta,_=model.proposal(case["label"],p,bank,"teacher")
            outputs.append({"case_id":case["id"],"p":p.tolist(),"q":q.tolist(),"delta":model.host(delta).tolist()})
        sync();durations.append(time.perf_counter()-t0)
        if xp is not np:
            free.append(int(xp.cuda.runtime.memGetInfo()[0]))
            if free[-1]<512*1024**2:raise MemoryError("GPU physical reserve")
    if model.updates or model.hidden_updates:raise AssertionError("No learning in profiling")
    if any(not np.array_equal(value,model.host(getattr(model.core,name))) for name,value in initial_weights.items()):
        raise AssertionError("Profiling mutated weights")
    result={"backend":args.backend,"setup_seconds":setup_seconds,"repeat_seconds":durations,
            "median_seconds_per_two_episodes":statistics.median(durations),"outputs":outputs,
            "forwards_backend":model.forwards,"forwards_CPU_teacher_setup":reference.forwards,
            "accepted_learning_updates":model.updates,"device":device,"memory_cap":cap,
            "process_memory":memory(),"gpu_before":before,"gpu_after":gpu(),"GPU_free_samples_bytes":free,
            "pool_used_bytes":int(xp.get_default_memory_pool().used_bytes()) if xp is not np else 0,
            "pool_total_bytes":int(xp.get_default_memory_pool().total_bytes()) if xp is not np else 0,
            "wall_seconds":time.perf_counter()-START,"shape":{"H":64,"D":64,"fan_in":8,"ticks":32},
            "scope":"Real ALIF forward and local trace/proposed physical delta; no applied update; transfers, kernels and sync included"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="outputs"}))


if __name__=="__main__":main()
