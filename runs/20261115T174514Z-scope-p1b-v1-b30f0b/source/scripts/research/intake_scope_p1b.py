"""Bounded audit/profiling/prerequisite phase only; never calls a learner."""
import os
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[name]="1"
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
import winreg
import psutil
from scope_p1b_resources import Monitor,gpu,memory
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from scarx.runtime.limits import apply_memory_limit


def serving():
    listeners=[c.pid for c in psutil.net_connections(kind="tcp") if c.laddr and c.laddr.port==8775 and c.status=="LISTEN"]
    records=[]
    for pid in sorted(set(listeners+[61708])):
        try:
            p=psutil.Process(pid);records.append({"pid":pid,"name":p.name(),"created":p.create_time(),"running":p.is_running()})
        except psutil.NoSuchProcess:records.append({"pid":pid,"running":False})
    return {"gateway_listeners":sorted(set(listeners)),"processes":records}


def main():
    start=time.perf_counter();stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run=ROOT/"runs"/(stamp+"-scope-p1b-v1-"+uuid.uuid4().hex[:6]);run.mkdir()
    (ROOT/"local/p1b-20261116/session-path.txt").write_text(str(run),encoding="utf-8")
    cap=apply_memory_limit(2_000_000_000);monitor=Monitor();monitor.start();stages=[]
    def save(name,x):
        (run/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    prior=60.0 # Conservative reservation for earlier data-only audit and hardware intake.
    def command(name,script,args,timeout):
        remaining=1800-prior-(time.perf_counter()-start)
        if remaining<=5:raise TimeoutError("Aggregate execution budget")
        t0=time.perf_counter()
        try:
            p=subprocess.run([sys.executable,str(ROOT/"scripts/research"/script),*map(str,args)],
                             capture_output=True,text=True,encoding="utf-8",timeout=min(timeout,remaining-2))
            code=p.returncode;stdout=p.stdout;stderr=p.stderr
        except subprocess.TimeoutExpired as ex:
            code=124;stdout=ex.stdout or b"";stderr=ex.stderr or b""
            if isinstance(stdout,bytes):stdout=stdout.decode("utf-8",errors="replace")
            if isinstance(stderr,bytes):stderr=stderr.decode("utf-8",errors="replace")
        (run/(name+".stdout.txt")).write_text(stdout,encoding="utf-8")
        (run/(name+".stderr.txt")).write_text(stderr,encoding="utf-8")
        stages.append({"stage":name,"exit_code":code,"wall_seconds":time.perf_counter()-t0})
        save("execution-ledger.json",{"prior_reserved_seconds":prior,"stages":stages,
             "elapsed_intake_seconds":time.perf_counter()-start,"limit_seconds":1800})
        print(json.dumps(stages[-1]),flush=True)
        if monitor.exceeded:raise MemoryError("Owned tree exceeded2GB")
        return code
    files=["experiments/scope-p1b-temporal-preflight-v1.md",*[
        "scripts/research/"+n for n in ("audit_scope_p1_saved_v2.py","scope_p1b_common.py","scope_p1b_resources.py",
            "profile_scope_p1b_backend.py","preflight_scope_p1b.py","audit_scope_p1b_preflight.py","intake_scope_p1b.py")],
        "src/scarx/neural/core.py","src/scarx/hybrid/learner.py","src/scarx/hybrid/cuda_alif.py",
        "src/scarx/evaluation/readout_eligibility.py"]
    source_hashes={}
    for name in files:
        data=(ROOT/name).read_bytes();target=run/"source"/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        source_hashes[name]=hashlib.sha256(data).hexdigest()
    before=serving()
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu=winreg.QueryValueEx(key,"ProcessorNameString")[0]
    save("hardware.json",{"CPU":cpu,"physical_cores":psutil.cpu_count(logical=False),"logical_cores":psutil.cpu_count(),
        "RAM_bytes":psutil.virtual_memory().total,"RAM_available_bytes":psutil.virtual_memory().available,
        "OS":platform.platform(),"NVIDIA":gpu(),"CPU_temperature_measured":False,
        "cupy_installed":importlib.util.find_spec("cupy") is not None,"serving_before":before})
    save("manifest.json",{"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "dirty_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True).strip(),
        "source_hashes":source_hashes,"cap":cap,"scope":"P1b intake only, no learning","UTC_started":stamp})
    try:
        old=ROOT/"runs/20261115T165305Z-scope-teacher-preflight-v1"
        if command("p1_saved_audit","audit_scope_p1_saved_v2.py",["--original",old,"--output",run/"p1-original-audit.json"],30):
            raise RuntimeError("Original P1 audit failed")
        if command("profile_cpu","profile_scope_p1b_backend.py",["--backend","cpu","--output",run/"profile_cpu.json"],120):
            raise RuntimeError("CPU profile failed")
        code=command("profile_gpu","profile_scope_p1b_backend.py",["--backend","gpu","--output",run/"profile_gpu.json"],120)
        cpu=json.loads((run/"profile_cpu.json").read_text());backend="cpu";reason="CUDA did not qualify"
        parity=None
        if code==0:
            gpu_result=json.loads((run/"profile_gpu.json").read_text())
            err=max(float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for a,b in zip(cpu["outputs"],gpu_result["outputs"]) for k in ("p","q","delta"))
            parity={"max_abs_error":err,"pass":err<=1e-5}
            reserve=min(gpu_result["GPU_free_samples_bytes"],default=0)>=512*1024**2
            safe=reserve and gpu_result["process_memory"]["peak_rss"]<2_000_000_000
            if parity["pass"] and safe and gpu_result["median_seconds_per_two_episodes"]<cpu["median_seconds_per_two_episodes"]:
                backend="gpu";reason="Matched CUDA workload faster and qualified"
            else:reason="CPU faster, or CUDA failed parity/reserve"
        save("backend-selection.json",{"backend":backend,"reason":reason,"parity":parity,"CPU_median_seconds":cpu["median_seconds_per_two_episodes"]})
        if serving()!=before:raise RuntimeError("Serving PID/creation contract changed; stop")
        if command("new_preflight","preflight_scope_p1b.py",["--run",run/"preflight","--backend",backend],180):
            raise RuntimeError("New preflight runtime error")
        if command("preflight_audit","audit_scope_p1b_preflight.py",["--run",run/"preflight","--output",run/"preflight-audit.json"],60):
            raise RuntimeError("Preflight audit failed")
        result=json.loads((run/"preflight/result.json").read_text())
        audit=json.loads((run/"preflight-audit.json").read_text())
        qualified=all(result["gates"].values()) and audit["audit_pass"] and all(audit["independently_reconstructed_gates"].values())
        save("intake-result.json",{"verdict":"PREFLIGHT_QUALIFIED_REGISTER_P1B_NEXT" if qualified else "PREFLIGHT_NO_GO_STOP_BEFORE_TRAINING",
             "backend":backend,"gates":result["gates"],"audit":audit["audit_pass"],"training_updates":0,
             "P1b_registered":False,"serving_unchanged":serving()==before,"wall_seconds":time.perf_counter()-start})
    except Exception as ex:
        save("intake-result.json",{"verdict":"INTAKE_RUNTIME_STOP","error":repr(ex),"training_updates":0,
             "wall_seconds":time.perf_counter()-start,"serving_after":serving()})
        raise
    finally:
        save("resource.json",{"owned_tree":monitor.close(),"parent_OS_peaks":memory(),"serving_after":serving(),
             "intake_wall_seconds":time.perf_counter()-start,"prior_reserved_seconds":prior})
    print(str(run),flush=True)


if __name__=="__main__":main()
