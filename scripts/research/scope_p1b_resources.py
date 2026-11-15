"""Research-only instrumentation. Never stops or discovers serving services."""
import ctypes as ct
from ctypes import wintypes as wt
import json
import os
import subprocess
import threading
import time
import psutil


def memory():
    class PMC(ct.Structure):
        _fields_=[("cb",wt.DWORD),("faults",wt.DWORD)]+[(k,ct.c_size_t) for k in
            ("peak_rss","rss","peak_paged","paged","peak_nonpaged","nonpaged","commit","peak_commit","private")]
    kernel=ct.WinDLL("kernel32",use_last_error=True);api=ct.WinDLL("psapi",use_last_error=True)
    kernel.GetCurrentProcess.restype=wt.HANDLE
    api.GetProcessMemoryInfo.argtypes=[wt.HANDLE,ct.POINTER(PMC),wt.DWORD]
    info=PMC();info.cb=ct.sizeof(info)
    if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ct.byref(info),info.cb):raise OSError(ct.get_last_error())
    return {k:int(getattr(info,k)) for k in ("peak_rss","rss","commit","peak_commit","private")}


def gpu():
    query="name,compute_cap,memory.total,memory.used,memory.free,temperature.gpu,clocks.current.sm,utilization.gpu,driver_version"
    p=subprocess.run(["nvidia-smi","--query-gpu="+query,"--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
    if p.returncode:return {"available":False,"stderr":p.stderr[-1000:]}
    r=[x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
    return {"available":True,"name":r[0],"compute_capability":r[1],"total_MiB":int(r[2]),
            "used_MiB":int(r[3]),"free_MiB":int(r[4]),"temperature_C":int(r[5]),
            "SM_clock_MHz":int(r[6]),"utilization_percent":int(r[7]),"driver":r[8]}


class Monitor:
    def __init__(self):
        self.peak_rss=self.peak_commit=0;self.pids=set();self.exceeded=False
        self.stop_event=threading.Event();self.thread=None

    def sample(self):
        root=psutil.Process();processes=[root]+root.children(recursive=True)
        rss=commit=0
        for p in processes:
            try:
                m=p.memory_info();rss+=m.rss;commit+=getattr(m,"private",m.rss);self.pids.add(p.pid)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        self.peak_rss=max(self.peak_rss,rss);self.peak_commit=max(self.peak_commit,commit)
        self.exceeded|=max(rss,commit)>=2_000_000_000

    def start(self):
        self.sample()
        def loop():
            while not self.stop_event.wait(.05):self.sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=1)
        self.sample()
        return {"sampled_tree_peak_rss":self.peak_rss,"sampled_tree_peak_commit":self.peak_commit,
                "observed_owned_PIDs":sorted(self.pids),"exceeded":self.exceeded,"sample_interval_seconds":.05}
