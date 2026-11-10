"""Sample process-tree memory, with native Windows process high-water marks.

Sampling is observational, not a hard operating-system limit. Hardware shared
memory and pagefile attribution are not inferred from process counters.
"""
import os
import threading
import time
import psutil


class ResourceMonitor:
    def __init__(self, limit=2_000_000_000, interval=.01):
        self.limit, self.interval = limit, interval
        self.peak_rss = self.peak_commit = self.peak_os_working_set = self.peak_os_pagefile = 0
        self.samples = 0
        self.exceeded = False
        self.stop_event = threading.Event()

    def sample(self):
        try:
            parent = psutil.Process(os.getpid())
            processes = [parent]+parent.children(recursive=True)
            rss = commit = 0
            for proc in processes:
                try:
                    m = proc.memory_info()
                    rss += m.rss
                    commit += getattr(m,"private",m.vms)
                    if proc.pid == os.getpid():
                        self.peak_os_working_set = max(self.peak_os_working_set,getattr(m,"peak_wset",m.rss))
                        self.peak_os_pagefile = max(self.peak_os_pagefile,getattr(m,"peak_pagefile",0))
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    pass
            self.peak_rss = max(self.peak_rss,rss)
            self.peak_commit = max(self.peak_commit,commit)
            self.samples += 1
            self.exceeded |= max(rss,commit,self.peak_os_working_set,self.peak_os_pagefile) > self.limit
        except psutil.Error:
            pass

    def __enter__(self):
        self.start = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self._run,daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def check(self):
        self.sample()
        if self.exceeded:
            raise MemoryError("Observed process-tree memory exceeded budget")

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.sample()

    def report(self):
        return {"peak_rss_bytes":self.peak_rss,"peak_commit_or_vms_bytes":self.peak_commit,
                "os_process_peak_working_set_bytes":self.peak_os_working_set,
                "os_process_peak_pagefile_commit_bytes":self.peak_os_pagefile,
                "samples":self.samples,"interval_seconds":self.interval,
                "limit_bytes":self.limit,"observed_over_limit":self.exceeded,
                "elapsed_seconds":time.perf_counter()-self.start,
                "hard_cap_enforced":False,"igpu_measured":False,
                "energy_joules":None,"note":"Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."}
