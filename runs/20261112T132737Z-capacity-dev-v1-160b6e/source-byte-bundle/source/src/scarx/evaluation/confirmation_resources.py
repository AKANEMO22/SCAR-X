"""Sample owned confirmation tree plus explicitly adopted observer services."""
import os
import psutil
from ..runtime.resources import ResourceMonitor


class ConfirmationResources(ResourceMonitor):
    def __init__(self,observer_identity=None,**kwargs):
        self.observer_identity=observer_identity;self.observed_pids=set();self.measurement_errors=[]
        super().__init__(**kwargs)

    def sample(self):
        try:
            parent=psutil.Process(os.getpid());processes={p.pid:p for p in [parent]+parent.children(recursive=True)}
            if self.observer_identity:
                pid,created=self.observer_identity
                try:
                    service=psutil.Process(pid)
                    if abs(service.create_time()-created)<.001:processes[pid]=service
                except psutil.NoSuchProcess:pass
            rss=commit=0
            for process in processes.values():
                try:
                    m=process.memory_info();rss+=m.rss;commit+=getattr(m,'private',m.vms);self.observed_pids.add(process.pid)
                    if process.pid==os.getpid():
                        self.peak_os_working_set=max(self.peak_os_working_set,getattr(m,'peak_wset',m.rss))
                        self.peak_os_pagefile=max(self.peak_os_pagefile,getattr(m,'peak_pagefile',0))
                except psutil.NoSuchProcess:pass
                except psutil.Error as exc:self.measurement_errors.append(type(exc).__name__)
            self.peak_rss=max(self.peak_rss,rss);self.peak_commit=max(self.peak_commit,commit);self.samples+=1
            self.exceeded|=max(rss,commit,self.peak_os_working_set,self.peak_os_pagefile)>self.limit
        except psutil.Error as exc:self.measurement_errors.append(type(exc).__name__)

    def check(self):
        super().check()
        if self.measurement_errors:raise RuntimeError('Resource measurement unavailable for an owned process')

    def report(self):
        return {**super().report(),'observed_owned_pids':sorted(self.observed_pids),
                'adopted_readonly_observer_pid':self.observer_identity[0] if self.observer_identity else None,
                'measurement_errors':list(self.measurement_errors),
                'scope':'Supervisor + worker descendants + same-identity adopted readonly observer; process IDs deduplicated'}
