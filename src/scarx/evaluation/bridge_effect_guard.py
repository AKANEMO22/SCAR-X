"""Proposed P1A-v2 guard repair. Pure instrumentation, not installed in v1.

Local gethostname is not an outbound request. Unknown process/network attempts
remain denied, and a sticky violation survives exceptions swallowed by an SDK.
Pinned stdlib VER probes may be declined as unavailable without spawning a
shell. Only an exact pinned GPU telemetry subprocess can be allowed.
"""
import hashlib
import os
from pathlib import Path
import re
import traceback


class StickyEffectGuard:
    def __init__(self,platform_source=None,platform_sha=None,cmd_executable=None,cmd_sha=None,
                 monitor_source=None,monitor_sha=None,gpu_executable=None,gpu_sha=None):
        self.platform_source=Path(platform_source).resolve() if platform_source else None
        self.platform_sha=platform_sha;self.cmd=Path(cmd_executable).resolve() if cmd_executable else None
        self.cmd_sha=cmd_sha;self.violations=[];self.allowed_queries=[];self.declined_sdk_probes=[]
        self.monitor_source=Path(monitor_source).resolve() if monitor_source else None
        self.monitor_sha=monitor_sha;self.gpu=Path(gpu_executable).resolve() if gpu_executable else None
        self.gpu_sha=gpu_sha;self.gpu_queries=0;self.ver_queries=0

    def _gpu_probe_allowed(self,args,frames):
        if not self.monitor_source or not self.gpu:return False
        from .bridge_environment_metrics import GPU_QUERY_ARGS
        executable,argv=args[:2]
        if not isinstance(executable,str) or not isinstance(argv,(tuple,list)):return False
        if Path(executable).resolve()!=self.gpu or tuple(argv)!=(str(self.gpu),*GPU_QUERY_ARGS):return False
        if hashlib.sha256(self.gpu.read_bytes()).hexdigest()!=self.gpu_sha:return False
        if hashlib.sha256(self.monitor_source.read_bytes()).hexdigest()!=self.monitor_sha:return False
        return any(f['function']=='sample_physical_gpu' and Path(f['filename']).resolve()==self.monitor_source for f in frames)

    def _ver_probe_allowed(self,args,frames):
        if not self.platform_source or not self.cmd:return False
        executable,argv=args[:2]
        if not isinstance(executable,str) or not isinstance(argv,str):return False
        if Path(executable).resolve()!=self.cmd:return False
        if hashlib.sha256(self.cmd.read_bytes()).hexdigest()!=self.cmd_sha:return False
        if hashlib.sha256(self.platform_source.read_bytes()).hexdigest()!=self.platform_sha:return False
        prefixes=(str(self.cmd),'"'+str(self.cmd)+'"')
        commands=('ver','command /c ver','cmd /c ver')
        if not any(re.fullmatch(re.escape(p)+r'\s+/c\s+"'+re.escape(command)+'"',argv.strip(),re.I)
                   for p in prefixes for command in commands):return False
        return any(f['function']=='_syscmd_ver' and Path(f['filename']).resolve()==self.platform_source for f in frames)

    def hook(self,event,args):
        if event=='socket.gethostname':
            self.allowed_queries.append({'event':event,'kind':'local identity query; no returned name logged'});return
        prohibited=event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.posix_spawn')
        if not prohibited:return
        frames=[{'filename':f.f_code.co_filename,'function':f.f_code.co_name,'line':line}
                for f,line in traceback.walk_stack(None)]
        ver=False;gpu=False;predicate_error=None
        if event=='subprocess.Popen':
            try:
                ver=self.ver_queries<4 and self._ver_probe_allowed(args,frames)
                gpu=self.gpu_queries<128 and self._gpu_probe_allowed(args,frames)
            except (OSError,ValueError,TypeError) as exc:predicate_error=type(exc).__name__
        if ver:
            self.ver_queries+=1
            self.declined_sdk_probes.append({'event':event,'kind':'pinned stdlib VER probe declined, no shell spawned'})
            raise FileNotFoundError('Declared SDK metadata probe unavailable; no subprocess was spawned')
        if gpu:
            self.gpu_queries+=1
            self.allowed_queries.append({'event':event,'kind':'pinned dedicated GPU telemetry query'});return
        record={'event':event,'frames':frames[:18]}
        if predicate_error:record['predicate_error']=predicate_error
        if event=='subprocess.Popen':
            record['executable_basename']=Path(str(args[0])).name
            record['argv_sha256']=hashlib.sha256(repr(args[1]).encode()).hexdigest()
        self.violations.append(record)
        raise PermissionError('Unexpected P1A side effect denied; sticky stop is set')

    def check(self):
        if self.violations:raise RuntimeError('P1A sticky guard stop; do not charge or run another forward')
