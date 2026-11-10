"""Per-process Windows commit limit; no global OS/pagefile changes."""
import ctypes as c
import os


def apply_memory_limit(limit=2_000_000_000):
    if os.name!='nt': return {'enforced':False,'reason':'Giới hạn cứng Windows Job chỉ áp dụng trên Windows.'}
    from ctypes import wintypes as w
    class Basic(c.Structure):
        _fields_=[('process_time',c.c_longlong),('job_time',c.c_longlong),('flags',w.DWORD),
                  ('min_ws',c.c_size_t),('max_ws',c.c_size_t),('active',w.DWORD),
                  ('affinity',c.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
    class IO(c.Structure):
        _fields_=[(name,c.c_ulonglong) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
    class Extended(c.Structure):
        _fields_=[('basic',Basic),('io',IO),('process_memory',c.c_size_t),('job_memory',c.c_size_t),
                  ('peak_process',c.c_size_t),('peak_job',c.c_size_t)]
    kernel=c.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR]; kernel.CreateJobObjectW.restype=w.HANDLE
    kernel.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD]; kernel.SetInformationJobObject.restype=w.BOOL
    kernel.GetCurrentProcess.restype=w.HANDLE
    kernel.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]; kernel.AssignProcessToJobObject.restype=w.BOOL
    kernel.CloseHandle.argtypes=[w.HANDLE]
    handle=kernel.CreateJobObjectW(None,None)
    info=Extended(); info.basic.flags=0x100; info.process_memory=limit
    if not handle or not kernel.SetInformationJobObject(handle,9,c.byref(info),c.sizeof(info)) or not kernel.AssignProcessToJobObject(handle,kernel.GetCurrentProcess()):
        error=c.get_last_error()
        if handle: kernel.CloseHandle(handle)
        return {'enforced':False,'reason':f'Windows Job không khả dụng (mã {error}); vẫn kiểm tra RAM và giới hạn batch.'}
    # Keep handle alive for the process. No kill-on-close flag is set.
    global _JOB_HANDLE
    _JOB_HANDLE=handle
    return {'enforced':True,'limit_bytes':limit,'kind':'Windows per-process committed memory; shared pages monitored separately'}
