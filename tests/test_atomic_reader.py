import ctypes,os,threading
import pytest
from scarx.runtime.atomic_file import read_text_with_retry


@pytest.mark.skipif(os.name!='nt',reason='Real Windows sharing denial')
def test_reader_recovers_after_real_deny_share_handle_and_bounds_persistent_failure(tmp_path):
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE;kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    p=tmp_path/'heartbeat.json';p.write_text('{"measured":true}',encoding='utf-8')
    def lock():
        handle=kernel.CreateFileW(str(p),0x80000000,0,None,3,0x80,None)
        assert handle!=ctypes.c_void_p(-1).value;return handle
    handle=lock();timer=threading.Timer(.05,lambda:kernel.CloseHandle(handle));timer.start()
    try:assert read_text_with_retry(p)=='{"measured":true}'
    finally:timer.join()
    handle=lock()
    try:
        with pytest.raises(PermissionError):read_text_with_retry(p,attempts=2)
    finally:kernel.CloseHandle(handle)
    with pytest.raises(FileNotFoundError):read_text_with_retry(tmp_path/'missing.json')
