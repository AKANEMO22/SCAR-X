"""One live writer per data directory; process exit releases the OS lock."""
import os


class DirectoryLease:
    def __init__(self,path):
        self.stream=open(path,'a+b')
        try:
            self.stream.seek(0,2)
            if self.stream.tell()==0: self.stream.write(b'0'); self.stream.flush()
            self.stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close(); self.stream=None
            raise RuntimeError('Thư mục dữ liệu đang được một phiên SCAR-X khác sử dụng. Dừng phiên cũ hoặc chọn --data-dir khác.') from exc

    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream=None
