"""Межпроцессная блокировка обучения; ОС освобождает её при завершении процесса."""
import errno
import os
from pathlib import Path


class TrainingLease:
    def __init__(self,path):
        self.path=Path(path)
        self.fd=None

    def acquire(self):
        if self.fd is not None:
            return False
        self.path.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(self.path,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
        try:
            if os.name=='nt':
                import msvcrt
                if os.fstat(fd).st_size==0:os.write(fd,b'0')
                os.lseek(fd,0,os.SEEK_SET)
                msvcrt.locking(fd,msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in {errno.EAGAIN,errno.EACCES}:
                return False
            raise
        self.fd=fd
        return True

    def release(self):
        if self.fd is None:
            return
        fd,self.fd=self.fd,None
        try:
            if os.name=='nt':
                import msvcrt
                os.lseek(fd,0,os.SEEK_SET);msvcrt.locking(fd,msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(fd,fcntl.LOCK_UN)
        finally:
            os.close(fd)
