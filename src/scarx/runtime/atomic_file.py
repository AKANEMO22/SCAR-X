"""Bounded retry for transient Windows readers denying atomic replacement."""
import os,time
from pathlib import Path


def replace_with_retry(source,target,attempts=20):
    if type(attempts) is not int or not 1<=attempts<=40:raise ValueError('Bounded replace attempts required')
    for index in range(attempts):
        try:
            os.replace(source,target);return
        except PermissionError:
            if index+1==attempts:raise
            time.sleep(min(.005*(index+1),.05))


def read_text_with_retry(path,*,encoding='utf-8',attempts=20):
    """Bounded PermissionError retry when Windows temporarily denies reading.

    Never mask missing files, malformed JSON, or a permanently unreadable file.
    Like replacement, maximum default backoff is under one second.
    """
    if type(attempts) is not int or not 1<=attempts<=40:raise ValueError('Bounded read attempts required')
    for index in range(attempts):
        try:return Path(path).read_text(encoding=encoding)
        except PermissionError:
            if index+1==attempts:raise
            time.sleep(min(.005*(index+1),.05))
