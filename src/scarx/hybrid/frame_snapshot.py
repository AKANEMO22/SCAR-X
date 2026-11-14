"""Bounded content-addressed JPEG snapshots, atomically published per image."""

import hashlib
import os
from pathlib import Path
import tempfile


def publish_snapshot(directory, payload):
    root=Path(directory).resolve()
    if not root.is_dir() or not isinstance(payload,bytes) or not 0<len(payload)<=2_000_000:
        raise ValueError('Expected bounded JPEG bytes in an owned stream directory')
    target=root/('question-frame-'+hashlib.sha256(payload).hexdigest()+'.jpg')
    if target.is_symlink():raise ValueError('Snapshot target must not be a symbolic link')
    if target.exists():
        if target.read_bytes()!=payload:raise ValueError('Immutable snapshot integrity mismatch')
        return target
    fd,name=tempfile.mkstemp(dir=root,prefix='.snapshot-',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as out:out.write(payload);out.flush();os.fsync(out.fileno())
        # Windows rename does not overwrite an existing target; unlike replace
        # it cannot race a reader that has the same immutable image open.
        try:
            if os.name=='nt':os.rename(name,target)
            else:os.link(name,target)
        except (FileExistsError,PermissionError):
            if not target.is_file() or target.is_symlink():raise
            if target.read_bytes()!=payload:raise ValueError('Immutable snapshot integrity mismatch')
    finally:
        if os.path.exists(name):os.unlink(name)
    return target
