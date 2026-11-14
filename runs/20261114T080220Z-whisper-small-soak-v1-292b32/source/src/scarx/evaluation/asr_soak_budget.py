"""Small aggregate experiment budget; close the owned Windows handle before unlink."""

import json
import os
from pathlib import Path
import tempfile


def charge_attempt(path):
    path=Path(path)
    lock=path.with_suffix('.lock')
    handle=lock.open('x')
    try:
        with handle:
            handle.write(json.dumps({'owner_pid':os.getpid()}));handle.flush()
            data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'attempted':0,'cap':256}
            attempted=data.get('attempted')
            if data.get('cap')!=256 or type(attempted) is not int or not 0<=attempted<=256:
                raise ValueError('Corrupt aggregate ASR budget; refuse resetting it')
            if attempted>=256:raise RuntimeError('Registered aggregate256 ASRattempt budget exhausted')
            data['attempted']+=1
            fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.asr-soak-budget-',suffix='.tmp')
            with os.fdopen(fd,'w',encoding='utf-8') as f:
                json.dump(data,f,separators=(',',':'));f.flush();os.fsync(f.fileno())
            os.replace(temporary,path)
            return data['attempted']
    finally:
        # The with block closes before this finally, including return/errors.
        lock.unlink()
