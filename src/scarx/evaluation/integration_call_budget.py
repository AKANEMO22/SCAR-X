"""Aggregate ASR/VLM acceptance attempt budget; no refund or reset on failure."""

import json
import os
from pathlib import Path
import tempfile

LIMITS={'ASR':4,'VLM':4}


def charge(path,kind,amount=1):
    if kind not in LIMITS or type(amount) is not int or amount<=0:
        raise ValueError('Unknown integration call type or invalid reservation')
    path=Path(path);lock=path.with_suffix('.lock');handle=lock.open('x')
    try:
        with handle:
            state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
                'limits':LIMITS.copy(),'attempted':dict.fromkeys(LIMITS,0)}
            if state.get('limits')!=LIMITS or set(state.get('attempted',{}))!=set(LIMITS):
                raise ValueError('Integration call ledger incompatible; refuse resetting it')
            for name,value in state['attempted'].items():
                if type(value) is not int or not 0<=value<=LIMITS[name]:
                    raise ValueError('Integration call ledger corrupt; refuse resetting it')
            if state['attempted'][kind]+amount>LIMITS[kind]:
                raise RuntimeError('Registered aggregate integration call budget exhausted')
            state['attempted'][kind]+=amount
            fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.integration-budget-',suffix='.tmp')
            with os.fdopen(fd,'w',encoding='utf-8') as f:
                json.dump(state,f,separators=(',',':'));f.flush();os.fsync(f.fileno())
            os.replace(temporary,path)
            return state['attempted'][kind]
    finally:lock.unlink()
