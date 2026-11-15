"""Persistent precharge ledger for the approved single P1A run; no refunds."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

FORMAT='scarx-neural-outcome-p1a-attempt-v1'
LIMITS={'forward':24,'dispatch':10}


class AttemptBudget:
    def __init__(self,path,started_epoch):
        self.path=Path(path);self.started_epoch=started_epoch
        if type(started_epoch) not in (int,float) or not math.isfinite(started_epoch):raise ValueError('Invalid start')
        self.lock=self.path.with_suffix(self.path.suffix+'.owner.lock');self.nonce=uuid.uuid4().hex
        with self.lock.open('x',encoding='utf-8') as stream:
            json.dump({'pid':os.getpid(),'nonce':self.nonce,'started_epoch':started_epoch},stream)

    def close(self):
        # Close the file handle before unlinking on Windows; never delete a
        # foreign/stale lease or reset a consumed attempt ledger.
        if self.lock.exists() and json.loads(self.lock.read_text())['nonce']==self.nonce:self.lock.unlink()

    def records(self):
        if not self.path.exists():return []
        raw=self.path.read_bytes()
        if not raw.endswith(b'\n') or len(raw)>100_000:raise ValueError('Broken ledger; never reset')
        rows=[json.loads(line) for line in raw.splitlines()];previous='0'*64;counts={k:0 for k in LIMITS}
        for i,row in enumerate(rows,1):
            value={k:v for k,v in row.items() if k!='sha256'}
            if (set(value)!={'format','attempt','kind','seed','reason','epoch','previous'}
                    or value['format']!=FORMAT or value['attempt']!=i or value['previous']!=previous
                    or value['kind'] not in LIMITS or value['seed'] not in (260113,260137)
                    or not math.isfinite(value['epoch']) or not self.started_epoch<=value['epoch']<=self.started_epoch+600
                    or hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()!=row['sha256']):
                raise ValueError('Ledger integrity failure; no repair/reset')
            counts[value['kind']]+=1
            if counts[value['kind']]>LIMITS[value['kind']]:raise ValueError('Ledger exceeds cap')
            previous=row['sha256']
        return rows

    def charge(self,kind,seed,reason):
        rows=self.records()
        if kind not in LIMITS or seed not in (260113,260137):raise ValueError('Unapproved attempt type/seed')
        if time.time()-self.started_epoch>600:raise RuntimeError('P1A expired; do not restart')
        if sum(r['kind']==kind for r in rows)>=LIMITS[kind]:raise RuntimeError('Attempt budget exhausted')
        value={'format':FORMAT,'attempt':len(rows)+1,'kind':kind,'seed':seed,'reason':reason,
               'epoch':time.time(),'previous':rows[-1]['sha256'] if rows else '0'*64}
        value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        with self.path.open('ab') as stream:
            stream.write((json.dumps(value,sort_keys=True)+'\n').encode());stream.flush();os.fsync(stream.fileno())
        return value
