"""Fixed-capacity raw-sequence replay; task labels are explicit context cues."""
import json
import io
import zipfile
from pathlib import Path
import numpy as np


class ReplayBuffer:
    def __init__(self,capacity,steps,inputs,policy,seed):
        if policy not in ("none","reservoir","balanced"): raise ValueError("Replay policy")
        if not all(isinstance(v,int) and v>0 for v in (capacity,steps,inputs)) or capacity*steps*inputs*4>16_000_000:
            raise ValueError("Invalid or oversized replay allocation")
        self.policy=policy; self.capacity=capacity; self.size=0; self.seen=0
        self.x=np.zeros((capacity,steps,inputs),np.float32)
        self.y=np.zeros(capacity,np.int32); self.context=np.zeros(capacity,np.int32)
        self.rng=np.random.default_rng(seed)

    def add(self,x,y,context):
        self.seen+=1
        if self.policy=="none": return
        if self.size<self.capacity:
            index=self.size; self.size+=1
        elif self.policy=="reservoir":
            index=int(self.rng.integers(self.seen))
            if index>=self.capacity: return
        else:
            groups=self.context[:self.size]*100+self.y[:self.size]
            group=context*100+y
            values,counts=np.unique(groups,return_counts=True)
            count=dict(zip(values.tolist(),counts.tolist()))
            maximum=int(counts.max())
            if count.get(group,0)<maximum:
                candidates=np.where(np.isin(groups,values[counts==maximum]))[0]
            else:
                candidates=np.where(groups==group)[0]
                if not len(candidates): return
            index=int(self.rng.choice(candidates))
        self.x[index]=x; self.y[index]=y; self.context[index]=context

    def sample(self,n):
        if not self.size: return []
        indices=self.rng.choice(self.size,size=n,replace=self.size<n)
        return [(self.x[i].copy(),int(self.y[i])) for i in indices]

    def save(self,path):
        Path(path).write_bytes(self.dumps())

    def dumps(self):
        metadata={"policy":self.policy,"capacity":self.capacity,"size":self.size,"seen":self.seen,"rng":self.rng.bit_generator.state}
        stream=io.BytesIO()
        np.savez(stream,x=self.x,y=self.y,context=self.context,metadata=np.array(json.dumps(metadata)))
        return stream.getvalue()

    @classmethod
    def load(cls,path):
        path=Path(path)
        if path.stat().st_size>20_000_000: raise ValueError("Oversized replay")
        return cls.loads(path.read_bytes())

    @classmethod
    def loads(cls,payload):
        if len(payload)>20_000_000: raise ValueError("Oversized replay")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(i.file_size for i in archive.infolist())>20_000_000 or archive.getinfo('metadata.npy').file_size>65536:
                raise ValueError("Oversized replay")
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            meta=json.loads(str(z["metadata"]))
            if set(z.files)!={'x','y','context','metadata'} or z['x'].ndim!=3: raise ValueError('Invalid replay fields')
            m=cls(meta["capacity"],z["x"].shape[1],z["x"].shape[2],meta["policy"],0)
            for name in ('x','y','context'):
                a=z[name]; template=getattr(m,name)
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all(): raise ValueError('Invalid replay array')
                template[:]=a
            if not isinstance(meta['size'],int) or not 0<=meta['size']<=m.capacity or not isinstance(meta['seen'],int) or meta['seen']<meta['size']:
                raise ValueError('Invalid replay counters')
            m.size=meta["size"]; m.seen=meta["seen"]; m.rng.bit_generator.state=meta["rng"]
        return m
