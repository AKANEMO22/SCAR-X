"""Atomic, non-pickle, bounded checkpoint format with checksums."""
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from .core import Config, SpikingCore


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload=dumps_checkpoint(model)
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def dumps_checkpoint(model):
    arrays = model.arrays()
    meta = {"format":1, "config":asdict(model.config), "updates":model.updates,
            "reward_mean":model.reward_mean,
            "sha256":{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
    stream=io.BytesIO();np.savez(stream,metadata=np.array(json.dumps(meta)),**arrays)
    return stream.getvalue()


def load_checkpoint(path):
    path = Path(path)
    if path.stat().st_size>1_000_000_000: raise ValueError('Checkpoint too large')
    return loads_checkpoint(path.read_bytes())


def loads_checkpoint(payload):
    if len(payload)>1_000_000_000:raise ValueError('Checkpoint too large')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if sum(x.file_size for x in z.infolist()) > 1_000_000_000:
            raise ValueError("Checkpoint too large for bounded loader")
        if z.getinfo("metadata.npy").file_size > 65536:
            raise ValueError("Oversized metadata")
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta.get("format") != 1:
            raise ValueError("Unsupported checkpoint format")
        model = SpikingCore(Config(**meta["config"]))
        expected = model.arrays()
        if set(data.files) != set(expected) | {"metadata"}:
            raise ValueError("Checkpoint fields mismatch")
        for name, template in expected.items():
            a = data[name]
            if a.shape != template.shape or a.dtype != template.dtype or not np.all(np.isfinite(a)):
                raise ValueError("Invalid array: "+name)
            if hashlib.sha256(a.tobytes()).hexdigest() != meta["sha256"][name]:
                raise ValueError("Checksum mismatch: "+name)
            template[:] = a
        if np.any(model.indices < 0) or np.any(model.indices >= model.config.hidden):
            raise ValueError("Invalid recurrent index")
        if np.any(model.rows < 0) or np.any(model.rows >= model.config.hidden):
            raise ValueError("Invalid plastic row")
        model.updates = int(meta["updates"])
        model.reward_mean = float(meta["reward_mean"])
        if model.updates < 0 or not np.isfinite(model.reward_mean):
            raise ValueError("Invalid learning state")
    return model

