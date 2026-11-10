from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import uuid
import zipfile
import os


def new_run(root, protocol, config, *, data_description=None, data_files=()):
    root = Path(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+protocol+"-"+uuid.uuid4().hex[:6]
    run = root/"runs"/run_id
    run.mkdir(parents=True)
    source_hashes = {str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                     for base in ("src","native","configs","experiments","scripts")
                     for p in sorted((root/base).rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    for item in data_files:
        p=(root/item).resolve()
        source_hashes[p.relative_to(root.resolve()).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"run_id":run_id,"protocol":protocol,"created_utc":datetime.now(timezone.utc).isoformat(),
                "config":config,"python":sys.version,"platform":platform.platform(),
                "dependencies":{n:importlib.metadata.version(n) for n in ("numpy","psutil")},
                "source_sha256":source_hashes,
                "command":sys.argv,
                "thread_environment":{k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
                "data":data_description or "Repository-authored synthetic generators; exact version and splits in source/protocol snapshot; no downloaded/private data"}
    write_json(run/"manifest.json",manifest)
    with zipfile.ZipFile(run/"source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source_hashes:
            archive.write(root/name,name)
    with (root/"runs"/"INDEX.md").open("a",encoding="utf-8") as f:
        f.write(f"- [{run_id}]({run_id}/report.md) — {protocol}\n")
    from ..runtime.experiment_status import start_run
    start_run(run,config)
    return run


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    path=Path(path)
    if path.name=='result.json' and path.parent.parent.name=='runs':
        from ..runtime.experiment_status import complete_run
        complete_run(path.parent,value)
    elif path.parent.parent.name=='runs':
        from ..runtime.experiment_status import progress
        if isinstance(value,dict):
            source=value.get('summary',value)
            keys=('task','condition','seed','epoch','pass','accuracy','direct','probe_gain','dev_exact','dev_digits','updates','train_seconds')
            metrics={k:source[k] for k in keys if isinstance(source,dict) and k in source and isinstance(source[k],(str,int,float,bool))}
            if metrics:progress(path.parent,'Đã ghi '+path.name,metrics=metrics,task=metrics.get('task'))
        elif isinstance(value,list) and path.name in ('progress.json','selection-progress.json'):
            progress(path.parent,'Đã ghi '+str(len(value))+' mục: '+path.name,len(value),metrics={'records_written':len(value)})


def cluster_interval(values, seed=919):
    """Bootstrap seed-level paired effects; 3 seeds provide weak precision."""
    import numpy as np
    a = np.asarray(values,dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(a,size=(5000,len(a)),replace=True).mean(axis=1)
    return {"mean":float(a.mean()),"lower95":float(np.quantile(draws,.025)),
            "upper95":float(np.quantile(draws,.975)),"independent_seeds":len(a),
            "method":"paired seed-cluster bootstrap; exploratory with only three clusters"}

