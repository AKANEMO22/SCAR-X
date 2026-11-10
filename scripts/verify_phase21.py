"""Replay the recorded update schedule; do not rerank or tune methods."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import hashlib
import json
import zipfile
from scarx.evaluation.phase21 import retention_trial,analyze
from scarx.evaluation.runs import new_run,write_json

root=Path(__file__).resolve().parents[1]
base=root/'runs/20261108T162931Z-phase21-v1-f3fa83'
data=json.loads((base/'result.json').read_text()); settings=data['settings']
out=new_run(root,'phase21-reproduction',{'source_run':base.name,'purpose':'integrity and fixed update-schedule replay'})
manifest=json.loads((base/'manifest.json').read_text())
with zipfile.ZipFile(base/'source.zip') as z:
    assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest['source_sha256'].items())
matches=[]
for budget in ('updates','time'):
    row=next(r for r in data['confirmation_retention'] if r['seed']==101 and r['condition']=='selected' and r['budget']==budget)
    replayed,_,_=retention_trial(row['seed'],row['policy'],row['consolidation'],settings,rule=row['rule'],forced_updates=row['updates_per_stage'])
    assert replayed['matrix']==row['matrix']
    matches.append({'budget':budget,'seed':101,'same_accuracy_matrix':True,'updates':replayed['updates']})
analysis=analyze(data['confirmation_representation'],data['confirmation_retention'],settings)
assert analysis['representation']==data['analysis']['representation']
assert analysis['retention']==data['analysis']['retention']
result={'source_hashes_ok':True,'statistics_recomputed':True,'replications':matches}
write_json(out/'result.json',result)
(out/'report.md').write_text('# Phase2.1 reproducibility\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
print(out)
