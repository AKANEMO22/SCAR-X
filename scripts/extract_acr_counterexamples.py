"""Illustrative paired errors from completed static50 cells. Zero model calls.

Examples are selected after DEV outcomes, not fresh tests or candidate selection.
"""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.r8_matched_study import event_oracle

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();run=(ROOT/args.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned run only')
    output=run/'diagnostics/static50-counterexamples.json'
    if output.exists():raise ValueError('Never overwrite evidence')
    examples=[]
    for task in ('strict_binding','xor','distractor_binding'):
        chosen=None
        for seed in (140011,140017,140023):
            ap=run/'jobs'/f'{task}-{seed}-static_0-equal_updates/result-0.json'
            bp=run/'jobs'/f'{task}-{seed}-static_50-equal_updates/result-0.json'
            a=read(ap)['predictions']['id'];b=read(bp)['predictions']['id']
            if a['target']!=b['target']:raise ValueError('Paired labels differ')
            for i,(aa,bb,y) in enumerate(zip(a['direct'],b['direct'],a['target'])):
                wanted=aa==y and bb!=y if task=='strict_binding' else aa!=y and bb==y
                if wanted:
                    source=run/'datasets'/f'{task}-{seed}.npz'
                    with np.load(source,allow_pickle=False) as z:x=z['id_x'][i].copy();target=int(z['id_y'][i])
                    labels,summary=event_oracle(task,x[None])
                    if target!=y or int(labels[0])!=y:raise ValueError('Independent event oracle mismatch')
                    chosen={'task':task,'seed':seed,'id_index':i,'source_events':summary[0],
                       'own_direct':aa,'static50_direct':bb,'target':y,'input':x.tolist(),
                       'own_result_sha256':sha(ap),'mixed_result_sha256':sha(bp),'dataset_sha256':sha(source)}
                    break
            if chosen:break
        if chosen:examples.append(chosen)
    with output.open('x',encoding='utf-8') as f:json.dump({'examples':examples,'new_model_calls':0,'new_learning_updates':0,
       'scope':'Post-outcome illustrations, deterministic first paired error within registered DEV seeds; not fresh evidence or selection'},f,ensure_ascii=False,indent=2)
    for e in examples:print(json.dumps({k:e[k] for k in ('task','seed','id_index','source_events','own_direct','static50_direct','target')},ensure_ascii=False))
if __name__=='__main__':main()
