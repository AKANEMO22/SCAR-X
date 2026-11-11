"""Arithmetic over pinned older trial rows, never a model reproduction claim."""
from pathlib import Path
import json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[2]


def main():
    queries=[
      ('r2','7d590f768b36eb9d60e5c9433f1966e4e620cff3','runs/20261111T031658Z-r2-substrate-confirmation-v1-37d670/result.json'),
      ('r3','42ed574264816e3a246a984a274f903f7eecdbfa','runs/20261111T035124Z-r3-temporal-confirmation-v1-7fe1b9/result.json'),
      ('r5','d6c609df6f90fc8df96fb3de424dd54388f8dc59','runs/20261111T064558Z-r5-information-path-v1-cddd2b/result.json')]
    output={}
    for label,sha,path in queries:
        d=json.loads(subprocess.check_output(['git','-C',str(PROJECT),'show',sha+':'+path]))
        if label=='r2':means={s:float(np.mean([r['direct'] for r in d['trials'] if r['substrate']==s])) for s in sorted({r['substrate'] for r in d['trials']})}
        elif label=='r3':means={t:{s:float(np.mean([r['head'] for r in d['trials'] if r['task']==t and r['substrate']==s])) for s in sorted({r['substrate'] for r in d['trials']})} for t in sorted({r['task'] for r in d['trials']})}
        else:means={key:float(np.mean([r['R1_pre'][key] for r in d['trials']])) for key in ['vA','vB','joint4','mixed_pair']}
        output[label]={'sha':sha,'result':path,'evidence':'AGGREGATE_ROW_ARITHMETIC_ONLY_NOT_RUNTIME_REPRODUCED','means':means}
    target=ROOT/'docs/research/forensic-r6-20261111/OLDER_ROUND_RECALCULATION.json'
    target.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'methods':list(output),'neural_training':False}))


if __name__=='__main__':main()
