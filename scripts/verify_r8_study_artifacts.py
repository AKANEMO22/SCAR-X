"""Independent stored-prediction/counter/hash audit; never trains a model."""
from pathlib import Path
import argparse,hashlib,json,os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    run=Path(args.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned run required')
    output=run/'independent-artifact-check.json'
    if output.exists():raise ValueError('Do not overwrite an audit')
    result=json.loads((run/'result.json').read_text(encoding='utf-8'))
    if result['error'] is not None:raise ValueError('Complete run required')
    cls=StoppedResetCredit if 'reset-credit' in run.name else MatchedLocalCredit
    data_run=ROOT/'runs'/result['base_run'] if 'base_run' in result else run
    hashes=json.loads((data_run/'dataset-hashes.json').read_text(encoding='utf-8'))
    checks=[];prediction_rows=0;forward_replay_rows=0;cap=apply_memory_limit()
    with ResourceMonitor(interval=.05) as monitor:
        for file in sorted((run/'jobs').glob('*/*-result.json')):
            row=json.loads(file.read_text(encoding='utf-8'));budget=row['budget']
            checkpoint=file.parent/(budget+'-model.zip');raw=checkpoint.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
            dataset=data_run/'datasets'/f"{row['task']}-{row['seed']}.npz"
            if hashlib.sha256(dataset.read_bytes()).hexdigest()!=hashes[dataset.name]:raise ValueError('Dataset hash mismatch')
            with np.load(dataset,allow_pickle=False) as data:
                for split in ('id','ood'):
                    observed=row['predictions'][split];target=np.asarray(observed['target'])
                    if not np.array_equal(target,data[split+'_y']):raise ValueError('Stored targets differ from frozen labels')
                    for metric in ('direct','probe'):
                        p=np.asarray(observed[metric])
                        if p.shape!=target.shape or not np.isin(p,[0,1]).all():raise ValueError('Invalid predictions')
                        score=float(np.mean(p==target))
                        if score!=row['metrics'][split+'_'+metric]:raise ValueError('Metric recomputation mismatch')
                    prediction_rows+=len(target)
                # Replay first16direct rows for each seed140011arm/budget through
                # inherited source predict_many, independently of runner helper.
                if row['seed']==140011:
                    model=cls.loads(raw)
                    if model.dumps()!=raw:raise ValueError('Snapshot changed under current compatible loader')
                    p=model.predict_many(data['id_x'][:16])
                    if p.tolist()!=row['predictions']['id']['direct'][:16]:raise ValueError('Forward replay mismatch')
                    forward_replay_rows+=len(p)
            with np.load(file.parent/(budget+'-learning.npz'),allow_pickle=False) as log:
                history=log['history']
            if len(history)!=row['training']['episodes'] or not np.isfinite(history).all():raise ValueError('Learning history mismatch')
            if history[:,0].tolist()!=list(range(1,len(history)+1)):raise ValueError('Missing update steps')
            if row['arm'].endswith(('zero','frozen','readout_only')) and (history[:,2]!=0).any():raise ValueError('Frozen-control hidden weights moved')
            checks.append({'file':file.relative_to(run).as_posix(),'metrics_and_counters_exact':True});monitor.check()
    if len(checks)!=result['row_count']:raise ValueError('Missing result records')
    audit={'status':'PASS','conditions':len(checks),'independently_scored_prediction_rows':prediction_rows,
        'inherited_forward_replay_rows':forward_replay_rows,'quality_training_calls':0,'checks':checks,'cap':cap,'resources':monitor.report(),
        'scope':'Artifact accuracy/counters/hash verification; not independent fresh-seed confirmation'}
    output.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('checks','resources','cap')}))


if __name__=='__main__':main()
