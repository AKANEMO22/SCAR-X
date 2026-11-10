"""Additive technical/attribution audit; never reselect or change the P1 gate."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,sys,json,hashlib,subprocess
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.paired_score import PairedScoreCore
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval,signflip,holm
from scarx.runtime.experiment_status import progress
from run_p1_local_classifier_v2 import dataset


def continue_hash(path):
    m=PairedScoreCore.loads(path.read_bytes());x=np.random.default_rng(817).normal(0,.1,(12,m.config.inputs)).astype(np.float32)
    for value in x[:3]:m.step(value)
    m.learn_example(x,0);return hashlib.sha256(m.dumps()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.checkpoint:
        p=args.checkpoint.resolve();out=args.out.resolve()
        if not p.is_relative_to(root/'artifacts') or not out.is_relative_to(root/'runs'):raise ValueError('Audit path scope')
        write_json(out,{'sha256':continue_hash(p)});return
    source=json.loads((root/'artifacts/p1-paired-score-v2-latest.json').read_text(encoding='utf-8'));path=root/source['run']
    selected=source['selection']['selected_condition'];lr=source['selection']['selected_lr'];pairing=selected.split('-')[1]
    phase='confirmation' if source['confirmation_trials'] else 'development'
    rows=json.loads((path/(phase+'.json')).read_text(encoding='utf-8'));seeds=sorted({r['seed'] for r in rows});tasks=('order','xor','strict_binding')
    run=new_run(root,'paired-score-technical-attribution',{'display_title':'Audit paired-score · cold-resume, attribution và head clipping',
        'source_run':source['run'],'posthoc_descriptive':True,'new_candidate_training':False,'gate_changes':False,
        'model_scope':'Đọc checkpoint/log; chỉ thử continuation trên bản sao bỏ đi, không đổi model live'},
        data_description='Same completed run; no reselection, target learning, new test split or gate change. Random continuation only on discarded technical clones.')
    summaries={};contrasts={};cold=[];diagnostics=[]
    for index,t in enumerate(tasks):
        summaries[t]={}
        for condition in sorted({r['condition'] for r in rows}):
            rs=[r for r in rows if r['task']==t and r['condition']==condition and (r['lr']==lr if condition.startswith(('raw','shuffle')) else r['lr']==.002)]
            summaries[t][condition]={'direct':float(np.mean([r['direct'] for r in rs])),'probe_gain':float(np.mean([r['probe_gain'] for r in rs]))}
        rs=[r for r in rows if r['task']==t and r['condition']==selected and r['lr']==lr]
        contrasts[t]={}
        for other in ('frozen-'+pairing,'shuffle-'+pairing,'clipped-sgd-independent','clipped-adam-independent'):
            for metric in ('probe','direct'):
                other_lr=lr if other.startswith('shuffle') else .002
                diff=[r[metric]-next(b[metric] for b in rows if b['task']==t and b['seed']==r['seed'] and b['condition']==other and b['lr']==other_lr) for r in rs]
                contrasts[t][other+'/'+metric]={'paired_values':diff,'bootstrap95':interval(diff),'signflip_p':signflip(diff),
                    'secondary_exploratory':not other.startswith('frozen')}
        for row in rs:
            checkpoint=root/row['checkpoint'];output=run/f'cold-{t}-{row["seed"]}.json'
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(checkpoint),'--out',str(output)],check=True,timeout=20,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            actual=json.loads(output.read_text(encoding='utf-8'))['sha256'];expected=continue_hash(checkpoint)
            cold.append({'task':t,'seed':row['seed'],'exact':actual==expected,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
            m=PairedScoreCore.loads(checkpoint.read_bytes());x,_=dataset(t,row['seed']+6500000,64)
            features=m.features_many(x,row['seed']+6900000);normalized=(features-m.head_mean)/m.head_scale
            centered=features-features.mean(0);cov=centered.T@centered/max(1,len(features)-1)
            diagnostics.append({'task':t,'seed':row['seed'],'normalized_feature_clipped_fraction':float(np.mean(np.abs(normalized)>5)),
                'near_constant_neuron_fraction':float(np.mean(features.std(0)<1e-4)),
                'covariance_effective_rank':float(np.trace(cov)**2/max(float(np.sum(cov*cov)),1e-20)),
                'readout_gap_to_refit':row['probe']-row['direct'],'interpretation':'posthoc heldout feature statistics; no retraining or causal retention claim'})
        progress(run,'Đã audit '+t,index+1,3,metrics={'cold_checks':len(cold),'source_gate':source['primary_representation_gate']},task=t)
    for condition in ('shuffle-'+pairing,'clipped-sgd-independent','clipped-adam-independent'):
        for metric in ('probe','direct'):
            values=[contrasts[t][condition+'/'+metric] for t in tasks]
            for value,p in zip(values,holm([v['signflip_p'] for v in values])):value['holm_3tasks_p']=p
    result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':source['decision'],'selection':source['selection'],
        'primary_representation_gate':source['primary_representation_gate'],'source_confirmation_statistics':source['confirmation_statistics'],
        'control_summary':summaries,'secondary_contrasts':contrasts,'cold_checks':cold,'all_cold_exact':all(c['exact'] for c in cold),
        'head_diagnostics':diagnostics,'deployed':False,'normal_chat_or_vision_proven':False,'retention_proven':False,
        'limits':['Gate/selection inherited unchanged; secondary contrasts and head statistics are exploratory.',
            'Existing bit-tasks with heldout noise/timing do not prove broad novel-combination or language generalization.',
            'No claim paired sampling/Adam uniquely explains gains vs compute-matched clipped baselines.']}
    (run/'report.md').write_text('# Paired-score technical/attribution audit\n\nGate inherited unchanged; no new candidate training or deployment.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    write_json(run/'result.json',result);write_json(root/'artifacts/p1-paired-score-v2-audit.json',result)
    print(json.dumps({'run':str(run),'cold_checks':len(cold),'all_cold_exact':result['all_cold_exact'],'source_gate':source['primary_representation_gate']}))


if __name__=='__main__':main()
