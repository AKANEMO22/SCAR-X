"""Posthoc attribution/cold-state audit. Never reselect a candidate or change gates."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.local_classifier_conditioned import ConditionedLocalClassifierCore
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval,signflip,holm
from scarx.runtime.experiment_status import progress


def continuation(path):
    m=ConditionedLocalClassifierCore.loads(path.read_bytes())
    for value in np.random.default_rng(773).normal(0,.2,(7,m.config.inputs)).astype(np.float32):
        m.step(value,True);m.learn_auxiliary(0,True)
    return hashlib.sha256(m.dumps()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.checkpoint:
        path=args.checkpoint.resolve();out=args.out.resolve()
        if not path.is_relative_to(root/'artifacts') or not out.is_relative_to(root/'runs'):raise ValueError('Audit path scope')
        write_json(out,{'continued_sha256':continuation(path)});return
    source=json.loads((root/'artifacts/p1-local-classifier-v2-latest.json').read_text(encoding='utf-8'))
    source_run=root/source['run'];dev=json.loads((source_run/'development.json').read_text(encoding='utf-8'));controls=json.loads((source_run/'controls.json').read_text(encoding='utf-8'))
    run=new_run(root,'local-classifier-v2-attribution',{'display_title':'Audit P1 · tín hiệu dạy, thời điểm, cold-resume',
        'source_run':source['run'],'new_candidate_training':False,'posthoc_descriptive_analysis':True,
        'model_scope':'Phân tích log/checkpoint; không sửa model đang dùng','gate_changes':False},
        data_description='Read existing v2 logs; random continuation only on discarded audit clones, not quality training')
    tasks=('order','xor','strict_binding');seeds=(12101,12109,12113)
    def record(task,seed,condition,lr=.002,rows=dev):
        return next(r for r in rows if r['task']==task and r['seed']==seed and r['condition']==condition and r['lr']==lr)
    def contrast(task,a,b,key='probe',rows=dev):
        values=[record(task,s,a,rows=rows)[key]-record(task,s,b,rows=rows)[key] for s in seeds]
        return {'paired_values':values,'bootstrap95':interval(values),'exact_signflip_p':signflip(values)}
    effects={};matched=[];cold=[]
    for index,t in enumerate(tasks):
        selected=source['selection']['selected_condition'];lr=source['selection']['selected_lr'];timing=selected.split('-')[1]
        effects[t]={'normalization_probe':contrast(t,'norm-last4','raw-last4'),
            'normalization_direct':contrast(t,'norm-last4','raw-last4','direct'),
            'final_vs_last_timing_probe':contrast(t,'norm-final4','norm-last4'),
            'correct_vs_shuffled_probe':contrast(t,selected,'shuffle-'+timing),
            'correct_vs_shuffled_direct':contrast(t,selected,'shuffle-'+timing,'direct'),
            'bptt_vs_readout_probe':contrast(t,'bptt-full','bptt-readout',rows=controls),
            'bptt_vs_readout_direct':contrast(t,'bptt-full','bptt-readout','direct',controls),
            'selected_gain':interval([record(t,s,selected,lr)['probe_gain'] for s in seeds]),
            'selected_direct':interval([record(t,s,selected,lr)['direct'] for s in seeds]),
            'mean_seconds':{c:float(np.mean([record(t,s,c,rows=rows)['train_seconds'] for s in seeds]))
                for c,rows in ((selected,dev),('raw-last4',dev),('bptt-readout',controls),('bptt-full',controls))}}
        for seed in seeds:
            a=json.loads((source_run/f'development-{t}-{seed}-readout-0.002.json').read_text(encoding='utf-8'))
            b=json.loads((source_run/f'control-{t}-{seed}-bptt-readout-0.002.json').read_text(encoding='utf-8'))
            matched.append({'task':t,'seed':seed,'identical_targets':a['targets']==b['targets'],'identical_predictions':a['predictions']==b['predictions']})
            row=record(t,seed,selected,lr);path=root/row['checkpoint'];output=run/f'cold-{t}-{seed}.json'
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(path),'--out',str(output)],
                check=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            actual=json.loads(output.read_text(encoding='utf-8'))['continued_sha256'];expected=continuation(path)
            cold.append({'task':t,'seed':seed,'exact_rng_trace_continuation':actual==expected,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        progress(run,'Đã audit '+t,index+1,3,metrics={'cold_checks':len(cold),'matched_readout_checks':len(matched)},task=t)
    for key in ('normalization_probe','final_vs_last_timing_probe','correct_vs_shuffled_probe','bptt_vs_readout_probe'):
        adjusted=holm([effects[t][key]['exact_signflip_p'] for t in tasks])
        for t,p in zip(tasks,adjusted):effects[t][key]['holm_3tasks_p']=p
    result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':source['decision'],
        'selection':source['selection'],'paired_effects':effects,'control_summary':source['control_summary'],
        'matched_readout':matched,'cold_checks':cold,'all_matched':all(x['identical_targets'] and x['identical_predictions'] for x in matched),
        'all_cold_exact':all(x['exact_rng_trace_continuation'] for x in cold),
        'posthoc':True,'confirmation_opened':False,'gate_changed':False,'deployed':False,
        'normal_chat_or_vision_proven':False,'retention_proven':False,
        'limits':['Three development seeds; bootstrap exploratory; these data already selected the candidate, not an independent confirmation.',
            'Same forward/head/examples; hidden optimizers/update density and computation differ, so no equal-time efficiency superiority claim.',
            'Local auxiliary loss transports G-transpose teacher error, not task W_out or BPTT; not a faithful published DECOLLE reproduction.']}
    write_json(run/'result.json',result);write_json(root/'artifacts/p1-local-classifier-v2-audit.json',result)
    (run/'report.md').write_text('# P1 v2 attribution and cold-process audit\n\nPosthoc, no new candidate selection or training. Same stopped decision.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':str(run),'all_matched':result['all_matched'],'all_cold_exact':result['all_cold_exact']}))


if __name__=='__main__':main()
