"""Finite-path likelihood and sampling diagnostics, not model training."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.score_integrity import fixture,exact,variance
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter()
    run=new_run(root,'p1-score-integrity-audit-v1',{'protocol':'experiments/p1-score-integrity-audit-v1.md',
        'display_title':'P1 audit · eligibility score và phương sai · không huấn luyện',
        'model_scope':'Liệt kê path/Monte Carlo trên fixture2neuron; không phải model live hay bằng chứng năng lực',
        'new_training':False,'seeds':[211,223,227],'pairs':8192,'candidate_family_reopened':False})
    records=[]
    with ResourceMonitor(interval=.1) as monitor:
        for index,name in enumerate(('moderate','strong','delay')):
            start=time.perf_counter();f=fixture(name);oracle=exact(f)
            if not oracle['oracle_pass']:raise ValueError('Expected score gradient oracle failed')
            samples=[{'seed':s,**variance(f,s)} for s in (211,223,227)]
            record={'fixture':name,'oracle':oracle,'variance':samples};records.append(record)
            write_json(run/(name+'.json'),record);monitor.check()
            progress(run,'Đã đo '+name,index+1,3,metrics={'derivative_error':oracle['max_absolute_error'],
                'trace_clip_bias':oracle['variants']['trace_clip10']['bias_l2'],
                'antithetic_variance_ratio':[s['antithetic_to_independent_variance_ratio'] for s in samples]})
            if time.perf_counter()-start>60 or time.perf_counter()-began>300:raise TimeoutError('Registered score-audit budget')
        result={'run':run.relative_to(root).as_posix(),'decision':'SCORE_MATH_DIAGNOSTIC_ONLY','fixtures':records,
            'all_oracles_pass':all(r['oracle']['oracle_pass'] for r in records),'new_training':False,'deployed':False,
            'normal_chat_or_vision_proven':False,'retention_proven':False,'resources':monitor.report(),'cap':cap,
            'limits':['Float64 finite-path model, not floating-point Adam update guarantee.','Constructed clipping bias does not prove its prevalence in ordinary H48tasks.','Variance ratio may differ at larger scale or changing weights; no representation-quality claim.']}
        (run/'report.md').write_text('# Local score integrity audit\n\nFinite-path and sampling diagnostic, no training or ability claim.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/p1-score-integrity-latest.json',result)
        print(json.dumps({'run':str(run),'all_oracles_pass':result['all_oracles_pass'],'ratios':{r['fixture']:[s['antithetic_to_independent_variance_ratio'] for s in r['variance']] for r in records}}))


if __name__=='__main__':main()
