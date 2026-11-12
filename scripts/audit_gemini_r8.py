"""Independent R8 audit, strict evaluator fixes and historical unit fixtures only."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.forensic.research_gates import retention_gate,audit_p1_summaries,binary_query_ambiguity_bound,require_injective_codebook
from scarx.evaluation.forensic.reference_evaluator import AuditViolation
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import activity
from forensic_run_r6 import run_child

SHA='fe472b3b1257cdb264805bf79fd53e5647e3568d'
SOURCE=ROOT.parent/'review-gemini-r8-fe472b3'
INPUTS=['artifacts/gemini-r8/'+n+'.json' for n in ('confirmation','continual-stream','dev-screen','multikey-associative','redteam-falsification','runtime-bench')]


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def main():
    cap=apply_memory_limit();main_before=git('rev-parse','main').decode().strip()
    if subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()!=SHA:raise ValueError('Pinned source moved')
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip():raise ValueError('Owned source is not clean')
    run=new_run(ROOT,'gemini-r8-independent-audit-v1',{'display_title':'Gemini R8 · causal/integrity audit and gate repair',
        'peer_sha':SHA,'new_quality_training':False,'new_confirmation':False,'candidate_change':False},
        data_description='Pinned existing Git artifacts; unit fixtures, finite codebook oracle and stored R-matrices only.',
        data_files=['experiments/gemini-r8-independent-audit-v1.md','tests/test_research_gate_repairs.py'])
    activity(run,'Kiểm toán R8 và sửa evaluator nghiên cứu','Đối chiếu raw, query-code ambiguity, gate và persistence; không train campaign.',status='started',evidence='manifest.json')
    result={'peer_sha':SHA,'cap':cap,'new_quality_training':False,'main_or_live_modified':False}
    data={};(run/'inputs').mkdir();hashes={}
    for path in INPUTS:
        raw=git('show',SHA+':'+path);(run/'inputs'/Path(path).name).write_bytes(raw)
        data[Path(path).stem]=json.loads(raw);hashes[path]=hashlib.sha256(raw).hexdigest()
    result['input_sha256']=hashes
    result['p1_summary_compliance']=audit_p1_summaries(data['confirmation'])
    result['p1_reported_not_reproduced']={task:{'direct':v['learned_mtsg']['direct_acc_mean'],
        'probe':v['learned_mtsg']['probe_acc_mean'],'gain_lower95':v['learned_mtsg']['probe_gain_lower95']}
        for task,v in data['confirmation']['p1_results'].items()}
    seeds=[140011,140017,140023];streams=data['continual-stream'];checks={}
    for arm in ('unconsolidated','bf_cascade','bf_cascade_slow'):
        # Historical artifact lacks seed IDs: source iteration order is the only mapping.
        records=[{'seed':seed,'matrix':row['matrix']} for seed,row in zip(seeds,streams[arm]['per_seed'],strict=True)]
        checks[arm]=retention_gate(records,seeds)
        checks[arm]['seed_mapping']='Source iteration order only; IDs absent from author per_seed rows'
        checks[arm]['means']={key:float(np.mean([r[key] for r in checks[arm]['per_seed']])) for key in ('acquisition','final','mean_forgetting','worst_forgetting')}
    result['recomputed_dev_retention']=checks
    result['bf_minus_unconsolidated_pp']={key:100*(checks['bf_cascade']['means'][key]-checks['unconsolidated']['means'][key]) for key in checks['bf_cascade']['means']}
    codes={0:(1.5,0),1:(0,1.5),2:(1,1),3:(1.5,1.5),4:(0,0),5:(1.5,0),6:(0,0),7:(1.5,0)}
    result['query_codebook']=codes;result['query_ambiguity']=binary_query_ambiguity_bound(codes)
    try:require_injective_codebook(codes);result['query_preflight_rejected']=False
    except AuditViolation:result['query_preflight_rejected']=True
    commands=[]
    commands.append(run_child(run,'peer-p0-tests',[sys.executable,'-X','utf8','-m','pytest','-q','-p','no:cacheprovider',
        'tests/test_r8_mtsg_eprop.py','tests/test_r8_bf_cascade.py','tests/test_r8_benchmark_integrity.py','--junitxml='+str(run/'peer-p0.xml')],SOURCE,60))
    commands.append(run_child(run,'engineering-counterexamples',[sys.executable,'-X','utf8',str(ROOT/'scripts/r8_engineering_worker.py'),str(SOURCE),str(run/'engineering-counterexamples.json')],SOURCE,45))
    commands.append(run_child(run,'strict-audit-tests',[sys.executable,'-X','utf8','-m','pytest','-q','-p','no:cacheprovider',
        'tests/test_research_gate_repairs.py','tests/test_cross_snn_review.py','tests/test_forensic_evaluator.py','tests/test_forensic_mutations.py',
        '--junitxml='+str(run/'strict-audit-tests.xml')],ROOT,60))
    result['commands']=commands
    result['engineering_counterexamples']=json.loads((run/'engineering-counterexamples.json').read_text(encoding='utf-8')) if (run/'engineering-counterexamples.json').exists() else None
    result['main_sha_before']=main_before;result['main_sha_after']=git('rev-parse','main').decode().strip()
    result['peer_source_tracked_unchanged']=not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--untracked-files=no'],text=True).strip()
    result['decision']='REQUEST_CHANGES_RESEARCH_ONLY_NOT_PRODUCTION_READY'
    result['engineering_checks_passed']=all(c['exit_code']==0 and not c['error'] for c in commands)
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/gemini-r8-independent-audit-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# Gemini R8 independent audit\n\nHistorical unit fixtures and raw arithmetic; no new quality training or confirmation.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    activity(run,'Hoàn tất kiểm toán R8',result['decision'],status='completed' if result['engineering_checks_passed'] else 'failed',evidence='result.json')
    print(json.dumps({'run':run.name,'engineering_passed':result['engineering_checks_passed'],'bf_delta_pp':result['bf_minus_unconsolidated_pp'],'decision':result['decision']}))
    return int(not result['engineering_checks_passed'])


if __name__=='__main__':raise SystemExit(main())
