"""Read-only follow-up on607ffca; does not rerun the author's campaigns."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.r8_matched_credit import peer_module,PEER_SHA,PEER
from scarx.evaluation.forensic.research_gates import audit_p1_summaries,retention_gate
from scarx.evaluation.runs import new_run,write_json


def raw(path):return subprocess.check_output(['git','show',PEER_SHA+':'+path],cwd=ROOT)


def main():
    run=new_run(ROOT,'r8-report-repair-audit-v2',{'display_title':'607ffca report repair audit · no new training','peer_sha':PEER_SHA},
        data_description='Pinned repaired-peer JSON and pure evaluator counterexamples; no model learning.')
    inputs={name:json.loads(raw('artifacts/gemini-r8/'+name+'.json')) for name in ('confirmation','continual-stream','multikey-associative','runtime-bench')}
    # Load only the evaluator, not its training scripts.
    import importlib.util
    spec=importlib.util.spec_from_file_location('peer607_evaluator',PEER/'src/scarx/evaluation/r8_gate_evaluator.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);G=module.R8GateEvaluator
    matrix=[[.86,0,0],[.84,.8,0],[.8,.8,.8]]
    fake={'bf_cascade':{'acquisition_mean':.82,'final_mean':.8,'mean_forgetting':.02,'worst_forgetting':.06},
          'unconsolidated':{'mean_forgetting':.03}}
    result={'peer_sha':PEER_SHA,'reported_sha_from_attachment':'607ffca76974fb7c69992ad81b94d930f7356c3a',
        'reported_sha_exists':subprocess.run(['git','cat-file','-e','607ffca76974fb7c69992ad81b94d930f7356c3a'],cwd=ROOT,capture_output=True).returncode==0,
        'empty_peer_pass':G({}).evaluate_full()['overall_pass'],'peer_accepts_mean2pp_worst6pp':G({'p2_results':fake}).evaluate_p2()['pass'],
        'independent_counterexample_retention':retention_gate([{'seed':42,'matrix':matrix}],[42]),
        'p1_summary_compliance':audit_p1_summaries(inputs['confirmation']),
        'confirmation_has_raw_seed_records':bool(inputs['confirmation'].get('per_seed_records')),
        'new_confirmation_or_quality_training':False}
    dev={}
    for arm in ('unconsolidated','bf_cascade','bf_cascade_slow'):
        rows=inputs['continual-stream'][arm]['per_seed'];records=[{'seed':s,'matrix':v['matrix']} for s,v in zip((140011,140017,140023),rows,strict=True)]
        checked=retention_gate(records,[140011,140017,140023]);checked['means']={key:float(np.mean([r[key] for r in checked['per_seed']])) for key in ('acquisition','final','mean_forgetting','worst_forgetting')};dev[arm]=checked
    result['recomputed_dev_retention']=dev
    result['decision']='REQUEST_CHANGES_KEEP_LOCAL_RESEARCH_ONLY'
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/r8-report-repair-audit-v2.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('#607ffca follow-up\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':run.name,'empty_pass':result['empty_peer_pass'],'worst_gate_missing':result['peer_accepts_mean2pp_worst6pp'],'decision':result['decision']}))


if __name__=='__main__':main()
