"""R9 source/raw review only, isolated from the running Codex confirmation."""
from pathlib import Path
import argparse,ast,hashlib,json,os,subprocess,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.forensic.r9_review import manifest_summary,screening_checks
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from forensic_run_r6 import run_child

SHA='2c41fdaf6fc491845879d638ac80449caf597471';OLD='39fd104';PEER=ROOT.parent/'review-gemini-r9-2c41fda'


def blob(path,sha=SHA):return subprocess.check_output(['git','show',sha+':'+path],cwd=ROOT)
def read(name,sha=SHA):return json.loads(blob('artifacts/gemini-r9/'+name+'.json',sha))


def assigned_literal(source,name):
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing literal seed declaration: '+name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-engineering-run');args=parser.parse_args()
    cap=apply_memory_limit();run=new_run(ROOT,'gemini-r9-independent-review-v1',{'display_title':'R9 source/raw audit · no new sweep or confirmation',
        'peer_sha':SHA,'current_codex_confirmation_untouched':True},data_files=['experiments/r9-independent-review-v1.md','tests/test_r9_review.py'])
    manifest=[json.loads(line) for line in blob('artifacts/gemini-r9/search_manifest.jsonl').splitlines() if line.strip()]
    screened=[json.loads(line) for line in blob('artifacts/gemini-r9/dev_screen_results.jsonl').splitlines() if line.strip()]
    script=blob('scripts/r9_massive_search_runner.py').decode('utf-8');tree=ast.parse(script)
    old_script=blob('scripts/r9_massive_search_runner.py',OLD).decode('utf-8')
    old_bank=assigned_literal(old_script,'CONFIRMATION_SEEDS');new_bank=assigned_literal(script,'CONFIRMATION_SEEDS')
    codex_manifest=ROOT.parent/'scarx-r8-matched-controls-20261112/runs/20261112T060357Z-r8-reset-credit-confirmation-v1-84547e/manifest.json'
    codex_bank=json.loads(codex_manifest.read_text(encoding='utf-8'))['config']['seed_bank']
    bh_calls=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='benjamini_hochberg' for n in ast.walk(tree))
    old_confirmation=read('locked_confirmation',OLD);confirmation=read('locked_confirmation');summary=read('search_summary')
    result={'peer_sha':SHA,'cap':cap,'manifest':manifest_summary(manifest),'screening':screening_checks(screened),
        'bh_fdr_function_actual_calls_in_runner':bh_calls,
        'screen_cache_byte_identical_after_factory_and_prediction_fix':blob('artifacts/gemini-r9/dev_screen_results.jsonl')==blob('artifacts/gemini-r9/dev_screen_results.jsonl',OLD),
        'old_confirmation_candidates':list(old_confirmation),'new_confirmation_candidates':list(confirmation),
        'old_locked_bank':old_bank,'new_locked_bank':new_bank,
        'same_locked_bank_in_old_and_new_runner':old_bank==new_bank,
        'overlap_with_running_codex_12seed_bank':sorted(set(new_bank)&set(codex_bank)),
        'author_final_summary':summary,'confirmation_summaries':{},'new_quality_or_confirmation_runs':0}
    for cid,data in confirmation.items():
        tasks={t:{'direct':r['learned_candidate']['direct_acc_mean'],'gain':r['learned_candidate']['probe_gain_mean'],
            'gain_lower95':r['learned_candidate']['probe_gain_lower95'],'gain_arithmetic_matches':abs((r['learned_candidate']['probe_acc_mean']-r['frozen_hidden']['probe_acc_mean'])-r['learned_candidate']['probe_gain_mean'])<1e-12}
            for t,r in data['tasks'].items()}
        cont=data['continual_stream'];result['confirmation_summaries'][cid]={'tasks':tasks,'continual':cont,'gate_verdict':data['gate_verdict'],
            'worst_equals_per_seed_max':cont['worst_forgetting']==max(cont['per_seed_worst_forgettings'])}
    commands=[]
    if args.reuse_engineering_run:
        previous=Path(args.reuse_engineering_run).resolve()
        if previous.parent!=ROOT/'runs':raise ValueError('Owned engineering evidence required')
        receipt=json.loads((previous/'result.json').read_text());saved=json.loads((previous/'manifest.json').read_text())
        if receipt['peer_sha']!=SHA or not receipt['engineering_checks_passed']:raise ValueError('Wrong/failed peer evidence')
        for path in ('src/scarx/evaluation/forensic/r9_review.py','scripts/r9_engineering_review_worker.py'):
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=saved['source_sha256'][path]:raise ValueError('Engineering source changed')
        commands=receipt['commands'];result['counterexamples']=receipt['counterexamples'];result['engineering_evidence_run']=previous.name
        result['engineering_reexecuted']=False
    else:
        commands.append(run_child(run,'peer-unit-tests',[sys.executable,'-X','utf8','-m','pytest','-q','-p','no:cacheprovider',
            'tests/test_r9_unified_mechanisms.py','tests/test_r9_suite.py','--junitxml='+str(run/'peer-unit-tests.xml')],PEER,60))
        commands.append(run_child(run,'counterexamples',[sys.executable,'-X','utf8',str(ROOT/'scripts/r9_engineering_review_worker.py'),str(PEER),str(run/'counterexamples.json')],PEER,30))
        commands.append(run_child(run,'audit-unit-tests',[sys.executable,'-X','utf8','-m','pytest','-q','-p','no:cacheprovider','tests/test_r9_review.py','--junitxml='+str(run/'audit-unit-tests.xml')],ROOT,30))
        result['counterexamples']=json.loads((run/'counterexamples.json').read_text()) if (run/'counterexamples.json').exists() else None
        result['engineering_reexecuted']=True
    result['commands']=commands
    result['engineering_checks_passed']=all(c['exit_code']==0 and not c['error'] for c in commands)
    result['decision']='REQUEST_CHANGES_NO_MAIN_PROMOTION'
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/gemini-r9-independent-review-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# R9 independent review\n\nNo search or quality training rerun.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':run.name,'manifest':result['manifest']['declared_rows'],'effective_models':result['manifest']['effective_model_signatures'],
        'bh_calls':bh_calls,'engineering_pass':result['engineering_checks_passed'],'decision':result['decision']}))
    return int(not result['engineering_checks_passed'])


if __name__=='__main__':raise SystemExit(main())
