"""Offline portability check of the committed dataset/code handoff.

Uses an auditor-owned detached worktree with no private CLI evidence. Never
generates data, migrates a codec, starts serving or teaches a neural model.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,zipfile
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from public_research_mirror import publish

COLD=ROOT.parent/'review-basicfactory-037d77e'
PIN='037d77e24e2a3a545b72bc0d1dbcbfbd4b3369c7'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def git(*args):return subprocess.check_output(['git','-C',str(COLD),*args],text=True).strip()


def main():
    if git('rev-parse','HEAD')!=PIN:raise ValueError('Cold worktree source changed')
    if git('status','--porcelain','--untracked-files=no'):raise ValueError('Cold checkout is not initially clean')
    if (COLD/'local/basic-knowledge-factory').exists():raise ValueError('Private CLI logs must not be present')
    receipt=read(COLD/'artifacts/basic-knowledge-verified-v1.json')
    frozen={name:hashlib.sha256((COLD/name).read_bytes()).hexdigest() for name in (receipt['data'],receipt['source_ledger'])}
    if frozen[receipt['data']]!=receipt['sha256'] or frozen[receipt['source_ledger']]!=receipt['source_ledger_sha256']:
        raise ValueError('Frozen bytes changed on checkout')
    main_before=git('rev-parse','main');cap=apply_memory_limit()
    run=new_run(ROOT,'basic-knowledge-cold-checkout-v1',{'display_title':'Committed factory cold-check · no private logs / no network / no learning',
        'observability_kind':'source_portability','source_pin':PIN,'no_new_model_or_dataset_calls':True},
        data_description='Existing public frozen candidates and committed source; no private CLI stream/history.')
    env=dict(os.environ,PYTHONPATH=str(COLD/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    checks=[]
    commands=[('public-provenance',[sys.executable,'scripts/verify_basic_knowledge_provenance.py','--public-only']),
        ('targeted-tests',[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_knowledge_factory.py','tests/test_basic_knowledge_eval.py',
            'tests/test_adaptive_curriculum_data.py','tests/test_public_research_mirror.py','--junitxml='+str(run/'cold-tests.xml')]),
        ('offline-wheel',[sys.executable,'-m','pip','wheel','--no-index','--no-deps','--no-build-isolation','--wheel-dir',str(run/'wheels'),'.'])]
    error=None
    for label,command in commands:
        started=time.perf_counter()
        with ResourceMonitor(interval=.02) as monitor:
            output=subprocess.run(command,cwd=COLD,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
            monitor.check()
        (run/(label+'.log')).write_text(output.stdout+output.stderr,encoding='utf-8')
        check={'label':label,'command':command,'cwd':str(COLD),'exit_code':output.returncode,'seconds':time.perf_counter()-started,'resources':monitor.report()}
        checks.append(check);write_json(run/(label+'-command.json'),check)
        if output.returncode:error=label+' failed';break
    provenance=read(COLD/'artifacts/basic-knowledge-public-provenance-review-v1.json')
    if provenance['issues'] or provenance['decision']!='PUBLIC_SOURCE_LINEAGE_VERIFIED_ORIGIN_NOT_REATTESTED_CODEC_BLOCKED':
        error=error or 'Public provenance scope/count mismatch'
    wheel_checks=[]
    for wheel in (run/'wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as z:
            names=z.namelist();required={'scarx/evaluation/knowledge_factory.py','scarx/evaluation/basic_knowledge_eval.py'}
            if not required<=set(names):error=error or 'Factory evaluation modules missing from wheel'
            if any('local' in Path(n).parts or Path(n).name=='.env' for n in names):error=error or 'Private path in wheel'
            wheel_checks.append({'file':wheel.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),
                'files':len(names),'required_modules_present':required<=set(names),'private_content_paths_present':False,
                'scope':'Python package only. Dataset/CLI source scripts are delivered through the source checkout, not falsely claimed to be bundled training data.'})
    if not wheel_checks:error=error or 'No wheel produced'
    unchanged=all(hashlib.sha256((COLD/name).read_bytes()).hexdigest()==digest for name,digest in frozen.items())
    protected=['src/scarx/neural','src/scarx/app','src/scarx/learning','artifacts/generation1-release.json']
    protected_diff=git('diff',PIN,'--',*protected)
    if not unchanged or protected_diff or git('rev-parse','main')!=main_before:error=error or 'Source/data/main changed unexpectedly'
    result={'decision':'PUBLIC_FACTORY_COLD_CHECK_PASS' if not error else 'COLD_CHECK_FAILED','source_pin':PIN,
        'private_cli_evidence_present':False,'cold_pool_hashes':frozen,'frozen_data_preserved':unchanged,
        'public_records_verified':provenance['records'],'private_origin_attestation_claimed_in_cold_check':False,
        'checks':checks,'wheel_checks':wheel_checks,'cap':cap,'main_before_after':main_before,
        'protected_source_changed':bool(protected_diff),'new_generation_calls':0,'student_training':False,
        'codec_implemented':False,'live_state_accessed':False,'production_ready':False,'error':error}
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/basic-knowledge-cold-checkout-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# Cold public factory/source handoff\n\nNo private logs, external calls, fitting, migration or live access.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    publish(run)
    print(json.dumps({'run':run.name,'decision':result['decision'],'verified_rows':provenance['records'],'commands':[{'label':c['label'],'exit':c['exit_code']} for c in checks],'error':error}))
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
