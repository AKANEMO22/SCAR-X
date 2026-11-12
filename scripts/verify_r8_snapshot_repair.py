"""R8 isolated state-engineering repair verification, no quality campaign."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.forensic import r8_snapshot as S
from scarx.evaluation.forensic.r8_snapshot_fixture import fixture_model,continuation,PEER
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import activity
from forensic_run_r6 import run_child


def main():
    cap=apply_memory_limit();main_before=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()
    run=new_run(ROOT,'r8-snapshot-repair-p0-v1',{'display_title':'R8 · isolated model snapshot/RNG/rollback repair',
        'source_sha':S.SOURCE,'candidate_equations_changed':False,'quality_training':False,'app_migration':False},
        data_description='Small synthetic state fixtures on pinned peer models, no benchmark/teacher/final data.',
        data_files=['experiments/r8-snapshot-repair-p0-v1.md','tests/test_r8_snapshot_adapter.py','tests/test_research_gate_repairs.py'])
    activity(run,'Sửa snapshot nghiên cứu R8','Wrapper data-only giữ đủ RNG/threshold/state; source learner và main không đổi.',status='started',evidence='manifest.json')
    commands=[];cold=[]
    for kind in ('mtsg','bf'):
        model=fixture_model(kind);state={'phase':'pending-fixture','position':1,'replay':[{'id':'P0-only'}],
            'rng':np.random.default_rng(919).bit_generator.state}
        path=run/(kind+'-state.zip');S.save(model,path,state);raw=path.read_bytes()
        expected_model,expected_state=S.loads(fixture_model(kind),raw);expected=continuation(expected_model,expected_state)
        output=run/(kind+'-cold.json')
        command=run_child(run,kind+'-cold',[sys.executable,'-X','utf8',str(ROOT/'scripts/r8_snapshot_cold.py'),kind,str(path),str(output)],ROOT,30)
        commands.append(command);actual=json.loads(output.read_text(encoding='utf-8')) if output.exists() else None
        cold.append({'kind':kind,'next_forward_and_update_exact':actual==expected,'expected':expected,'actual':actual,
                     'saved_checkpoint_unchanged':path.read_bytes()==raw,'snapshot_sha256':hashlib.sha256(raw).hexdigest()})
    if not (ROOT/'build/scarx_cpu.dll').exists():
        commands.append(run_child(run,'native-build',[sys.executable,'scripts/build_native.py'],ROOT,90))
    files=['tests/test_r8_snapshot_adapter.py','tests/test_research_gate_repairs.py','tests/test_cross_snn_review.py',
           'tests/test_forensic_evaluator.py','tests/test_forensic_mutations.py','tests/test_neural.py','tests/test_native.py','tests/test_generation1.py']
    commands.append(run_child(run,'repair-regression',[sys.executable,'-X','utf8','-m','pytest','-q','-rs','-p','no:cacheprovider',*files,'--junitxml='+str(run/'repair-regression.xml')],ROOT,90))
    result={'source_sha':S.SOURCE,'cap':cap,'cold_continuation':cold,'commands':commands,
        'actual_cold_verification_fixture_update_calls':4,'pytest_fixture_updates_counted_separately':True,
        'new_quality_training':False,'candidate_equations_changed':False,'main_sha_before':main_before,
        'main_sha_after':subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip(),
        'peer_tracked_unchanged':not subprocess.check_output(['git','-C',str(PEER),'status','--porcelain','--untracked-files=no'],text=True).strip(),
        'limits':['H48 D8 O2 K8 R12 only; explicit matching peer template required.',
                  'Model plus explicit caller JSON controller only, no live SQLite/Generation/Hermes migration.',
                  'Next-forward and next-episode update continuation, not a new mid-episode training controller.']}
    passed=all(r['next_forward_and_update_exact'] and r['saved_checkpoint_unchanged'] for r in cold) and all(c['exit_code']==0 and not c['error'] for c in commands)
    passed=passed and result['main_sha_before']==result['main_sha_after'] and result['peer_tracked_unchanged']
    result['decision']='P0_STATE_REPAIR_PASS_RESEARCH_ONLY' if passed else 'P0_STATE_REPAIR_FAILED'
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/r8-snapshot-repair-p0-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# Isolated R8 snapshot state repair\n\nNo new quality experiment.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    activity(run,'Hoàn tất sửa snapshot R8',result['decision'],status='completed' if passed else 'failed',evidence='result.json')
    print(json.dumps({'run':run.name,'decision':result['decision']}));return int(not passed)


if __name__=='__main__':raise SystemExit(main())
