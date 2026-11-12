"""Explicitly approved once-only P1 confirmation supervisor and independent audit."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import psutil
from scarx.evaluation.r8_confirmation import SEEDS,PRIMARY_TASKS,TASKS,ARMS,expected_cells,summarize_confirmation
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit,PEER,PEER_SHA
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit
from scarx.evaluation.confirmation_resources import ConfirmationResources
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.forensic.confirmation_reference import check_statistics_and_gate
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import activity
from run_r8_matched_dev import atomic_json

PROPOSAL='experiments/r8-reset-credit-confirmation-proposal-v1.md'
APPROVAL='experiments/r8-confirmation-approval-20261112.json'
REGISTRATION='experiments/r8-reset-credit-confirmation-v1.md'
EXPECTED_REPLY='Duyệt confirmation 12 seed'
APPROVAL_CALL='call_c9e5cfcca5aa4a7cac0e937a8b852134'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def observer_identity():
    owner=json.loads((ROOT/'local/r8-observer-owner.json').read_text(encoding='utf-8'))
    process=psutil.Process(owner['pid'])
    if Path(process.cwd()).resolve()!=ROOT or not any(arg.endswith('serve_matched_observer.py') for arg in process.cmdline()):
        raise ValueError('Observer PID ownership mismatch')
    return (process.pid,process.create_time())


def approved_preflight():
    approval=json.loads((ROOT/APPROVAL).read_text(encoding='utf-8'))
    if approval.get('user_reply_verbatim')!=EXPECTED_REPLY or APPROVAL_CALL not in approval.get('authorization_source',''):
        raise ValueError('Explicit user approval receipt missing')
    if approval.get('new_bank_execution_allowed_once') is not True:raise ValueError('Once-only approval required')
    registered=git('show','7f538241ae464c99d7dab2e9e977defc53b01975:'+PROPOSAL)
    if registered.replace(b'\r\n',b'\n')!=(ROOT/PROPOSAL).read_bytes().replace(b'\r\n',b'\n'):
        raise ValueError('Approved proposal was modified')
    if subprocess.check_output(['git','-C',str(PEER),'rev-parse','HEAD'],text=True).strip()!=PEER_SHA:
        raise ValueError('Pinned peer source moved')
    if subprocess.check_output(['git','-C',str(PEER),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Pinned peer source dirty')
    if git('status','--porcelain').strip():raise ValueError('Commit runner/tests/protocol before opening bank')
    if (ROOT/'artifacts/r8-reset-credit-confirmation-v1.json').exists():raise ValueError('Confirmation already has a receipt')
    return approval,observer_identity()


def independent_audit(run,deadline,monitor):
    result=json.loads((run/'worker-result.json').read_text(encoding='utf-8'))
    if result['error'] is not None or result['completed_jobs']!=204 or result['row_count']!=408:
        raise ValueError('Incomplete confirmation cannot PASS')
    hashes=json.loads((run/'dataset-hashes.json').read_text(encoding='utf-8'));rows=[];index={};predictions_count=0;replayed=0
    for file in sorted((run/'jobs').glob('*/*-result.json')):
        if time.perf_counter()>deadline:raise TimeoutError('Global budget includes final audit')
        row=json.loads(file.read_text(encoding='utf-8'));key=tuple(row[k] for k in ('task','seed','arm','budget'))
        if key in index:raise ValueError('Duplicate result cell')
        dataset=run/'datasets'/f"{row['task']}-{row['seed']}.npz"
        if sha(dataset.read_bytes())!=hashes[dataset.name]:raise ValueError('Frozen data hash mismatch')
        cls=MatchedLocalCredit if row['arm']=='own_reset_correct' else StoppedResetCredit
        blob=(file.parent/(row['budget']+'-model.zip')).read_bytes();initial=(file.parent/(row['budget']+'-initial.zip')).read_bytes()
        if sha(blob)!=row['checkpoint_sha256'] or sha(initial)!=row['initial_checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        model=cls.loads(blob)
        if model.dumps()!=blob:raise ValueError('Checkpoint roundtrip is not exact')
        if model.signal_rng.bit_generator.state!=row['signal_rng_state']:raise ValueError('RNG receipt mismatch')
        with np.load(dataset,allow_pickle=False) as data:
            for split in ('id','ood'):
                target=data[split+'_y'];stored=row['predictions'][split]
                if not np.array_equal(stored['target'],target):raise ValueError('Stored target differs from frozen labels')
                for name in ('direct','probe'):
                    pred=np.asarray(stored[name])
                    if pred.shape!=target.shape or not np.isin(pred,[0,1]).all():raise ValueError('Prediction schema')
                    if float(np.mean(pred==target))!=row['metrics'][split+'_'+name]:raise ValueError('Independent metric mismatch')
                predictions_count+=len(target)
            for name,intervention in row['interventions'].items():
                if name not in ('revert_in','revert_rec','revert_both'):raise ValueError('Unknown intervention')
                target=data['id_y']
                if not np.array_equal(intervention['target'],target):raise ValueError('Intervention target mismatch')
                for metric in ('direct','probe'):
                    pred=np.asarray(intervention[metric])
                    if pred.shape!=target.shape or not np.isin(pred,[0,1]).all():raise ValueError('Intervention predictions invalid')
                    score=float(np.mean(pred==target))
                    if score!=intervention[metric+'_accuracy'] or score!=row['metrics'][name+'_'+metric]:raise ValueError('Intervention metric mismatch')
            if row['seed']==SEEDS[0]:
                if model.predict_many(data['id_x'][:16]).tolist()!=row['predictions']['id']['direct'][:16]:raise ValueError('Independent inherited forward replay mismatch')
                replayed+=16
        with np.load(file.parent/(row['budget']+'-learning.npz'),allow_pickle=False) as log:history=log['history']
        n=row['training']['episodes']
        if history.shape!=(n,5) or not np.isfinite(history).all() or history[:,0].tolist()!=list(range(1,n+1)):raise ValueError('Update history mismatch')
        if row['budget']=='equal_updates' and n!=1024:raise ValueError('Primary budget changed')
        if row['budget']=='equal_cpu' and (n>4096 or (history[-1,4]<.5 and n!=4096)):raise ValueError('Secondary budget changed')
        if row['arm'] in ('stopped_zero','stopped_readout_only'):
            baseline=cls.loads(initial)
            if not np.array_equal(model.core.w_in,baseline.core.w_in) or not np.array_equal(model.core.w_rec,baseline.core.w_rec):raise ValueError('Frozen hidden weights changed')
            if (history[:,2]!=0).any() or model.hidden_updates!=0:raise ValueError('Control hidden update recorded')
            if row['feature_sha256']!=row['initial_feature_sha256'] or row['initial_features_match'] is not True:raise ValueError('Initial feature evidence differs')
        rows.append(row);index[key]=row;monitor.check()
    if set(index)!=set(expected_cells()):raise ValueError('Missing/unregistered confirmation cell')
    for task in PRIMARY_TASKS:
        for seed in SEEDS:
            for budget in ('equal_updates','equal_cpu'):
                readout=index[(task,seed,'stopped_readout_only',budget)]
                for arm in ARMS:
                    row=index[(task,seed,arm,budget)]
                    if row['interventions']['revert_both']['probe']!=readout['predictions']['id']['probe']:
                        raise ValueError('Full revert does not recover initial/frozen representation')
                if budget=='equal_updates':
                    states=[index[(task,seed,arm,budget)]['signal_rng_state'] for arm in ARMS]
                    if any(state!=states[0] for state in states):raise ValueError('Matched RNG draw schedule violated')
    recomputed=summarize_confirmation(rows)
    if recomputed!=result['summary']:raise ValueError('Independent statistics/gate recomputation mismatch')
    reference=check_statistics_and_gate(recomputed)
    return {'status':'PASS','conditions':len(rows),'independently_scored_prediction_rows':predictions_count,
        'inherited_forward_replay_rows':replayed,'fresh_seed_count':12,'quality_training_calls_in_audit':0,
        'independent_scalar_statistics':reference},recomputed


def stop_owned_worker(process):
    if process.poll() is not None:return
    try:children=psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:children=[]
    for child in reversed(children):
        try:child.terminate()
        except psutil.Error:pass
    process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.kill();process.wait()


def main():
    approval,observer=approved_preflight();cap=apply_memory_limit();start=time.perf_counter();deadline=start+3600
    reservation=ROOT/'local/r8-reset-credit-confirmation-v1-reservation';reservation.mkdir(exist_ok=False)
    main_before=git('rev-parse','main').decode().strip();code_commit=git('rev-parse','HEAD').decode().strip()
    run=new_run(ROOT,'r8-reset-credit-confirmation-v1',{'display_title':'Approved P1 confirmation ·12freshseeds',
        'protocol':REGISTRATION,'approved_proposal':PROPOSAL,'approval_reply':EXPECTED_REPLY,'source_commit':code_commit,
        'peer_sha':PEER_SHA,'seed_bank':SEEDS,'jobs':204,'fresh_confirmation':True,'main_or_live_change':False},
        data_description='Approved sealed12seed synthetic P1 confirmation. No teacher/private data or production training.',
        data_files=[REGISTRATION,PROPOSAL,APPROVAL,'tests/test_r8_confirmation_statistics.py','tests/test_r8_stopped_reset.py','tests/test_r8_matched_credit.py'])
    (reservation/'run.txt').write_text(str(run),encoding='utf-8')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    source_hashes={path:sha((ROOT/path).read_bytes()) for path in manifest['source_sha256'] if path.startswith(('src/','scripts/','configs/','native/','experiments/'))}
    peer_sources={}
    (run/'peer-source').mkdir()
    for filename in ('src/scarx/neural/r8_mtsg_eprop.py','src/scarx/evaluation/r8_benchmark_integrity.py'):
        raw=(PEER/filename).read_bytes();(run/'peer-source'/Path(filename).name).write_bytes(raw);peer_sources[filename]=sha(raw)
    lock={'authorization':approval,'code_commit':code_commit,'source_sha256':source_hashes,'peer_sha':PEER_SHA,'peer_source_sha256':peer_sources,
        'protocol_sha256':sha((ROOT/REGISTRATION).read_bytes()),'approved_proposal_sha256':sha((ROOT/PROPOSAL).read_bytes()),
        'global_limit_seconds':3600,'started_pid':os.getpid(),'observer_identity':list(observer)}
    write_json(run/'approval-and-lock.json',lock)
    atomic_json(run/'progress.json',{'run':run.name,'status':'APPROVED_CONFIRMATION_PREFLIGHT','completed_jobs':0,'total_jobs':204,'fresh_confirmation':True})
    atomic_json(run/'events.json',{'learning':[],'answers':[]});atomic_json(ROOT/'local/research-observer-active.json',{'run':str(run)})
    activity(run,'Khóa và chạy confirmation đã duyệt','12seed/204jobs, tối đa60phút; không đổi gate/ALIF/main/Hermes.',status='started',evidence='approval-and-lock.json')
    print(json.dumps({'run':run.name,'supervisor_pid':os.getpid(),'observer_url':'http://127.0.0.1:8767'}),flush=True)
    tests=[sys.executable,'-X','utf8','-m','pytest','-q','-p','no:cacheprovider','tests/test_r8_confirmation_statistics.py',
        'tests/test_r8_stopped_reset.py','tests/test_r8_matched_credit.py','tests/test_r8_snapshot_adapter.py','tests/test_r8_peer607_snapshot.py',
        'tests/test_confirmation_reference_and_resources.py',
        '--junitxml='+str(run/'preflight.xml')]
    process=None;error=None;audit=None;summary=None;worker_exit=None
    with ConfirmationResources(observer_identity=observer,interval=.05) as monitor:
        try:
            checked=subprocess.run(tests,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=45)
            (run/'preflight.log').write_text(checked.stdout+checked.stderr,encoding='utf-8')
            if checked.returncode:raise RuntimeError('Preflight failed before opening new bank')
            monitor.check();remaining=deadline-time.perf_counter()-30
            if remaining<=0:raise TimeoutError('No approved time remains for worker')
            command=[sys.executable,'-X','utf8',str(ROOT/'scripts/r8_confirmation_worker.py'),'--run',str(run),'--seconds',str(remaining)]
            with (run/'worker.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                write_json(run/'worker-command.json',{'command':command,'pid':process.pid,'owned':True,'wall_limit_seconds':remaining})
                print(json.dumps({'worker_pid':process.pid,'time_remaining_seconds':remaining}),flush=True)
                while process.poll() is None:
                    if time.perf_counter()>deadline-30:raise TimeoutError('Approved global wall watchdog fired')
                    monitor.check();time.sleep(.1)
            worker_exit=process.returncode
            if worker_exit!=0:raise RuntimeError('Confirmation worker stopped; preserve partial bank')
            audit,summary=independent_audit(run,deadline,monitor);monitor.check()
            if any(sha((ROOT/path).read_bytes())!=value for path,value in source_hashes.items()):raise ValueError('Frozen source drift during confirmation')
            for path,value in peer_sources.items():
                if sha((PEER/path).read_bytes())!=value:raise ValueError('Peer source drift')
            if time.perf_counter()>deadline:raise TimeoutError('Final audit exceeded approved global wall')
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        finally:
            if process is not None:stop_owned_worker(process)
    main_after=git('rev-parse','main').decode().strip()
    if main_after!=main_before:error=error or 'Main reference changed externally; review provenance before any PASS'
    decision='P1_GO_RESEARCH_ONLY' if error is None and summary and summary['master_p1_pass'] else 'P1_FAIL_RESEARCH_ONLY' if error is None else 'STOPPED_PRESERVE_PARTIAL_NO_PASS'
    result={'summary':summary,'integrity_audit':audit,'error':error,'worker_exit_code':worker_exit,'cap':cap,'resources':monitor.report(),
        'approval':approval,'code_commit':code_commit,'fresh_confirmation':True,'main_sha_before':main_before,'main_sha_after':main_after,
        'main_or_live_modified':False,'p2_tested':False,'production_ready':False,'decision':decision,'global_wall_seconds':time.perf_counter()-start,
        'preflight_command':tests}
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/r8-reset-credit-confirmation-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    progress=json.loads((run/'progress.json').read_text(encoding='utf-8'));progress.update(status='CONFIRMATION_COMPLETE' if error is None else 'CONFIRMATION_STOPPED',
        decision=decision,error=error,total_owned_resources=result['resources']);atomic_json(run/'progress.json',progress)
    (run/'report.md').write_text('# Approved independent P1 confirmation\n\nNo P2 or production approval.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    activity(run,'Hoàn tất confirmation',decision,status='completed' if error is None else 'failed',evidence='result.json')
    print(json.dumps({'run':run.name,'decision':decision,'error':error,'integrity':audit}),flush=True)
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
