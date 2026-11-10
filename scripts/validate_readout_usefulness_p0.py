"""Approved P0 only. Fixed technical fixtures, no P2 pilot/quality corpus."""
from pathlib import Path
import os,sys,json,time,hashlib,traceback,subprocess
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.usefulness_validation import math_check,trajectories,checkpoint_check,budget_check,signal_audit,utility_timing
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity,append


def checkpoint_stage():
    path=root/'artifacts/usefulness-p0-fixture-checkpoint.zip';result=checkpoint_check(path)
    output=root/'artifacts/usefulness-p0-cold-checkpoint.zip'
    completed=subprocess.run([sys.executable,str(root/'scripts/usefulness_p0_cold.py'),'--input',str(path),'--output',str(output)],cwd=root,capture_output=True,text=True,timeout=20,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if completed.returncode:raise RuntimeError('Separate P0 process failed: '+completed.stderr[-500:])
    child=json.loads(completed.stdout)
    expected=path.with_name(path.stem+'-expected.zip').read_bytes()
    if output.read_bytes()!=expected or child['final_sha256']!=hashlib.sha256(expected).hexdigest():raise AssertionError('Fresh process continuation differs from parent')
    result.update(separate_process_resume_exact=True,child_resources=child['resources'],child_cap=child['cap'])
    return result


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();start=time.perf_counter();neural_files=['src/scarx/neural/core.py','src/scarx/neural/unified.py'];before={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in neural_files}
    run=new_run(root,'readout-usefulness-p0-v1',{'protocol':'experiments/readout-usefulness-p0-v1.md','display_title':'P0 usefulness · math/parity/checkpoint/ΔW · KHÔNG pilot','observability_kind':'engineering','model_scope':'User approved P0 ONLY. Fresh48ALIF fixture harness, no privateweights, quality corpus, maintraining or P2pilot.','seeds':[42201,42203,42207],'beta':[0,4],'decay':.995,'segments_per_seed_beta':12,'new_pilot':False},data_description='Three literal projectP0fixtures only; no AGY/Qwen/private or quality data.',data_files=['experiments/readout-usefulness-p0-v1.md'])
    (run/'report.md').write_text('# P0 outputusefulness\n\nTechnicalchecks running, no pilot/qualityclaim.\n',encoding='utf-8')
    activity(run,'P0 đã được duyệt, bắt đầu kiểm kỹ thuật','Chỉ math/output-deletion, beta0 parity, codec/RNG/replay/rollback và appliedΔW/resource. Không pilotP2 hoặc trainingchat/vision; không đọc privateweights.',status='started',evidence={'protocol':'experiments/readout-usefulness-p0-v1.md','pilot_approved':False})
    checks={};error=None
    with ResourceMonitor(interval=.05) as monitor:
        try:
            jobs=[('math',math_check),('trajectories',trajectories),('checkpoint',checkpoint_stage),('applied_norm',budget_check),('signal_audit',signal_audit),('max_output_timing',utility_timing)]
            for i,(name,fn) in enumerate(jobs,1):
                checks[name]=fn();progress(run,'P0 đã kiểm '+name,i,len(jobs),metrics={'checks_done':i,'pilot_runs':0});monitor.check()
                if time.perf_counter()-start>180:raise TimeoutError('Registered180s P0 cap')
                activity(run,'P0: '+name,'Đã kiểm bằng fixture kỹ thuật đã đăng ký; không là accuracy/retention/normalchat.',evidence=checks[name] if isinstance(checks[name],dict) else {'seed_trajectories':len(checks[name]),'beta0_parity_all_exact':all(r['beta0_base_payload_exact_after_each_segment'] for r in checks[name])})
                if name=='trajectories':
                    for row in checks[name]:
                        append(run,'learning',{'scope':f"COPY ONLY P0 engineering · seed{row['seed']}/beta4 · NOT MAIN/PILOT",'prompt':'P0 alpha/beta/gamma fixture stream','target':'RegisteredliteralP0oracles, not languagebenchmark labels','passes':12,'sample_count':12,'losses':row['beta4_losses'],'loss_mean':sum(row['beta4_losses'])/len(row['beta4_losses']),'update_steps':row['beta4_endpoints'],'update_unit':'output-local fixtureendpoints','weight_delta':row['weight_delta'],'changed_synapses':row['changed_synapses'],'accepted':True,'rolled_back':False,'guard_checked':False,'reason':'Technicaltrajectory only; no qualityselection or product update. Same G1 hidden scalar rule, omega prototype isolated.'})
                print(json.dumps({'check':name,'completed':i,'total':6}),flush=True)
            assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in before.items())
        except Exception as exc:error=type(exc).__name__+': '+str(exc)[:300];(run/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        result={'run':run.relative_to(root).as_posix(),'decision':'P0_ENGINEERING_PASS_WITH_MATCHING_LIMITATION' if error is None and len(checks)==6 else 'P0_ENGINEERING_FAILED','checks':checks,'completed_checks':len(checks),'expected_checks':6,'error':error,'pilot_approved':False,'pilot_run':False,'quality_accuracy_retention_measured':False,'normal_chat_or_vision_proven':False,'main_training':False,'main_restored':False,'neural_source_unchanged':all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in before.items()),'cap':cap,'resources':monitor.report(),'seconds':time.perf_counter()-start,'limits':['P0 does not establish usefulimportance, representationgain or lifelonglearning.','Applied-norm matching must reject unrepresentable float32 states; not universally feasible.','Oracle registry validates P0 labels only, not arbitrary userfacts or verified flags.','Pilot and deployment require separate userapproval.']}
    write_json(run/'result.json',result);write_json(root/'artifacts/readout-usefulness-p0.json',result)
    (run/'report.md').write_text('# P0 outputusefulness engineering\n\nNo pilot/qualityclaim.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    activity(run,'P0 hoàn tất' if error is None else 'P0 phát hiện lỗi cần xử lý',result['decision']+'; pilot chưa được duyệt, chất lượng/retention chưa đo. Float32 matched bank có giới hạn phải báo rõ.',status='completed' if error is None else 'failed',evidence={'receipt':'artifacts/readout-usefulness-p0.json','completed_checks':len(checks),'error':error,'main_training':False,'pilot_run':False})
    print(json.dumps({k:result[k] for k in ('run','decision','completed_checks','seconds','error')}))
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
