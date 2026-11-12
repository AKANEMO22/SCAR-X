"""Record isolated build/native/smoke evidence, with explicit temp path guards."""
from pathlib import Path
import os,sys,json,time,subprocess,importlib.util,tempfile,io,contextlib
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from public_research_mirror import publish


def main():
    start=time.perf_counter();cap=apply_memory_limit()
    run=new_run(root,'directive-release-followup-v1',{'display_title':'Baseline release · native parity / temp-session rollback / cold restart','observability_kind':'engineering','model_scope':'Existing release smoke only, guarded Temp dirs/random ports. Not P1/P2pilot or live modelteaching.','stage':0},data_description='Existing public release smoke fixtures; no liveSQLite/checkpoints.')
    (run/'report.md').write_text('# Baseline release followup\n\nActual commands running in isolated researchworktree.\n',encoding='utf-8');publish(run)
    records=[];error=None;smoke_report=None
    with ResourceMonitor(interval=.05) as monitor:
        try:
            for name,args in [('build_native',[sys.executable,'scripts/build_native.py']),('native_parity',[sys.executable,'-m','pytest','-q','-rs','tests/test_native.py'])]:
                began=time.perf_counter();result=subprocess.run(args,cwd=root,capture_output=True,text=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                (run/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8');records.append({'name':name,'command':args,'exit_code':result.returncode,'seconds':time.perf_counter()-began,'log':name+'.log'})
                if result.returncode:raise RuntimeError(name+' failed')
                monitor.check()
            spec=importlib.util.spec_from_file_location('isolated_release_smoke',root/'scripts/smoke_release.py');smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)
            temp_root=Path(tempfile.gettempdir()).resolve();real_delete=smoke.shutil.rmtree;real_launch=smoke.launch
            def check_owned(path):
                path=Path(path).resolve()
                if path.parent!=temp_root or not path.name.startswith('scarx-smoke-') or path==temp_root:raise ValueError('Smoke path outside ownedTemp root')
                return path
            def guarded_delete(path,*args,**kwargs):return real_delete(check_owned(path),*args,**kwargs)
            def guarded_launch(port,data):
                if port==8765:raise ValueError('Live port forbidden')
                return real_launch(port,check_owned(data))
            smoke.shutil.rmtree=guarded_delete;smoke.launch=guarded_launch;output=io.StringIO();errors=io.StringIO();began=time.perf_counter()
            try:
                with contextlib.redirect_stdout(output),contextlib.redirect_stderr(errors):code=smoke.main()
            finally:smoke.shutil.rmtree=real_delete;smoke.launch=real_launch
            (run/'release-smoke.log').write_text(output.getvalue()+errors.getvalue(),encoding='utf-8');smoke_report=json.loads(output.getvalue());records.append({'name':'release_smoke','command':'Existing smoke_release.main with real-model/temp path guards only','exit_code':code,'seconds':time.perf_counter()-began,'log':'release-smoke.log'})
            if code or not smoke_report.get('passed'):raise RuntimeError('Release smoke failed')
            monitor.check()
        except Exception as exc:error=type(exc).__name__+': '+str(exc)[:220]
        result={'run':run.relative_to(root).as_posix(),'decision':'ISOLATED_RELEASE_BASELINE_PASS' if error is None else 'ISOLATED_RELEASE_BASELINE_FAILED','commands':records,'smoke':smoke_report,'error':error,'no_live_sqlite_writes':True,'no_live_port_used':True,'recursive_cleanup_owned_temp_checked':True,'research_pilot_run':False,'main_training':False,'newly_acquired_skill_claim':False,'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-start}
    write_json(run/'result.json',result);write_json(root/'artifacts/directive-release-baseline.json',result)
    (run/'report.md').write_text('# Isolated release baseline\n\nNative/build and prior smoke fixtures, no newqualityclaim.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');publish(run)
    print(json.dumps({k:result[k] for k in ('run','decision','error','seconds')}))
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
