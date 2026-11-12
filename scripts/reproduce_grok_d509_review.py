"""One exact registered replay for review, not a new hypothesis/sweep/P2pilot."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.experiment_status import progress
from public_research_mirror import publish,main_checkout


def main():
    review=main_checkout(root)/'local/worktrees/review-grok-d509878';sha='d50987862ff80d2bd8556eecfb6af1b259cc591e'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=review,text=True).strip()==sha
    historical=review/'runs/grok-slow-trace-v1/result.json';recorded=review/'runs/grok-slow-trace-v1a-confirmation-71005/result.json';before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (historical,recorded)};expected=json.loads(recorded.read_text(encoding='utf-8'))
    run=new_run(root,'grok-d509-reproduction-review-v1',{'display_title':'Grok d509 · exact verification replay · no tuning/mainteaching','observability_kind':'reproducibility','reviewed_commit':sha,'model_scope':'Reproduce existing registered single-task experiment once, fixed seeds/data/budget. No new P2pilot, no privateweights/livewrites or changedthresholds.','source_snapshot_is_external_review_commit':True},data_description='Locked publicsynthetic ordergenerator/seeds fromGrokprotocol; outputdirectory only is overridden.')
    (run/'report.md').write_text('# Grok exactreplay review\n\nOne lockedverification replay, no newmodelselection.\n',encoding='utf-8');publish(run)
    code="""import sys,importlib.util\nfrom pathlib import Path\np=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('grok_locked_runner',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=Path(sys.argv[2]);m.main()\n"""
    args=[sys.executable,'-c',code,str(review/'scripts/grok_slow_trace_screen.py'),str(run/'reproduction')];env=os.environ.copy();env['PYTHONPATH']=str(review/'src');cap=apply_memory_limit();started=time.perf_counter();process=None;error=None
    with ResourceMonitor(interval=.05) as monitor,(run/'reproduction.log').open('wb') as out,(run/'reproduction-stderr.log').open('wb') as err:
        try:
            process=subprocess.Popen(args,cwd=review,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            while process.poll() is None:
                monitor.check();progress(run,'Đang tái lập đúng protocol Grok',0,1,metrics={'child_pid':process.pid,'seconds':time.perf_counter()-started,'new_sweeps':0});publish(run)
                if time.perf_counter()-started>120:raise TimeoutError('Reviewreplay120s cap')
                time.sleep(.5)
            if process.returncode:raise RuntimeError('Locked replay failed; see stderr')
            actual=json.loads((run/'reproduction/result.json').read_text(encoding='utf-8'))
            keys=('information_gate','development_labels','confirmation_split_seed','confirmation_labels','learning_screen','confirmation','data_sha256','confirmation_gain_versus_slow_frozen','confirmation_gain_versus_fast_correct','confirmation_lower_bound_passes_0_05')
            mismatch=[k for k in keys if actual[k]!=expected[k]]
            if mismatch:raise ValueError('Reproduction differs: '+','.join(mismatch))
            unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in before.items());assert unchanged
            result={'run':run.relative_to(root).as_posix(),'decision':'GROK_D509_NUMERICAL_REPRODUCTION_EXACT','reviewed_commit':sha,'all_registered_metrics_and_predictions_exact':True,'recorded_results_unchanged':unchanged,'lower95_slow_vs_frozen':actual['confirmation_gain_versus_slow_frozen']['lower'],'slow_zero_equals_frozen_all8':all(r['slow-zero']==r['slow-frozen'] for r in actual['confirmation']),'source_commit_of_replay':actual['source_commit'],'recorded_source_commit':expected['source_commit'],'output_path_override_only':True,'new_hypothesis_or_sweep':False,'p2_pilot_run':False,'main_training':False,'error':None,'seconds':time.perf_counter()-started,'resources':monitor.report(),'cap':cap,'command':args}
        except Exception as exc:error=type(exc).__name__+': '+str(exc)[:240];result={'run':run.relative_to(root).as_posix(),'decision':'GROK_REPRODUCTION_REVIEW_FAILED','error':error,'main_training':False,'p2_pilot_run':False}
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    write_json(run/'result.json',result);write_json(root/'artifacts/grok-d509-reproduction-review.json',result);(run/'report.md').write_text('# Grok exactreplay review\n\nNo newhypothesis, no sweep or changedgate.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');publish(run)
    print(json.dumps({k:result.get(k) for k in ('run','decision','lower95_slow_vs_frozen','slow_zero_equals_frozen_all8','error','seconds')}))
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
