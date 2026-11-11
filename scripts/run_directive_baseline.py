"""Directive V2 stage0: authoritative source/test audit in isolated worktree."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,urllib.request,contextlib
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import scarx,psutil,pytest
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from public_research_mirror import main_checkout,publish


def live_status():
    try:
        value=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/status',timeout=5))
        return {k:value[k] for k in ('cores_loaded','ticks','updates','ram_bytes') if k in value}
    except OSError:return {'available':False}


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    original=main_checkout(root);assert Path(scarx.__file__).resolve().is_relative_to(root/'src')
    directive=Path('C:/Users/hachimi/.codex/attachments/c6e3d678-b1eb-4ef3-84db-1e6a3383b001/pasted-text-1.txt')
    source_paths=[p for folder in ('neural','app','learning') for p in (original/'src/scarx'/folder).rglob('*.py')]
    protected={str(p.relative_to(original)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    git=lambda *args:subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    failure={}
    for name in ('p1-paired-score-v2-latest.json','p2-paired-score-diagnostic-latest.json','direct-eligibility-latest.json','direct-word-bridge-latest.json','direct-visual-bridge-latest.json','readout-usefulness-p0.json'):
        d=json.loads((root/'artifacts'/name).read_text(encoding='utf-8'));failure[name]={k:d[k] for k in ('run','decision','pilot_run','main_training') if k in d}
    audit={'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'worktrees':git('worktree','list','--porcelain'),'working_tree_before':git('status','--short'),'directive_sha256':hashlib.sha256(directive.read_bytes()).hexdigest(),'historical_main_reference':'495dab2f4bac0180e4d1cd28674391f055ccf09d','neural_matches_reference':subprocess.run(['git','diff','--quiet','495dab2','HEAD','--','src/scarx/neural'],cwd=root).returncode==0,'imported_source':str(Path(scarx.__file__).resolve()),'live_before':live_status(),'failure_matrix':failure,'ponytail_available':False,'no_live_sqlite_writes':True,'new_learning_authorized':False,'p2_pilot_authorized':False}
    cap=apply_memory_limit();run=new_run(root,'directive-v2-baseline-v1',{'display_title':'DirectiveV2 · baseline/source/authorization · no live changes','observability_kind':'engineering','model_scope':'Isolated worktree tests, no neural researchpilot or productionintegration. Publicfile mirror only, no liveSQLite writes.','git_head':audit['head'],'directive_sha256':audit['directive_sha256'],'stage':0},data_description='Existing public testfixtures/receipts only; no live checkpoints or privatetraining.')
    write_json(run/'baseline-audit.json',audit);(run/'report.md').write_text('# DirectiveV2 baseline\n\nAuthoritative source/test audit in progress. No liveSQL writes or neuralpilot.\n',encoding='utf-8');publish(run)
    class Results:
        def __init__(self):self.counts={};self.skips=[];self.total=0;self.completed=0
        def pytest_collection_finish(self,session):self.total=len(session.items)
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.failed or report.skipped:
                self.counts[report.outcome]=self.counts.get(report.outcome,0)+1
            if report.skipped:self.skips.append({'test':report.nodeid,'reason':str(report.longrepr)[-400:]})
            if report.when=='teardown':
                self.completed+=1
                if self.completed%40==0:progress(run,'Baseline tests thực tế',self.completed,self.total,metrics={'passed':self.counts.get('passed',0),'failed':self.counts.get('failed',0)});publish(run)
    plugin=Results();start=time.perf_counter();output=sys.stdout
    class Tee:
        def __init__(self,file):self.file=file
        def write(self,text):self.file.write(text);output.write(text)
        def flush(self):self.file.flush();output.flush()
        def isatty(self):return False
    with ResourceMonitor(interval=.1) as monitor,(run/'baseline-tests.log').open('w',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(log)):code=pytest.main(['-q','-rs','--disable-warnings'],plugins=[plugin])
        audit['live_after']=live_status();audit['protected_source_unchanged']=all(hashlib.sha256((original/p).read_bytes()).hexdigest()==h for p,h in protected.items())
        result={'run':run.relative_to(root).as_posix(),'decision':'DIRECTIVE_V2_BASELINE_PASS' if code==0 and audit['protected_source_unchanged'] else 'DIRECTIVE_V2_BASELINE_FAILED','stage':0,'git_head':audit['head'],'branch':audit['branch'],'exit_code':int(code),'tests':plugin.counts,'skips':plugin.skips,'command':'python scripts/run_directive_baseline.py → pytest -q -rs --disable-warnings','seconds':time.perf_counter()-start,'resources':monitor.report(),'cap':cap,'protected_source_unchanged':audit['protected_source_unchanged'],'live_before':audit['live_before'],'live_after':audit['live_after'],'live_sqlite_written':False,'main_training':False,'main_restored':False,'research_pilot_run':False,'knowledge_acceptance_passed':False}
    write_json(run/'baseline-audit.json',audit);write_json(run/'result.json',result);write_json(root/'artifacts/directive-v2-baseline.json',result)
    (run/'report.md').write_text('# DirectiveV2 baseline audit\n\nTest count is engineering evidence, not modelquality. Source in isolated branch; protected G1 unchanged.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');publish(run)
    print(json.dumps({k:result[k] for k in ('run','decision','tests','exit_code','seconds')}))
    return int(code)


if __name__=='__main__':raise SystemExit(main())
