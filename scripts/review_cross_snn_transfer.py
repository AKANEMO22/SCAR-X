"""Pinned Gemini source/raw review only; never execute the peer learner."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.cross_snn_review import paired_distractor,formula_counterexamples
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.work_trace import activity

SHA='631c89041ec7ed3ce5d6b08373b68e9dc2e4a122'
PREVIOUS='bad058238083ec875aaef3a271e3f3c68680e3c1'
SURVEY='docs/research/collab/scarx-neural-round1-20261111/gemini-r7/cross-snn-mechanism-survey.md'
PATHS=[SURVEY,'src/scarx/evaluation/r7_hidden_plasticity.py','scripts/r7_investigation_suite.py',
       'artifacts/gemini-r7/investigation-suite.json','artifacts/gemini-r7/dev-screen.json']


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def main():
    cap=apply_memory_limit();run=new_run(ROOT,'cross-snn-transfer-review-v1',{
        'display_title':'Gemini Cross-SNN · independent source/raw review, no training',
        'peer_sha':SHA,'base_sha':PREVIOUS,'candidate_training':False,'confirmation':False},
        data_description='Pinned public Git blobs, historical DEV summaries and scalar arithmetic counterexamples only.',
        data_files=['experiments/cross-snn-transfer-review-v1.md','tests/test_cross_snn_review.py',
                    'configs/cross_snn_primary_source_checks_v1.json'])
    activity(run,'Review khảo sát Cross-SNN','Tính lại paired DEV metrics và đối chiếu phương trình; không chạy learner.',status='started',evidence='manifest.json')
    with ResourceMonitor(interval=.02) as monitor:
        blobs={p:git('show',SHA+':'+p) for p in PATHS};inputs=run/'inputs';inputs.mkdir()
        for i,(path,raw) in enumerate(blobs.items()):(inputs/(str(i)+'-'+Path(path).name)).write_bytes(raw)
        tree=git('ls-tree','-r','--name-only',SHA).decode('utf-8').splitlines()
        diff=git('diff','--name-only',PREVIOUS,SHA).decode('utf-8').splitlines()
        suite=json.loads(blobs['artifacts/gemini-r7/investigation-suite.json'])
        result={'peer_sha':SHA,'previous_sha':PREVIOUS,'changed_paths':diff,
            'input_sha256':{p:hashlib.sha256(raw).hexdigest() for p,raw in blobs.items()},
            'paired_distractor':paired_distractor(suite['distractor_analysis']),
            'formula_counterexamples':formula_counterexamples(),
            'multiscale_named_files':[p for p in tree if any(word in Path(p).name.lower() for word in ('multiscale','mtsg','sensory','cascade'))],
            'raw_suite_sections':list(suite),
            'new_commit_changes_only_survey':diff==[SURVEY],
            'new_neural_training':False,'new_generation':False,'main_or_live_modified':False,
            'decision':'REQUEST_CHANGES_SURVEY; KEEP_MECHANISM_HYPOTHESES_RESEARCH_ONLY'}
        monitor.check()
    result['resources']=monitor.report();result['cap']=cap
    result['resource_scope']='Git loading, arithmetic and raw-statistics phase only; pytest child excluded'
    command=[sys.executable,'-X','utf8','-m','pytest','-q','-p','no:cacheprovider','tests/test_cross_snn_review.py','--junitxml='+str(run/'audit-tests.xml')]
    child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=30)
    (run/'audit-tests.log').write_text(child.stdout+child.stderr,encoding='utf-8')
    result['audit_tests']={'command':command,'exit_code':child.returncode}
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/cross-snn-transfer-review-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# Gemini Cross-SNN independent review\n\nNo new training; see the Vietnamese review in docs/research/forensic-r6-20261111.\n\n'+json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    activity(run,'Hoàn tất review Cross-SNN',result['decision'],status='completed',evidence='result.json')
    print(json.dumps({'run':run.name,'paired_probe_effect_pp':result['paired_distractor']['means']['probe_effect_pp'],
        'tests_exit':child.returncode,'decision':result['decision']}))
    return child.returncode


if __name__=='__main__':raise SystemExit(main())
