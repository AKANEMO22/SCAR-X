"""Bounded supervisor for independent historical audit/reproduction children."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import psutil
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.experiment_status import progress

SOURCES={
 'r6d':(ROOT.parent/'forensic-source-r6d-4bacd7b','4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893'),
 'r6b':(ROOT.parent/'forensic-source-r6b-c39e9db','c39e9db7771582d9933af402a209917328b71119')}
SOURCES['gemini']=(ROOT.parent/'forensic-source-gemini-8f24a5a','8f24a5abde0598201e558eee559564ccde889145')


def run_child(run,label,command,cwd,timeout):
    env=dict(os.environ,PYTHONPATH=str(cwd/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    start=time.perf_counter();process=None;error=None
    with (run/(label+'.log')).open('wb') as log,ResourceMonitor(interval=.1) as monitor:
        try:
            process=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            print(json.dumps({'phase':label,'pid':process.pid,'run':run.name}),flush=True)
            while process.poll() is None:
                monitor.check()
                progress(run,'Đang kiểm toán '+label,metrics={'owned_child_pid':process.pid,'seconds':time.perf_counter()-start,'fresh_confirmations':0})
                if time.perf_counter()-start>timeout:raise TimeoutError('Locked child timeout')
                time.sleep(.5)
            if process.returncode:raise RuntimeError('Child exit '+str(process.returncode))
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        finally:
            if process is not None and process.poll() is None:
                try:children=psutil.Process(process.pid).children(recursive=True)
                except psutil.Error:children=[]
                for child in reversed(children):
                    try:child.kill()
                    except psutil.Error:pass
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    result={'command':command,'cwd':str(cwd),'exit_code':None if process is None else process.returncode,
        'error':error,'seconds':time.perf_counter()-start,'resources':monitor.report(),'log':label+'.log'}
    write_json(run/(label+'-command.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('engineering','baseline','baseline-sharded','baseline-tail','native','gemini-engineering','gemini-replay','r6d-matrix','r6d-controls','r6b'),required=True)
    args=parser.parse_args()
    engineering_phase=args.phase in ('baseline','baseline-sharded','baseline-tail','native','engineering','gemini-engineering')
    protocol=ROOT/'docs/research/forensic-r6-20261111/PROTOCOL_LOCKED.md'
    expected=subprocess.check_output(['git','-C',str(ROOT),'show','794af32:docs/research/forensic-r6-20261111/PROTOCOL_LOCKED.md'])
    if protocol.read_bytes().replace(b'\r\n',b'\n')!=expected.replace(b'\r\n',b'\n'):raise ValueError('Locked audit protocol changed')
    for source,sha in SOURCES.values():
        if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=sha:raise ValueError('Pinned source moved')
        if subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Pinned tracked source dirty')
    cap=apply_memory_limit(2_000_000_000)
    run=new_run(ROOT,'forensic-'+args.phase+'-v1',{'display_title':'Codex independent audit · '+args.phase,
        'run_class':'ENGINEERING_CHECK' if engineering_phase else 'HISTORICAL_REPRODUCTION',
        'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'protocol_lock_commit':'794af32',
        'source_commits':{k:v[1] for k,v in SOURCES.items()},'fresh_confirmation':False,'new_candidate_or_learning_rule':False},
        data_description='Existing historical synthetic generators/seeds and raw result references; no private/live input.',
        data_files=['docs/research/forensic-r6-20261111/PROTOCOL_LOCKED.md'])
    commands=[]
    if args.phase.startswith('gemini'):
        source=SOURCES['gemini'][0]
        if args.phase=='gemini-engineering':
            commands.append(run_child(run,'gemini-tests',[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_r6_associative.py','--junitxml='+str(run/'gemini-tests.xml')],source,120))
        else:
            commands.append(run_child(run,'gemini-historical',[sys.executable,str(ROOT/'scripts/forensic_gemini_worker.py'),str(source),str(run/'gemini-replay')],source,300))
    elif args.phase=='engineering':
        for kind in ('r6d','r6b'):
            source=SOURCES[kind][0]
            tests=['tests/test_r6d_audit.py'] if kind=='r6d' else ['tests/test_r6_toys.py']
            commands.append(run_child(run,kind+'-author-tests',[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*tests,'--junitxml='+str(run/(kind+'-author-tests.xml'))],source,120))
            commands.append(run_child(run,kind+'-independent-checks',[sys.executable,str(ROOT/'scripts/forensic_runtime_checks.py'),'--source',str(source),'--kind',kind],source,120))
    elif args.phase=='baseline':
        commands.append(run_child(run,'audit-baseline',[sys.executable,'-m','pytest','-q','-rs','-p','no:cacheprovider','--junitxml='+str(run/'baseline-junit.xml')],ROOT,300))
    elif args.phase=='baseline-sharded':
        files=sorted((ROOT/'tests').glob('test_*.py'))
        # Same complete file set, split before the large visual/word control tail.
        # This addresses accumulated test-process memory, not the serving budget.
        groups=[[p for p in files if p.name<'test_visual_word_control.py'],
                [p for p in files if p.name>='test_visual_word_control.py']]
        for index,group in enumerate(groups,1):
            commands.append(run_child(run,'audit-baseline-shard'+str(index),[sys.executable,'-m','pytest','-q','-rs','-p','no:cacheprovider',
                *[str(p.relative_to(ROOT)) for p in group],'--junitxml='+str(run/('shard'+str(index)+'.xml'))],ROOT,300))
    elif args.phase=='baseline-tail':
        files=sorted(p for p in (ROOT/'tests').glob('test_*.py') if p.name>='test_visual_word_control.py')
        # test_word_control launches an isolated Torch child. Run it without a
        # parent pytest process that already imported Torch via other test files.
        groups=[[p for p in files if p.name<'test_word_control.py'],
                [p for p in files if p.name=='test_word_control.py'],
                [p for p in files if p.name>'test_word_control.py']]
        for index,group in enumerate(groups,1):
            commands.append(run_child(run,'tail'+str(index),[sys.executable,'-m','pytest','-q','-rs','-p','no:cacheprovider',
                *[str(p.relative_to(ROOT)) for p in group],'--junitxml='+str(run/('tail'+str(index)+'.xml'))],ROOT,120))
    elif args.phase=='native':
        commands.append(run_child(run,'build-native',[sys.executable,'scripts/build_native.py'],ROOT,120))
        if not commands[-1]['error']:
            commands.append(run_child(run,'native-parity',[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_native.py','--junitxml='+str(run/'native-parity.xml')],ROOT,120))
    else:
        kind='r6d' if args.phase.startswith('r6d') else 'r6b';source=SOURCES[kind][0]
        timeout=1200 if args.phase=='r6d-matrix' else 600
        commands.append(run_child(run,args.phase,[sys.executable,str(ROOT/'scripts/forensic_replay_worker.py'),'--source',str(source),
            '--output',str(run/'replay'),'--phase',args.phase],source,timeout))
    failed=[c for c in commands if c['error']]
    result={'decision':'AUDIT_EXECUTION_COMPLETE_REQUIRES_ANALYSIS' if not failed else 'REPRODUCTION_FAILED',
        'phase':args.phase,'run_class':'ENGINEERING_CHECK' if engineering_phase else 'HISTORICAL_REPRODUCTION',
        'commands':commands,'cap':cap,'source_commits':{k:v[1] for k,v in SOURCES.items()},'fresh_confirmation':False,
        'candidate_equations_changed':False,'production_or_live_state_modified':False,'error':failed[0]['error'] if failed else None}
    write_json(run/'result.json',result)
    (run/'report.md').write_text('# Forensic '+args.phase+'\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    folder=ROOT/'artifacts/forensic-r6';folder.mkdir(exist_ok=True)
    write_json(folder/(args.phase+'-receipt.json'),{'run':run.relative_to(ROOT).as_posix(),**result})
    print(json.dumps({'run':run.name,'phase':args.phase,'error':result['error']}),flush=True)
    return int(bool(failed))


if __name__=='__main__':raise SystemExit(main())
