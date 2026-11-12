"""Existing-suite regression for isolated approved P0; no research campaign."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,xml.etree.ElementTree as ET
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.work_trace import activity


def protected_hashes():
    files=[p for base in ('src/scarx/neural','src/scarx/app','src/scarx/learning') for p in (ROOT/base).rglob('*')
           if p.is_file() and '__pycache__' not in p.parts]
    files+=[ROOT/'artifacts/generation1-release.json']
    release=json.loads(files[-1].read_text(encoding='utf-8'));files.append(ROOT/release['checkpoint'])
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    cap=apply_memory_limit();before=protected_hashes()
    tests=sorted(ROOT.glob('tests/test_*.py'),key=lambda p:p.name)
    groups=[('prefix',[p for p in tests if p.name<'test_visual_word_control.py']),
            ('visual-and-word',[p for p in tests if 'test_visual_word_control.py'<=p.name<'test_word_control.py']),
            ('isolated-word-control',[p for p in tests if p.name=='test_word_control.py']),
            ('suffix',[p for p in tests if p.name>'test_word_control.py'])]
    assert sum(len(g) for _,g in groups)==len(tests)
    run=new_run(ROOT,'output-codec-p0-regression-v1',{'display_title':'P0 codec · existing regression and CPU/native parity',
        'scope':'Existing unit/regression fixture checks only; no dataset training or quality/confirmation experiment',
        'groups':'Separate word-control child from any parent that imported Torch; same assertions and 2GB budget',
        'main_or_live_migrated':False},data_files=[p.relative_to(ROOT).as_posix() for p in tests])
    activity(run,'Kiểm tra hồi quy P0','Build kernel hiện có và chạy các test fixture trên worktree riêng.',status='started',evidence='manifest.json')
    records=[];error=None
    with ResourceMonitor(interval=.02) as monitor:
        try:
            command=[sys.executable,'scripts/build_native.py'];start=time.perf_counter()
            build=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            (run/'native-build.log').write_text(build.stdout+build.stderr,encoding='utf-8')
            records.append({'name':'native-build','command':command,'exit_code':build.returncode,'seconds':time.perf_counter()-start})
            if build.returncode:raise RuntimeError('Existing native kernel build failed')
            monitor.check()
            for name,files in groups:
                xml=run/(name+'.xml');command=[sys.executable,'-X','utf8','-m','pytest','-q','-rs','-p','no:cacheprovider',
                    *[p.relative_to(ROOT).as_posix() for p in files],'--junitxml='+str(xml)]
                start=time.perf_counter()
                child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=240)
                (run/(name+'.log')).write_text(child.stdout+child.stderr,encoding='utf-8')
                suites=ET.parse(xml).getroot().findall('testsuite') if xml.exists() else []
                counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
                records.append({'name':name,'command':command,'exit_code':child.returncode,'seconds':time.perf_counter()-start,'junit':counts})
                write_json(run/(name+'-receipt.json'),records[-1]);monitor.check()
                if child.returncode:raise RuntimeError('Regression group failed: '+name)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
    totals={key:sum(r.get('junit',{}).get(key,0) for r in records) for key in ('tests','failures','errors','skipped')}
    unchanged=protected_hashes()==before
    result={'records':records,'junit_totals':totals,'test_files':len(tests),'protected_hashes':before,
        'protected_files_unchanged':unchanged,'cap':cap,'resources':monitor.report(),'error':error,
        'all_groups_completed':len(records)==1+len(groups),'dataset_training':False,'new_teacher_calls':0,
        'decision':'REGRESSION_PASS' if error is None and unchanged and len(records)==1+len(groups) else 'REGRESSION_FAILED'}
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/output-codec-p0-regression-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# P0 existing-suite regression\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    activity(run,'Hoàn tất hồi quy P0',result['decision'],status='completed' if result['decision']=='REGRESSION_PASS' else 'failed',evidence='result.json')
    print(json.dumps({'run':run.name,'decision':result['decision'],'totals':totals,'error':error}))
    return int(result['decision']!='REGRESSION_PASS')


if __name__=='__main__':raise SystemExit(main())
