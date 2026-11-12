"""Complete checkout regression with bounded partitioned Torch controls."""
from pathlib import Path
import json,os,subprocess,sys,xml.etree.ElementTree as ET
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from forensic_run_r6 import run_child


def main():
    cap=apply_memory_limit();files=sorted(ROOT.glob('tests/test_*.py'),key=lambda p:p.name)
    groups=[('prefix',[p for p in files if p.name<'test_visual_word_control.py']),
            ('visual',[p for p in files if 'test_visual_word_control.py'<=p.name<'test_word_control.py']),
            ('word-isolated',[p for p in files if p.name=='test_word_control.py']),
            ('suffix',[p for p in files if p.name>'test_word_control.py'])]
    run=new_run(ROOT,'r8-repairs-regression-v1',{'display_title':'Full R8 repair checkout regression and native parity','new_quality_training':False},
        data_files=[p.relative_to(ROOT).as_posix() for p in files])
    commands=[run_child(run,'native-build',[sys.executable,'scripts/build_native.py'],ROOT,90)]
    totals={key:0 for key in ('tests','failures','errors','skipped')}
    for name,group in groups:
        if commands[-1]['exit_code']!=0:break
        xml=run/(name+'.xml');command=[sys.executable,'-X','utf8','-m','pytest','-q','-rs','-p','no:cacheprovider',
            *[p.relative_to(ROOT).as_posix() for p in group],'--junitxml='+str(xml)]
        commands.append(run_child(run,name,command,ROOT,240))
        if xml.exists():
            for suite in ET.parse(xml).getroot().findall('testsuite'):
                for key in totals:totals[key]+=int(suite.attrib.get(key,0))
    result={'commands':commands,'totals':totals,'test_files':len(files),'cap':cap,'new_quality_training':False,
        'decision':'REGRESSION_PASS' if len(commands)==5 and all(c['exit_code']==0 and not c['error'] for c in commands) else 'REGRESSION_FAILED'}
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/r8-repairs-regression-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# R8 regression\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':run.name,'totals':totals,'decision':result['decision']}));return int(result['decision']!='REGRESSION_PASS')


if __name__=='__main__':raise SystemExit(main())
