"""Verify merged research plus serving isolation without weakening RAM guards.

The API teaching test uses process memory accounting; execute its module in a
fresh process, separately from numerical/PyTorch research-test allocations.
No tests are skipped. This is engineering verification, never a DEV run.
"""
from pathlib import Path
import json,os,subprocess,sys,time,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'local/consolidation-validation';out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    groups=[('api-memory-isolated',['tests/test_openai_backend.py']),('remaining-suite',['--ignore=tests/test_openai_backend.py'])]
    records=[];start=time.perf_counter()
    for name,args in groups:
        command=[sys.executable,'-X','utf8','-m','pytest','-q','-p','no:cacheprovider',*args,'--junitxml='+str(out/(name+'.xml'))]
        r=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',timeout=240)
        (out/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
        tree=ET.parse(out/(name+'.xml')).getroot();suites=list(tree.iter('testsuite'))
        rec={'name':name,'returncode':r.returncode,'tests':sum(int(s.attrib['tests']) for s in suites),
             'failures':sum(int(s.attrib.get('failures',0)) for s in suites),'errors':sum(int(s.attrib.get('errors',0)) for s in suites),
             'skipped':sum(int(s.attrib.get('skipped',0)) for s in suites),'command':command}
        records.append(rec);print(json.dumps(rec),flush=True)
        if r.returncode:raise RuntimeError('Verification failed, preserve logs: '+name)
    production=['src/scarx/neural','src/scarx/app','src/scarx/learning','integrations','run_app.py','capabilities.json','project.json']
    subprocess.run(['git','diff','--exit-code','e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541','--',*production],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/research/query_catalog.py','verify'],cwd=ROOT,env=env,check=True)
    result={'status':'PASS','groups':records,'tests_total':sum(r['tests'] for r in records),'failures':0,'errors':0,
       'skipped':sum(r['skipped'] for r in records),'wall_seconds':time.perf_counter()-start,'production_source_unchanged':True,
       'new_quality_experiments':0,'active_R13_family':'No mutations or process/cache operations',
       'scope':'All tests run; process-sensitive API module isolated, no RAM guard changes or test skips. Earlier monolithic failure preserved separately.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
