"""Cold continuation for repaired peer607and matched/stopped research models."""
from pathlib import Path
import json,os,subprocess,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.forensic import r8_snapshot as S
from scarx.evaluation.forensic.r8_snapshot_fixture import fixture_model,continuation
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from forensic_run_r6 import run_child


def main():
    cap=apply_memory_limit();run=new_run(ROOT,'r8-current-state-cold-v1',{'display_title':'Cold state verification · no quality training',
        'kinds':['matched','stopped','peer607_mtsg','peer607_bf'],'fixture_only':True},data_description='One tiny technical update per continuation, no evaluation labels/dataset.')
    checks=[];commands=[]
    for kind in ('matched','stopped','peer607_mtsg','peer607_bf'):
        model=fixture_model(kind);state={'phase':'fixture','position':1,'replay':[{'id':'P0'}],'rng':np.random.default_rng(919).bit_generator.state}
        path=run/(kind+'.zip');S.save(model,path,state);raw=path.read_bytes()
        clone,controller=S.load(fixture_model(kind),path);expected=continuation(clone,controller)
        output=run/(kind+'-cold.json')
        command=run_child(run,kind,[sys.executable,'-X','utf8',str(ROOT/'scripts/r8_snapshot_cold.py'),kind,str(path),str(output)],ROOT,30);commands.append(command)
        actual=json.loads(output.read_text()) if output.exists() else None
        checks.append({'kind':kind,'state_and_next_update_exact':actual==expected,'checkpoint_unchanged':path.read_bytes()==raw})
    passed=all(c['state_and_next_update_exact'] and c['checkpoint_unchanged'] for c in checks) and all(c['exit_code']==0 and not c['error'] for c in commands)
    result={'checks':checks,'commands':commands,'cap':cap,'quality_training_calls':0,'technical_fixture_update_calls':8,
        'decision':'COLD_STATE_PASS_RESEARCH_ONLY' if passed else 'COLD_STATE_FAIL'}
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/r8-current-state-cold-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# CurrentR8cold state\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':run.name,'decision':result['decision']}));return int(not passed)


if __name__=='__main__':raise SystemExit(main())
