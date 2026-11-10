"""Recorded product acceptance, not a research success claim."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import uuid
from scarx.app.state import Workbench
from scarx.evaluation.runs import new_run,write_json

root=Path(__file__).resolve().parents[1]
run=new_run(root,'mvp-v01-acceptance',{'seed':17,'build_iteration':2,'A_batches':2,'B_batches':2,'samples_per_batch':64,'epochs':4,'research_claim':False})
directory=root/'local/mvp-acceptance'/uuid.uuid4().hex
app=Workbench(directory)
try:
    initial=app.active; before=app._evaluate(); steps=[]
    for task in ('A','A','B','B'):
        row=app.teach({'task':task,'count':64,'epochs':4,'replay':True}); steps.append(row)
    learned=app.active; learned_payload=app._pack(); final=app._evaluate()
    resources=app.resources(); app.close(); app=None
    code="import sys,json,hashlib; from scarx.app.state import Workbench; a=Workbench(sys.argv[1]); print(json.dumps({'scores':a._evaluate(),'sha256':hashlib.sha256(a._pack()).hexdigest(),'updates':a.learner.model.updates})); a.close()"
    cp=subprocess.run([sys.executable,'-c',code,str(directory)],capture_output=True,text=True,check=True,timeout=30)
    restarted=json.loads(cp.stdout)
    assert restarted['scores']==final
    assert restarted['sha256']==hashlib.sha256(learned_payload).hexdigest()
    app=Workbench(directory); app.restore(initial); restored_scores=app._evaluate(); app.restore(learned)
    assert restored_scores==before and app._pack()==learned_payload
    assert steps[1]['after']['A']['accuracy']-before['A']['accuracy']>=.2
    assert all(r['after'][task]['seen_exact']==0 for r in steps for task in ('A','B'))
    result={'before':before,'steps':steps,'final':final,'restart':restarted,'restore_exact':True,
            'resources':resources,'data_directory':str(directory),'initial_checkpoint':initial,'learned_checkpoint':learned,
            'acceptance_pass':True,'research_gate_pass':False,
            'scope':'Product regression/demo after iterative development; 4 known signal types with unseen amplitude/noise variants. No claim of learned language, novel concepts, hidden representation advantage or solved continual learning.',
            'first_smoke_failure':'One A batch gained14.0625pp, below20pp target. Build iteration2 uses two A batches then two B batches; no neural rule/encoder changes.'}
    write_json(run/'result.json',result)
    lines=['# MVP V0.1 measured demo','',result['scope'],'','| Stage | A accuracy | B accuracy | Updates in batch |','|---|---:|---:|---:|',
           f"| Before | {before['A']['accuracy']:.4f} | {before['B']['accuracy']:.4f} | 0 |"]
    lines += [f"| After {r['task']} #{i+1} | {r['after']['A']['accuracy']:.4f} | {r['after']['B']['accuracy']:.4f} | {r['updates']} |" for i,r in enumerate(steps)]
    lines += ['',f'Restart and restore: exact. RSS={resources["rss_bytes"]} bytes; commit={resources["commit_bytes"]} bytes.',
              '',result['first_smoke_failure'],'','Product acceptance passed; research forgetting/representation claims remain unproven.']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'artifacts/mvp-v01-summary.json').write_text(json.dumps({'run':run.relative_to(root).as_posix(),'data_dir':directory.relative_to(root).as_posix(),'acceptance_pass':True},indent=2),encoding='utf-8')
    print(json.dumps({'run':str(run),'before':before,'after':final,'restart_exact':True},indent=2))
finally:
    if app is not None: app.close()
