"""Release only the owned idle VLM for the authorized native-precision pilot.

Restore the previous explicit intent and learning switch after the worker exits.
No private checkpoint restore, process kills, foreign model changes or weight edits.
"""
from datetime import datetime,timezone
import hashlib,importlib.util,json,os
from pathlib import Path
import subprocess,sys,time
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]


def status():
    with urlopen('http://127.0.0.1:8775/api/status?after=2147483647',timeout=5) as f:return json.load(f)


def main():
    run=ROOT/'local/lora-resource-window'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run.mkdir(parents=True)
    before=status()
    if before['cognition']['foreground_requests'] or before['cognition']['capture_active']:
        raise RuntimeError('User work/capture active; no resource-window change')
    intent=before['cognition']['state']['intent'];enabled=before['learning_enabled']
    with urlopen(Request('http://127.0.0.1:8775/api/status',headers={'X-SCARX-Backend':'hermes'}),timeout=5) as f:token=json.load(f)['token']
    def post(path,body):
        with urlopen(Request('http://127.0.0.1:8775'+path,data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json','X-SCARX-Token':token}),timeout=5) as f:return json.load(f)
    report={'previous_intent':intent,'previous_learning_enabled':enabled,'gateway_pid':before['pid'],
            'before_live_updates':before['neural']['updates'],'weight_or_checkpoint_restore_performed':False}
    try:
        post('/api/learning',{'enabled':False});post('/api/cognition',{'intent':'sleep'})
        deadline=time.monotonic()+45
        while status()['cognition']['VLM_resident']:
            if time.monotonic()>deadline:raise TimeoutError('Owned VLM did not release; no forced stop')
            time.sleep(.5)
        report['owned_idle_VLM_released']=True
        python=ROOT/'local/hybrid-v1/reference-env/Scripts/python.exe'
        with (run/'training.stdout.log').open('wb') as out,(run/'training.stderr.log').open('wb') as err:
            process=subprocess.Popen([str(python),'-B',str(ROOT/'scripts/train_hybrid_lora_error_repair.py')],cwd=ROOT,
                stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        report['worker_pid']=process.pid
        (run/'started.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'resource_window':str(run),'worker_pid':process.pid,'owned_idle_VLM_released':True}),flush=True)
        process.wait(timeout=1500);report['training_exit_code']=process.returncode
    finally:
        current=status()
        if current['cognition']['state']['intent']=='sleep':post('/api/cognition',{'intent':intent})
        if not status()['learning_enabled']:post('/api/learning',{'enabled':enabled})
        after=status();report.update(after_live_updates=after['neural']['updates'],
            restored_intent=after['cognition']['state']['intent'],restored_learning_enabled=after['learning_enabled'])
        (run/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (ROOT/'local/current-lora-resource-window.txt').write_text(str(run),encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
