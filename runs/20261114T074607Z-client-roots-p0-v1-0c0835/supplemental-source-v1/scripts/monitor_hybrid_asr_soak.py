"""Verify the actual existing ASR soak worker/source/receipt; never restart it."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from monitor_hybrid_endurance import read_heartbeat


def inspect(run):
    run=run.resolve()
    if not run.is_relative_to(ROOT/'runs'):raise ValueError('Expected owned run')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    active=None
    try:
        process=psutil.Process(manifest['pid'])
        args=process.cmdline()
        script=(ROOT/'scripts/soak_hybrid_whisper_small.py').resolve()
        correct_script=any(Path(arg).resolve()==script for arg in args if arg.endswith('.py'))
        correct_horizon=args[args.index('--seconds')+1]==str(manifest['seconds'])
        if (process.is_running() and abs(process.create_time()-manifest['process_created_epoch'])<.01
                and correct_script and correct_horizon):active=process
    except (psutil.Error,ValueError,IndexError):pass
    heartbeat=read_heartbeat(run/'heartbeat.json') if (run/'heartbeat.json').exists() else None
    final=json.loads((run/'result.json').read_text(encoding='utf-8')) if (run/'result.json').exists() else None
    overlay=ROOT/'local'/run.name/'resources_overlay.py'
    return {'checked_epoch':time.time(),'run_id':run.name,'actual_process_live':active is not None,
        'pid':manifest['pid'],'process_created_epoch':active.create_time() if active else None,
        'state':'terminal_receipt_present' if final else 'verified_live_process' if active else 'pid_missing_no_final_receipt',
        'frozen_source_matches':all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected
            for rel,expected in manifest['source_sha256'].items()),
        'qualified_monitor_overlay_matches':overlay.is_file() and hashlib.sha256(overlay.read_bytes()).hexdigest()==
            '8774db6c6f91adb8bcb2f990341aa086eebe88f57b047521837603829e0af2c8',
        'elapsed_seconds':heartbeat.get('elapsed_seconds') if heartbeat else None,
        'requested_seconds':manifest['seconds'],'complete_calls':heartbeat.get('complete_calls') if heartbeat else None,
        'completed_audio_seconds':heartbeat.get('completed_audio_seconds') if heartbeat else None,
        'heartbeat_age_seconds':time.time()-(run/'heartbeat.json').stat().st_mtime if heartbeat else None,
        'final_verdict':final.get('verdict') if final else None,'live_ASR_changed':False,
        'observation_does_not_restart_or_modify_job':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    print(json.dumps(inspect(a.run),ensure_ascii=False))
