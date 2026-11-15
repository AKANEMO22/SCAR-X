"""Read-only late-window resource observer, never backfills earlier peaks."""
from datetime import datetime,timezone
import json,os
from pathlib import Path
import subprocess,time
import psutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    run=Path((ROOT/'local/current-shared-lora-retention20.txt').read_text(encoding='utf-8').strip())
    if not run.is_relative_to(ROOT/'runs'):raise ValueError('Run outside workspace')
    target=ROOT/'scripts/train_shared_lora_retention20.py';matches=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            if not process.info['name'].casefold().startswith('python'):continue
            args=process.info['cmdline'] or []
            if any(Path(a).resolve()==target for a in args[1:] if not a.startswith('--')):matches.append(process)
        except (psutil.Error,OSError,ValueError):continue
    if not matches:raise RuntimeError('No confirmed live training process')
    # Windows venv launcher and actual interpreter share the same args; all
    # exact identities are sampled, RSS sum is not deduplicated physical RAM.
    owners={p.pid:p.create_time() for p in matches};started=time.time();rows=[]
    with (run/'late-resource-samples.jsonl').open('x',encoding='utf-8') as stream:
        while time.time()-started<900:
            live=[]
            for pid,created in owners.items():
                try:
                    p=psutil.Process(pid)
                    if abs(p.create_time()-created)<.01 and p.is_running():live.append(p)
                except psutil.NoSuchProcess:continue
            if not live:break
            memories=[p.memory_info() for p in live]
            out=subprocess.check_output([str(Path(os.environ['WINDIR'])/'System32/nvidia-smi.exe'),'--id=0',
                '--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=3,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            free,total=[int(v.strip())*1024**2 for v in out.strip().split(',')]
            row={'epoch':time.time(),'live_exact_owned_pids':[p.pid for p in live],
                'RSS_sum_bytes':sum(m.rss for m in memories),'private_commit_sum_bytes':sum(getattr(m,'private',0) for m in memories),
                'dedicated_GPU_free_bytes':free,'dedicated_GPU_total_bytes':total}
            stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
            time.sleep(2)
    report={'scope':'Late observation window only, earlier stages not backfilled; RSS sum is not physical-RAM deduplication',
        'started_epoch':started,'ended_epoch':time.time(),'owned_process_identities':owners,'samples':len(rows),
        'observed_RSS_sum_peak_bytes':max([r['RSS_sum_bytes'] for r in rows]+[0]),
        'observed_private_commit_sum_peak_bytes':max([r['private_commit_sum_bytes'] for r in rows]+[0]),
        'observed_min_dedicated_GPU_free_bytes':min([r['dedicated_GPU_free_bytes'] for r in rows],default=None),
        'model_calls_or_training_state_changes':0,'earlier_full_run_resource_PASS_certified':False}
    with (run/'late-resource-observer.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
