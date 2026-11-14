"""One current host-memory point for verified roots; never patch active monitor."""

from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.device_budget import gpu_headroom
from scarx.hybrid.store import Store
from monitor_hybrid_endurance import inspect
from audit_hybrid_endurance import gateway_identity


def main():
    began=time.monotonic()
    audit=ROOT/'runs/20261114T072942Z-endurance-data-audit-v1-a2ccf7'
    report=json.loads((audit/'result.json').read_text(encoding='utf-8'))
    run=ROOT/'runs'/report['parent_endurance_run']
    lab=inspect(run)
    if not lab['actual_process_live']:raise RuntimeError('Current-point supplement requires the same live lab')
    native=SharedNative(ROOT);native.check()
    gateway=gateway_identity()
    if gateway is None:raise RuntimeError('Current Hermes gateway identity unverified')
    roots={'SCAR-Xgateway':native.owner,'private_endurance':psutil.Process(lab['pid']),
           'current_Hermesgateway':psutil.Process(gateway)}
    desktop_exe=(Path(os.environ['LOCALAPPDATA'])/
        'hermes/hermes-agent/apps/desktop/release/win-unpacked/Hermes.exe').resolve()
    desktop_ids=[]
    for pid in report['resources']['initial_client_roots_used_by_fixed_monitor']:
        try:
            p=psutil.Process(pid)
            if Path(p.exe()).resolve()==desktop_exe:
                roots['HermesDesktop']=p;desktop_ids.append(pid)
        except psutil.Error:continue
    if len(desktop_ids)!=1:raise RuntimeError('Expected exactly one verified recorded desktop root')
    processes={}
    root_rows=[]
    for role,root in roots.items():
        created=root.create_time();root_rows.append({'role':role,'pid':root.pid,'created_epoch':created})
        for process in [root,*root.children(recursive=True)]:
            try:
                birth=process.create_time();memory=process.memory_info()
                if not process.is_running() or process.create_time()!=birth:continue
                processes[process.pid,birth]={'pid':process.pid,'created_epoch':birth,'name':process.name(),
                    'rss_bytes':memory.rss,'private_commit_bytes':getattr(memory,'private',getattr(memory,'pagefile',memory.vms))}
            except psutil.Error:continue
        if root.create_time()!=created:raise RuntimeError('Root identity changed during read-only snapshot')
    last=set(report['resources']['last_resource_PIDs'])
    current=list(processes.values())
    only_current=[row for row in current if row['pid'] not in last]
    result={'checked_UTC':datetime.now(timezone.utc).isoformat(),'parent_audit':audit.name,
        'parent_endurance':run.name,'validated_roots':root_rows,'processes':current,
        'current_group_RSS_bytes':sum(p['rss_bytes'] for p in current),
        'current_group_private_commit_bytes':sum(p['private_commit_bytes'] for p in current),
        'current_Hermes_gateway_root_absent_from_original_last_sample':gateway not in last,
        'current_processes_not_in_original_last_sample':only_current,
        'current_global_GPU_headroom':gpu_headroom(),'deduplicated_by_PID_and_creation_time':True,
        'historical_peak_reconstructed_or_changed':False,
        'current_point_not_simultaneous_with_old_heartbeat_or_historical_peaks':True,
        'auditor_process_not_in_reported_owned_roots':True,
        'active_run_source_changed_or_restarted':False,
        'scope':'Host current-point supplement. Does not recover missing historical gateway peaks or prove a whole-CPU RAM cap.',
        'seconds':time.monotonic()-began}
    if result['seconds']>10:raise TimeoutError('Current-point supplement exceeded10s')
    destination=audit/'gateway-coverage-supplement-v1.json'
    with destination.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    for path in (Path(__file__),ROOT/'experiments/hybrid-endurance-independent-audit-v1.md'):
        rel=path.relative_to(ROOT);p=audit/'supplemental-source-v1'/rel
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(path.read_bytes())
    trace=Store(ROOT/'local/hybrid-v1')
    trace.event('endurance_data_audit.current_gateway_scope',
        {k:v for k,v in result.items() if k not in ('processes',)},'orchestrator')
    trace.close()
    print(json.dumps({k:result[k] for k in ('current_group_RSS_bytes','current_group_private_commit_bytes',
        'current_Hermes_gateway_root_absent_from_original_last_sample','seconds')}))


if __name__=='__main__':main()
