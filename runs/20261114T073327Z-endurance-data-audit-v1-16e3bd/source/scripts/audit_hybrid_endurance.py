"""Read-only independent snapshot of one owned endurance run; no job mutation."""

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid
import zipfile
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from scarx.hybrid.store import encode,Store
from scarx.evaluation.endurance_independent_audit import (
    final_audit_decision,original_long_gate,video_coverage,resource_coverage)
from monitor_hybrid_endurance import inspect

KINDS=('shared_endurance.started','shared_endurance.heartbeat',
       'shared_endurance.measured_video','shared_endurance.completed')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def readonly(path):
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    db.execute('BEGIN')
    return db


def gateway_identity():
    path=Path(os.environ['LOCALAPPDATA'])/'hermes/gateway.pid'
    try:
        descriptor=json.loads(path.read_text(encoding='utf-8'))
        process=psutil.Process(descriptor['pid'])
        if (descriptor.get('kind')!='hermes-gateway'
                or abs(process.create_time()-descriptor['start_time']/100)>.1):
            return None
        if not Path(process.exe()).resolve().is_relative_to(path.parent.resolve()):
            return None
        return process.pid
    except (OSError,ValueError,KeyError,psutil.Error):
        return None


def private_snapshot(directory):
    db=readonly(directory/'hybrid.sqlite')
    try:
        counts=dict(db.execute('SELECT kind,count(*) FROM events GROUP BY kind'))
        objects=dict(db.execute('SELECT kind,count(*) FROM objects GROUP BY kind'))
        stream_rows=db.execute("SELECT payload FROM objects WHERE kind='stream' ORDER BY rowid").fetchall()
        if len(stream_rows)>1000:raise ValueError('Private stream summary exceeds registered bound')
        streams=[json.loads(row[0]) for row in stream_rows]
        evidence_count=db.execute('SELECT COUNT(*) FROM evidence').fetchone()[0]
        checkpoint_rows=db.execute('SELECT id,digest,payload FROM checkpoints').fetchall()
        metadata=[]
        pins={row[0] for row in db.execute("SELECT id FROM objects WHERE kind='checkpoint_pin'")}
        for row in checkpoint_rows:
            payload=row['payload']
            if len(payload)>16_000_000 or sha(payload)!=row['digest']:
                raise ValueError('Checkpoint size/hash failure')
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if sum(info.file_size for info in archive.infolist())>16_000_000:
                    raise ValueError('Unbounded checkpoint archive')
            with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
                meta=json.loads(arrays['metadata'].tobytes())
            controller=meta['controller']
            bounds={'replay':len(meta['replay'])<=64,'probes':len(controller['probes'])<=16,
                    'goals':len(controller['goals'])<=64,'review_pending':len(controller['review']['pending'])<=32}
            if not all(bounds.values()):raise ValueError('Recorded private state exceeded a registered bound')
            metadata.append({'id':row['id'],'sha256':row['digest'],'bytes':len(payload),
                'pinned':row['id'] in pins,'updates':meta['updates'],'hidden_updates':meta['hidden_updates'],
                'replay_seen':meta['replay_seen'],'replay_items':len(meta['replay']),'bounds':bounds})
        active_row=db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()
        active_id=json.loads(active_row[0])['id']
        active=next(row for row in metadata if row['id']==active_id)
        learning=[json.loads(row[0]) for row in db.execute("SELECT payload FROM events WHERE kind='local_learning_update'")]
        return {'event_counts':counts,'object_counts':objects,'streams':streams,
            'evidence_count':evidence_count,'checkpoints':metadata,'pinned_checkpoint_ids':sorted(pins),
            'active_checkpoint':active,'accepted_learning_events':sum(row.get('accepted') is True for row in learning),
            'learning_events_count':len(learning),'all_recorded_checkpoint_hashes_and_bounds_valid':True}
    finally:db.close()


def native_identity(start_event,started_epoch):
    pid=start_event.get('native_worker_pid')
    try:
        process=psutil.Process(pid)
        expected=ROOT/'local/hybrid-v1/runtime/llama.cpp-b12360-cuda12.4/llama-server.exe'
        args=process.cmdline()
        def argument(flag):return args[args.index(flag)+1]
        valid=(process.is_running() and process.create_time()<started_epoch
            and Path(process.exe()).resolve()==expected.resolve()
            and argument('--device')=='CUDA0' and argument('-ngl')=='99'
            and argument('-c')=='16384' and argument('--cache-type-k')=='bf16'
            and argument('--cache-type-v')=='bf16'
            and Path(argument('-m')).resolve()==(ROOT/'local/hybrid-v1/models/native/qwen-native-bf16.gguf').resolve())
        return {'pid':pid,'created_epoch':process.create_time(),'validated_current_identity_matches':bool(valid)}
    except (psutil.Error,ValueError,IndexError,TypeError):
        return {'pid':pid,'validated_current_identity_matches':False}


def main(run):
    began=time.monotonic();run=run.resolve()
    if not run.is_relative_to(ROOT/'runs'):raise ValueError('Expected owned registered run')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest['duration_seconds']!=28800:raise ValueError('This auditor only judges original8h horizon')
    state=inspect(run)
    archive_valid=True
    for rel,expected in manifest['source_sha256'].items():
        archive_valid &= sha((run/'source'/rel).read_bytes())==expected
    prior=ROOT/'runs'/manifest['preflight_run_id']
    preflight=json.loads((prior/'result.json').read_text(encoding='utf-8'))
    prior_manifest=json.loads((prior/'manifest.json').read_text(encoding='utf-8'))
    preflight_valid=bool(sha((prior/'result.json').read_bytes())==manifest['preflight_result_sha256']
        and prior_manifest['source_sha256']==manifest['source_sha256']
        and preflight['gate_pass'] and not preflight['failures']
        and preflight['elapsed_seconds']>=240 and len(preflight['video_cycles'])>=1
        and len(preflight['checkpoints'])>=1 and preflight['final_neural']['updates']>0
        and preflight['final_neural']['device']=='gpu' and preflight['critical_quantum_failures']==0)
    observer=readonly(ROOT/'local/hybrid-v1/hybrid.sqlite')
    try:
        cutoff=observer.execute('SELECT MAX(id) FROM events').fetchone()[0]
        rows=observer.execute("SELECT * FROM events WHERE id<=? AND actor='orchestrator' AND kind IN (?,?,?,?) AND json_extract(payload,'$.run_id')=? ORDER BY id LIMIT 5001",
                             (cutoff,*KINDS,run.name)).fetchall()
        if len(rows)>5000:raise ValueError('Observer snapshot exceeds registered row budget')
        events=[dict(row,payload=json.loads(row['payload'])) for row in rows]
    finally:observer.close()
    starts=[e for e in events if e['kind']=='shared_endurance.started']
    if len(starts)!=1:raise ValueError('Expected exactly one owned started receipt')
    started=starts[0]
    beats=[e['payload'] for e in events if e['kind']=='shared_endurance.heartbeat']
    cycles=[e['payload'] for e in events if e['kind']=='shared_endurance.measured_video']
    finals=[e['payload'] for e in events if e['kind']=='shared_endurance.completed']
    video=video_coverage(cycles)
    resources=resource_coverage(beats,gateway_identity())
    directory=Path(manifest['private_data']).resolve()
    if directory!=ROOT/'local'/run.name:raise ValueError('Unexpected private data ownership')
    private=private_snapshot(directory)
    native=native_identity(started['payload'],started['created'])
    terminal=(run/'result.json').is_file()
    final=json.loads((run/'result.json').read_text(encoding='utf-8')) if terminal else None
    complete_streams=[s for s in private['streams'] if s['status']=='complete']
    unfinished=[s['id'] for s in private['streams'] if s['status']!='complete']
    critical=private['event_counts'].get('cognitive_quantum_failed',0)
    direct_gate=False;agreement=False;cleanup=False
    if terminal:
        direct_gate=original_long_gate(elapsed=final['elapsed_seconds'],cycles=len(final['video_cycles']),
            checkpoints=len(final['checkpoints']),updates=final['final_neural']['updates'],
            replay_seen=final['final_neural']['replay_seen'],device=final['final_neural']['device'],
            critical_failures=critical,active_capture=bool(unfinished))
        agreement=bool(len(cycles)==len(final['video_cycles'])==len(complete_streams)
            and {r['stream']['id'] for r in cycles}=={r['stream']['id'] for r in final['video_cycles']}
            and len(final['checkpoints'])==len(private['pinned_checkpoint_ids'])
            and final['critical_quantum_failures']==critical
            and final['final_neural']['updates']==private['active_checkpoint']['updates']
            and len(finals)==1 and finals[0]['verdict']==final['verdict'])
        cleanup=not final['failures'] and not unfinished
        direct_gate &= final.get('gate_pass') is True and native['validated_current_identity_matches']
    decision=final_audit_decision(terminal_receipt=terminal,worker_live=state['actual_process_live'],
        registered_gate=bool(direct_gate),source_archive_valid=bool(archive_valid),preflight_valid=preflight_valid,
        sampled_reserve_valid=resources['sampled_512MiB_reserve_met'],
        receipt_trace_private_agree=agreement,cleanup_valid=bool(cleanup))
    sample=ROOT/'local/hybrid-v1/samples/engineering-vi-video.mkv'
    sample_valid=sha(sample.read_bytes())==manifest['sample_sha256']
    if not sample_valid:raise ValueError('Original owned clip hash changed')
    name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-endurance-data-audit-v1-'+uuid.uuid4().hex[:6]
    output=ROOT/'runs'/name;output.mkdir()
    event_bytes=encode(events).encode()
    if len(event_bytes)>20_000_000:raise ValueError('Observer snapshot byte budget exceeded')
    (output/'observer-events.json').write_bytes(event_bytes)
    (output/'private-snapshot.json').write_text(encode(private),encoding='utf-8')
    (output/'parent-manifest.json').write_bytes((run/'manifest.json').read_bytes())
    report={'run_id':name,'parent_endurance_run':run.name,'verdict':decision,'process_snapshot':state,
        'original_source_archive_matches':bool(archive_valid),'preflight_source_and_result_valid':preflight_valid,
        'observer_cutoff_event_id':cutoff,'observer_snapshot_sha256':sha(event_bytes),
        'video':video,'resources':resources,'private_summary':{
            k:v for k,v in private.items() if k not in ('streams','checkpoints')},
        'current_native_identity':native,'independent_registered_gate':bool(direct_gate),
        'terminal_receipt_trace_private_agree':agreement if terminal else None,
        'cross_database_snapshots_not_atomic':True,'no_new_inference_training_restart_or_source_mutation':True,
        'P1_P2_P3_or_lifelong_quality_PASS':False,'seconds':time.monotonic()-began,
        'scope':'Engineering read-only audit; live counts may straddle a boundary. Fixed short-source repetitions are not continuous diverse-video or skill retention proof.'}
    sources=[Path(__file__),ROOT/'src/scarx/evaluation/endurance_independent_audit.py',
             ROOT/'scripts/monitor_hybrid_endurance.py',ROOT/'experiments/hybrid-endurance-independent-audit-v1.md']
    hashes={}
    for path in sources:
        rel=path.relative_to(ROOT);dest=output/'source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=sha(path.read_bytes())
    (output/'manifest.json').write_text(encode({'source_sha256':hashes,'parent_endurance_run':run.name,
        'observer_cutoff':cutoff,'posthoc_data_audit':True,'new_neural_work':False}),encoding='utf-8')
    if time.monotonic()-began>60:raise TimeoutError('Audit exceeded registered60seconds')
    (output/'result.json').write_text(encode(report),encoding='utf-8')
    trace=Store(ROOT/'local/hybrid-v1')
    trace.event('endurance_data_audit.snapshot',{
        k:v for k,v in report.items() if k not in ('private_summary',)} ,'orchestrator')
    trace.close()
    print(encode({k:report[k] for k in ('run_id','parent_endurance_run','verdict','seconds')}))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    main(p.parse_args().run)
