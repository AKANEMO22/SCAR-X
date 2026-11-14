"""Actual startup events and readonly hash calls, never another model restart."""

import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
import traceback
import uuid
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.store import Store,encode
from scarx.hybrid.backends import NativeBackends
from scarx.hybrid.config import load_runtime_config
from scarx.hybrid.qualification import verify_native_storage
from scarx.runtime.asr_profiles import PROFILES
from scarx.evaluation.shared_native import SharedNative


def main():
    start=time.monotonic()
    native=SharedNative(ROOT);native.check()
    activation=ROOT/'runs/20261114T090807Z-ASR-native-integration-p0-v1-57a681'
    receipt=json.loads((activation/'live-activation-result.json').read_text(encoding='utf-8'))
    assert receipt['new_gateway_pid']==native.owner.pid and receipt['new_native_worker_pid']==native.worker.pid
    directory=ROOT/'local/hybrid-v1'
    config=load_runtime_config(directory)
    assert config.asr_model_name=='openai/whisper-small'
    run=ROOT/'runs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-startup-storage-audit-v1-'+uuid.uuid4().hex[:6])
    run.mkdir()
    paths=[Path(__file__),ROOT/'src/scarx/hybrid/backends.py',ROOT/'src/scarx/hybrid/gateway.py',
           ROOT/'src/scarx/hybrid/qualification.py',ROOT/'src/scarx/runtime/asr_profiles.py',
           ROOT/'experiments/hybrid-startup-storage-audit-v1.md']
    hashes={}
    for path in paths:
        rel=path.relative_to(ROOT);dest=run/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    (run/'manifest.json').write_text(encode({'run_id':run.name,'source_sha256':hashes,'actual_owner_pid':native.owner.pid,
        'actual_worker_pid':native.worker.pid,'owner_created':native.owner_created,'worker_created':native.worker_created,
        'activation_receipt_sha256':hashlib.sha256((activation/'live-activation-result.json').read_bytes()).hexdigest(),
        'new_neural_calls':0,'timed_hash_stages':3,'OS_cache_uncontrolled':True}),encoding='utf-8')
    with sqlite3.connect((directory/'hybrid.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        cutoff=db.execute("SELECT MIN(id) FROM events WHERE kind='gateway_native_gate' AND id>?",(receipt['startup_event_cursor'],)).fetchone()[0]
        events=[{'id':i,'created':created,'kind':kind,'pid':json.loads(raw).get('pid')}
                for i,created,kind,raw in db.execute("SELECT id,created,kind,payload FROM events WHERE id>? AND id<=? AND kind IN ('native_storage_verified','neural_execution_device','worker_started','search_service_ready','native_parity_case','gateway_native_gate') ORDER BY id",(receipt['startup_event_cursor'],cutoff))]
    assert events and all(a['created']<=b['created'] for a,b in zip(events,events[1:]))
    (run/'actual-startup-timestamps.json').write_text(encode(events),encoding='utf-8')
    storage=json.loads((directory/'native-integrity.json').read_text(encoding='utf-8'))
    basebytes=sum(e['bytes'] for e in storage['exports'].values())
    small=PROFILES['openai/whisper-small']
    (run/'expected-hash-bytes.json').write_text(encode({'baseline_exports':{name:e['bytes'] for name,e in storage['exports'].items()},
        'selected_small_bytes':small.native_bytes,'stage_expected_bytes':[basebytes+small.native_bytes,small.native_bytes,0]}),encoding='utf-8')
    report={'run_id':run.name,'stages':[],'failures':[],'neural_or_native_model_calls':0,'live_source_config_or_workers_changed':False,
            'own_process_RSS_before':psutil.Process().memory_info().rss,'timing_scope':'One retrospective startup plus uncontrolled-cache readonly hashing; no speedup claim'}
    realsha=hashlib.sha256
    measured={'bytes':0,'objects':0}
    class Counted:
        def __init__(self,*args,**kwargs):
            self.actual=realsha(*args,**kwargs);measured['objects']+=1
            if args:measured['bytes']+=len(args[0])
        def update(self,block):
            if time.monotonic()-start>30:raise TimeoutError('Hash audit30second bound')
            measured['bytes']+=len(block);self.actual.update(block)
        def hexdigest(self):return self.actual.hexdigest()
        def digest(self):return self.actual.digest()
    def forbidden(*args,**kwargs):raise AssertionError('Readonly hash audit cannot invoke a worker/store action')
    backend=NativeBackends(directory,SimpleNamespace(event=forbidden),SimpleNamespace(check=forbidden),config)
    try:
        operations=[('qualification_full_storage',lambda:verify_native_storage(directory,config)),
                    ('fresh_backend_selected_ASR',backend.ensure_asr_storage),
                    ('same_backend_ASR_cachehit',backend.ensure_asr_storage)]
        for (name,operation),expected in zip(operations,(basebytes+small.native_bytes,small.native_bytes,0)):
            measured['bytes']=measured['objects']=0
            cpu0=time.process_time();wall0=time.monotonic()
            try:
                hashlib.sha256=Counted
                actual=operation()
            finally:hashlib.sha256=realsha
            result={'stage':name,'seconds':time.monotonic()-wall0,'process_CPU_seconds':time.process_time()-cpu0,
                    'actual_SHA256_update_bytes':measured['bytes'],'actual_SHA256_objects':measured['objects'],
                    'expected_bytes':expected,'actual_receipt':actual if isinstance(actual,dict) else backend.asr_storage_receipt}
            report['stages'].append(result)
            assert measured['bytes']==expected
            assert backend.vlm is None and backend.asr is None
        at={name:next(e['created'] for e in events if e['kind']==name) for name in
            ('native_storage_verified','neural_execution_device','worker_started','search_service_ready','gateway_native_gate')}
        report['timeline_seconds']={
            'process_start_to_storage_event_includes_imports_restore_and_hashes':at['native_storage_verified']-native.owner_created,
            'storage_event_to_ALIF_device_event':at['neural_execution_device']-at['native_storage_verified'],
            'ALIF_device_event_to_native_process_spawn_includes_ASR_recheck':at['worker_started']-at['neural_execution_device'],
            'native_spawn_to_search_ready_includes_VLM_health_and_search':at['search_service_ready']-at['worker_started'],
            'search_ready_through_parity_to_nativegate':at['gateway_native_gate']-at['search_service_ready'],
            'process_start_to_nativegate_ready_boundary_proxy':at['gateway_native_gate']-native.owner_created}
        assert len([e for e in events if e['kind']=='native_parity_case'])==3
        assert all(realsha((ROOT/rel).read_bytes()).hexdigest()==digest for rel,digest in hashes.items())
        native.check();report['verdict']='STARTUP_EVENTS_AND_DUPLICATE_HASH_READS_MEASURED_NOT_NEW_COLDSTART_BENCHMARK'
    except Exception:
        report['failures'].append(traceback.format_exc());report['verdict']='STARTUP_STORAGE_AUDIT_INCOMPLETE'
    finally:
        hashlib.sha256=realsha
        report['seconds']=time.monotonic()-start
        report['own_process_RSS_after']=psutil.Process().memory_info().rss
        with (run/'result.json').open('x',encoding='utf-8') as out:out.write(encode(report))
        store=Store(directory)
        try:store.event('startup_storage_audit.completed',{k:v for k,v in report.items() if k!='stages'},'orchestrator')
        finally:store.close()
    print(encode({k:report.get(k) for k in ('run_id','verdict','failures','seconds','timeline_seconds')}))


if __name__=='__main__':main()
