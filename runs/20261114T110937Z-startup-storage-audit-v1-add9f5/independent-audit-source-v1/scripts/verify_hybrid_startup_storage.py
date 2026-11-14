"""Recompute event intervals and instrumented byte accounting without rehashing models."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.store import Store,encode
from scarx.runtime.asr_profiles import PROFILES


def main(run):
    run=run.resolve();assert run.is_relative_to(ROOT/'runs')
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    manifest,result,events,expected=map(read,('manifest.json','result.json','actual-startup-timestamps.json','expected-hash-bytes.json'))
    assert not result['failures'] and result['neural_or_native_model_calls']==0 and len(result['stages'])==3
    for rel,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((run/'source'/rel).read_bytes()).hexdigest()==digest
    times={name:next(row['created'] for row in events if row['kind']==name) for name in
           ('native_storage_verified','neural_execution_device','worker_started','search_service_ready','gateway_native_gate')}
    assert [row['id'] for row in events]==sorted({row['id'] for row in events})
    assert len([row for row in events if row['kind']=='native_parity_case'])==3
    assert next(row['pid'] for row in events if row['kind']=='worker_started')==manifest['actual_worker_pid']
    recomputed={
        'process_start_to_storage_event_includes_imports_restore_and_hashes':times['native_storage_verified']-manifest['owner_created'],
        'storage_event_to_ALIF_device_event':times['neural_execution_device']-times['native_storage_verified'],
        'ALIF_device_event_to_native_process_spawn_includes_ASR_recheck':times['worker_started']-times['neural_execution_device'],
        'native_spawn_to_search_ready_includes_VLM_health_and_search':times['search_service_ready']-times['worker_started'],
        'search_ready_through_parity_to_nativegate':times['gateway_native_gate']-times['search_service_ready'],
        'process_start_to_nativegate_ready_boundary_proxy':times['gateway_native_gate']-manifest['owner_created']}
    assert recomputed==result['timeline_seconds'] and all(value>=0 for value in recomputed.values())
    profile=PROFILES['openai/whisper-small']
    sizes=[sum(expected['baseline_exports'].values())+profile.native_bytes,profile.native_bytes,0]
    assert expected['stage_expected_bytes']==sizes
    for stage,size,objects in zip(result['stages'],sizes,(4,1,0)):
        assert stage['actual_SHA256_update_bytes']==stage['expected_bytes']==size
        assert stage['actual_SHA256_objects']==objects
        assert 0<=stage['seconds']<=30 and stage['process_CPU_seconds']>=0
    assert result['seconds']<=30 and sum(s['seconds'] for s in result['stages'])<=result['seconds']
    assert result['stages'][0]['actual_receipt']['selected_ASR_storage']==result['stages'][1]['actual_receipt']==result['stages'][2]['actual_receipt']
    activation=ROOT/'runs/20261114T090807Z-ASR-native-integration-p0-v1-57a681/live-activation-result.json'
    assert hashlib.sha256(activation.read_bytes()).hexdigest()==manifest['activation_receipt_sha256']
    report={'run_id':run.name,'verdict':'EXISTING_STARTUP_TIMELINE_AND_HASH_ACCOUNTING_VERIFIED_NOT_SPEEDUP_PASS',
        'timeline_seconds':recomputed,'total_SHA256_update_bytes':sum(sizes),
        'duplicate_selected_ASR_bytes':profile.native_bytes,
        'fresh_selected_ASR_hash_seconds_uncontrolled_cache':result['stages'][1]['seconds'],
        'cachehit_SHA256_bytes':result['stages'][2]['actual_SHA256_update_bytes'],
        'additional_neural_or_model_hash_calls_in_audit':0,
        'no_cold_cache_or_repeated_startup_measurement':True,
        'no_worker_restart_or_optimization_applied':True}
    with (run/'independent-data-audit-v1.json').open('x',encoding='utf-8') as out:out.write(encode(report))
    store=Store(ROOT/'local/hybrid-v1')
    try:store.event('startup_storage_audit.data_verified',report,'orchestrator')
    finally:store.close()
    print(encode(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);main(parser.parse_args().run)
