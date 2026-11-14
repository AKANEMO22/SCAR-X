"""Audit completed component measurements, never run another ASR/model call."""

import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from scarx.hybrid.store import encode,Store
from scarx.evaluation.asr_soak_audit import measured_gate
from monitor_hybrid_asr_soak import inspect


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(2<<20):digest.update(block)
    return digest.hexdigest()


def main(run):
    started=time.monotonic();run=run.resolve()
    if not run.is_relative_to(ROOT/'runs'):raise ValueError('Expected owned candidate run')
    current=inspect(run)
    if current['actual_process_live'] or current['final_verdict'] is None:
        raise RuntimeError('Final data audit requires real worker terminal plus receipt; do not restart it')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    result=json.loads((run/'result.json').read_text(encoding='utf-8'))
    if (run/'independent-soak-audit-v1.json').exists():raise FileExistsError('Original data audit already exists; do not overwrite')
    for rel,expected in manifest['source_sha256'].items():
        assert sha(run/'source'/rel)==expected
    parent=ROOT/'runs/20261114T015546Z-asr-candidate-benchmark-v1-68e3c7'
    original=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    corpuspath=ROOT/'runs'/original['data_run']/'manifest.json'
    assert sha(corpuspath)==manifest['data_sha256']==original['data_sha256']
    corpus=json.loads(corpuspath.read_text(encoding='utf-8'));sources={row['id']:row for row in corpus['samples']}
    assert len(sources)==32
    assert sha(Path(original['model_paths']['small']))==manifest['model_sha256']==original['model_sha256']['small']
    exe=ROOT/'local/hybrid-v1/runtime/whisper.cpp-b5454-cuda12.4/Release/whisper-cli.exe'
    assert sha(exe)==manifest['exe_sha256']==original['executable_sha256']
    for item in sources.values():assert sha(Path(item['audio']))==item['audio_sha256']
    attempt_ids=set();transcripts=defaultdict(set);percall=[]
    for index,call in enumerate(result['calls']):
        attempt=call['attempt'];assert type(attempt) is int and 1<=attempt<=256 and attempt not in attempt_ids
        attempt_ids.add(attempt)
        row=sources[call['source_id']]
        assert row['language']==call['language'] and row['duration_seconds']==call['audio_seconds']
        text=' '.join(chunk.get('text','').strip() for chunk in call['raw']['transcription']).strip()
        assert text==call['actual_transcript']
        assert hashlib.sha256(text.encode()).hexdigest()==call['transcript_sha256']
        stored=json.loads((run/f'call-{attempt:03d}.json').read_text(encoding='utf-8'))
        assert stored==call
        log=(run/f'call-{attempt:03d}.log').read_text(encoding='utf-8',errors='replace')
        assert 'using CUDA0 backend' in log and 'ftype         = 0' in log
        assert call['native_CUDA_FP32_verified'] is True and call['new_independent_quality_sample'] is False
        transcripts[call['source_id']].add(text)
        percall.append({'attempt':attempt,'source_id':call['source_id'],'seconds':call['seconds'],
            'audio_seconds':call['audio_seconds'],'RTF':call['seconds']/call['audio_seconds']})
    preempted={row['attempt'] for row in result['preemptions']}
    assert len(preempted)==len(result['preemptions']) and not preempted&attempt_ids
    combined=sorted(attempt_ids|preempted)
    if combined:
        assert combined==list(range(min(combined),max(combined)+1))
        order={attempt:i for i,attempt in enumerate(combined)}
        for call in result['calls']:
            assert call['source_id']==corpus['samples'][order[call['attempt']]%32]['id']
    metrics=measured_gate(manifest['seconds'],result['elapsed_seconds'],result['calls'],
                          result['global_free_samples'],result['failures'])
    if not result['failures']:
        assert metrics['gate']==result['gate_pass']
        assert metrics['completed_calls']==result['complete_calls']
        assert abs(metrics['processed_audio_seconds']-result['completed_audio_seconds'])<1e-8
        assert abs(metrics['RTF_p95']-result['RTF_p95'])<1e-10
    database=(ROOT/'local/hybrid-v1/hybrid.sqlite').resolve()
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('BEGIN')
        events=[{'id':ident,'kind':kind,'payload':json.loads(raw)} for ident,kind,raw in db.execute(
            "SELECT id,kind,payload FROM events WHERE actor='orchestrator' AND kind IN ('ASR_small_soak.heartbeat','ASR_small_soak.actual_transcript','ASR_small_soak.completed') AND json_extract(payload,'$.run_id')=? ORDER BY id LIMIT 5001",(run.name,))]
    assert len(events)<=5000
    encoded=encode(events).encode();assert len(encoded)<=20_000_000
    calls=[event['payload'] for event in events if event['kind']=='ASR_small_soak.actual_transcript']
    beats=[event['payload'] for event in events if event['kind']=='ASR_small_soak.heartbeat']
    finals=[event['payload'] for event in events if event['kind']=='ASR_small_soak.completed']
    assert len(calls)==len(result['calls']) and len(finals)==1
    assert finals[0]['verdict']==result['verdict']
    for a,b in zip(calls,result['calls']):
        assert a=={'run_id':run.name,**b}
    for beat in beats:
        resources=beat['resources']
        roots=set(resources['explicit_root_pids'])|set(resources['discovered_hermes_roots'])
        assert roots==set(resources['included_client_pids'])
        assert manifest['original_native_pid'] in {p['pid'] for p in resources['processes']}
        assert resources['limit_bytes'] is None
    resource=result['resources']
    budget=json.loads((ROOT/'local/whisper-small-paced-soak-v1-budget.json').read_text(encoding='utf-8'))
    assert type(budget['attempted']) is int and budget['attempted']<=256
    assert not attempt_ids or max(attempt_ids)<=budget['attempted']
    if manifest['preflight']:
        prior=ROOT/'runs'/manifest['preflight'];pm=json.loads((prior/'manifest.json').read_text(encoding='utf-8'))
        pr=json.loads((prior/'result.json').read_text(encoding='utf-8'))
        assert pm['source_sha256']==manifest['source_sha256'] and pm['seconds']==30 and pr['gate_pass'] and not pr['failures']
    requests=len(result['calls'])
    med=lambda rows:float(np.median([r['seconds'] for r in rows])) if rows else None
    report={'run_id':run.name,'checked_UTC':datetime.now(timezone.utc).isoformat(),
        'actual_worker_terminal_verified':True,'source_model_exe_and_corpus_hashes_match':True,
        'actual_CUDA_FP32_logs_and_transcripts_rederived':True,'mechanical_metrics':metrics,
        'unique_audio_source_ids':len(transcripts),'unique_audio_seconds':sum(sources[key]['duration_seconds'] for key in transcripts),
        'repeated_audio_does_not_add_quality_samples':True,
        'actual_transcript_variants_per_source':{key:len(values) for key,values in transcripts.items()},
        'first20_call_latency_median_seconds':med(result['calls'][:20]),
        'last20_call_latency_median_seconds':med(result['calls'][-20:]),
        'observer_heartbeat_count':len(beats),'observer_transcript_events_match_complete_calls':True,
        'budget_current_attempts_including_failed_preflight':budget['attempted'],
        'component_attempts_in_this_run':len(combined),'preemptions':len(preempted),
        'peak_group_RSS_bytes':resource['peak_rss_bytes'],'peak_group_private_commit_bytes':resource['peak_private_commit_bytes'],
        'peak_group_dedicated_GPU_bytes':resource['peak_gpu_dedicated_bytes'],
        'recorded_root_union_consistent':True,'current_source_still_matches_run':current['frozen_source_matches'],
        'sampled_reserve_is_not_unobserved_peak_guarantee':True,
        'native_identity_evidence':'Primary runner SharedNative.check eachcall; manifest pins39372/create-time. This audit reads archived evidence; no new nativegeneration.',
        'candidate_live_promotion':False,'new_inference_or_learning':False,
        'quality_video_or_lifelong_PASS':False,
        'verdict':'COMPONENT_ENGINEERING_SOAK_DATA_VERIFIED_NOT_LIVE_PROMOTION' if metrics['gate'] else 'COMPONENT_DATA_AUDIT_NOT_QUALIFIED',
        'scope':'RepeatedownedFLEURSDEV, mixedGPUload with original8hjob and interactiveHermes. No new independentqualitycases or fullproductionclaim.'}
    with (run/'observer-audit-snapshot-v1.json').open('xb') as f:f.write(encoded)
    supplement=run/'audit-source-v1';hashes={}
    for path in (Path(__file__),ROOT/'src/scarx/evaluation/asr_soak_audit.py',
        ROOT/'scripts/monitor_hybrid_asr_soak.py',ROOT/'experiments/hybrid-whisper-soak-data-audit-v1.md'):
        rel=path.relative_to(ROOT);dest=supplement/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=sha(path)
    report['audit_source_sha256']=hashes;report['seconds']=time.monotonic()-started
    assert report['seconds']<=30
    with (run/'independent-soak-audit-v1.json').open('x',encoding='utf-8') as f:f.write(encode(report))
    trace=Store(ROOT/'local/hybrid-v1')
    trace.event('ASR_small_soak.independent_audit',{k:v for k,v in report.items() if k!='actual_transcript_variants_per_source'},'orchestrator');trace.close()
    print(encode({k:v for k,v in report.items() if k not in ('audit_source_sha256','actual_transcript_variants_per_source')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();main(a.run)
