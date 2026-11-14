"""Paced native originalFP32 ASR candidate; share VLM, never deploy or train."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from scarx.evaluation.shared_native import SharedNative
from scarx.evaluation.asr_soak_budget import charge_attempt
from scarx.hybrid.device_budget import gpu_headroom
from scarx.hybrid.store import Store,encode
from monitor_hybrid_endurance import inspect


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while part:=f.read(2<<20):h.update(part)
    return h.hexdigest()


def main(seconds,preflight):
    if seconds==1800 and preflight is None:raise ValueError('30min requires completed same-source30s preflight')
    name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-whisper-small-soak-v1-'+uuid.uuid4().hex[:6]
    run,data=ROOT/'runs'/name,ROOT/'local'/name;run.mkdir();data.mkdir()
    parent=ROOT/'runs/20261114T015546Z-asr-candidate-benchmark-v1-68e3c7'
    benchmark=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    data_path=ROOT/'runs'/benchmark['data_run']/'manifest.json'
    corpus=json.loads(data_path.read_text(encoding='utf-8'))
    model=Path(benchmark['model_paths']['small'])
    exe=ROOT/'local/hybrid-v1/runtime/whisper.cpp-b5454-cuda12.4/Release/whisper-cli.exe'
    if sha(data_path)!=benchmark['data_sha256'] or len(corpus['samples'])!=32:raise ValueError('ASRcorpus identity changed')
    if sha(model)!=benchmark['model_sha256']['small'] or sha(exe)!=benchmark['executable_sha256']:raise ValueError('ASRnative identity changed')
    for item in corpus['samples']:
        if sha(Path(item['audio']))!=item['audio_sha256']:raise ValueError('Owned ASR input bytes changed')
    prototype=ROOT/'runs/20261114T074607Z-client-roots-p0-v1-0c0835'
    overlay_raw=(prototype/'resources-candidate.py.source').read_bytes()
    if hashlib.sha256(overlay_raw).hexdigest()!='8774db6c6f91adb8bcb2f990341aa086eebe88f57b047521837603829e0af2c8':raise ValueError('Qualified overlay changed')
    overlay=data/'resources_overlay.py';overlay.write_bytes(overlay_raw)
    spec=importlib.util.spec_from_file_location('scarx.hybrid.resources_ASR_soak_overlay',overlay)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    paths=list((ROOT/'src/scarx/hybrid').glob('*.py'))+[
        Path(__file__),ROOT/'src/scarx/evaluation/shared_native.py',ROOT/'scripts/monitor_hybrid_endurance.py',
        ROOT/'src/scarx/evaluation/asr_soak_budget.py',
        ROOT/'experiments/hybrid-whisper-small-paced-soak-v1.md']
    hashes={}
    for path in paths:
        rel=path.relative_to(ROOT);dest=run/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=sha(path)
    if preflight:
        prior=json.loads((preflight/'result.json').read_text(encoding='utf-8'))
        old=json.loads((preflight/'manifest.json').read_text(encoding='utf-8'))
        if not prior['gate_pass'] or old['seconds']!=30 or old['source_sha256']!=hashes:raise ValueError('Missing/stale/failed preflight')
    native=SharedNative(ROOT);native.check()
    if gpu_headroom()['free_MiB']<1800:raise MemoryError('Candidate launch preflight requires1800MiB freeVRAM')
    roots={native.owner.pid}
    base_run=ROOT/'runs/20261114T011841Z-shared-endurance-v1-554c19'
    other=inspect(base_run)
    if other['actual_process_live']:roots.add(other['pid'])
    monitor=module.ResourceMonitor(limit=None,extra_pids=roots,include_hermes=True)
    monitor.start();trace=Store(ROOT/'local/hybrid-v1')
    manifest={'run_id':name,'pid':os.getpid(),'process_created_epoch':psutil.Process().create_time(),
        'seconds':seconds,'source_sha256':hashes,'model_sha256':sha(model),'exe_sha256':sha(exe),
        'data_sha256':sha(data_path),'original_native_pid':native.worker.pid,'original_native_created':native.worker_created,
        'source_samples':len(corpus['samples']),'aggregate_attempt_cap':256,'preflight':preflight.name if preflight else None,
        'external_base_endurance':other,'overlay_only_not_installed':True,'new_training_or_live_ASR_change':False}
    (run/'manifest.json').write_text(encode(manifest),encoding='utf-8')
    (ROOT/'local/current-whisper-small-soak-run.txt').write_text(name,encoding='utf-8')
    report={'run_id':name,'calls':[],'preemptions':[],'failures':[],'gate_pass':False,
        'candidate_promoted':False,'new_training_or_live_ASR_change':False,'global_free_samples':[]}
    process=None;started=time.monotonic();next_start=next_heartbeat=0.;index=0

    def validate():
        native.check()
        for rel,expected in hashes.items():
            if sha(ROOT/rel)!=expected:raise RuntimeError('FrozenASRsoak source changed: '+rel)

    def sample(active=False):
        row={'epoch':time.time(),'ASR_active':active,**gpu_headroom()}
        report['global_free_samples'].append(row)
        if row['free_MiB']<512:raise MemoryError('Sampled physical GPU reserve below512MiB')
        return row

    try:
        while time.monotonic()-started<seconds:
            elapsed=time.monotonic()-started
            if elapsed>=next_heartbeat:
                validate();headroom=sample();resource=monitor.sample()
                beat={'run_id':name,'pid':os.getpid(),'elapsed_seconds':elapsed,'requested_seconds':seconds,
                    'complete_calls':len(report['calls']),'preemptions':len(report['preemptions']),
                    'completed_audio_seconds':sum(row['audio_seconds'] for row in report['calls']),
                    'resources':resource,'GPU_headroom':headroom,'live_ASR_changed':False,'state':'running'}
                (run/'heartbeat.json').write_text(encode(beat),encoding='utf-8')
                trace.event('ASR_small_soak.heartbeat',beat,'orchestrator')
                next_heartbeat=elapsed+15
            if elapsed<next_start:
                time.sleep(min(.2,next_start-elapsed));continue
            if native.status().get('cognition',{}).get('phase')=='serving':
                time.sleep(.2);continue
            item=corpus['samples'][index%32];index+=1
            attempt=charge_attempt(ROOT/'local/whisper-small-paced-soak-v1-budget.json')
            output=data/f'call-{attempt:03d}';logpath=run/f'call-{attempt:03d}.log'
            command=[str(exe),'-m',str(model),'-f',item['audio'],'-l','auto','-t','2','-dev','0','-ojf','-of',str(output)]
            call_started=time.monotonic();preempted=False
            with logpath.open('wb') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                while process.poll() is None:
                    native.check();sample(active=True)
                    if time.monotonic()-call_started>120:raise TimeoutError('Owned ASR call exceeded120s')
                    if native.status().get('cognition',{}).get('phase')=='serving':
                        process.terminate();process.wait(10);preempted=True;break
                    time.sleep(.25)
            duration=time.monotonic()-call_started
            if preempted:
                report['preemptions'].append({'attempt':attempt,'source_id':item['id'],'seconds':duration})
            else:
                if process.returncode:raise RuntimeError('Owned ASRfailed; see '+logpath.name)
                logtext=logpath.read_text(encoding='utf-8',errors='replace')
                if 'using CUDA0 backend' not in logtext or 'ftype         = 0' not in logtext:raise RuntimeError('Expected actualCUDA0/nativeFP32 log')
                path=output.with_suffix('.json')
                if path.stat().st_size>1_000_000:raise ValueError('ASRoutput exceeds bound')
                raw=json.loads(path.read_text(encoding='utf-8'))
                text=' '.join(row.get('text','').strip() for row in raw['transcription']).strip()
                row={'attempt':attempt,'source_id':item['id'],'language':item['language'],
                    'audio_seconds':item['duration_seconds'],'seconds':duration,'RTF':duration/item['duration_seconds'],
                    'actual_transcript':text,'transcript_sha256':hashlib.sha256(text.encode()).hexdigest(),
                    'raw':raw,'native_CUDA_FP32_verified':True,'new_independent_quality_sample':False}
                report['calls'].append(row)
                trace.event('ASR_small_soak.actual_transcript',{'run_id':name,**row},'orchestrator')
                (run/f'call-{attempt:03d}.json').write_text(encode(row),encoding='utf-8')
            process=None;next_start=max(next_start+item['duration_seconds'],time.monotonic()-started)
        validate();sample()
        elapsed=time.monotonic()-started;calls=report['calls']
        audio=sum(row['audio_seconds'] for row in calls)
        p95=float(np.percentile([row['RTF'] for row in calls],95)) if calls else None
        report.update(elapsed_seconds=elapsed,completed_audio_seconds=audio,RTF_p95=p95,
            complete_calls=len(calls),active_free_samples=sum(row['ASR_active'] for row in report['global_free_samples']),
            min_sampled_global_free_MiB=min(row['free_MiB'] for row in report['global_free_samples']))
        report['gate_pass']=bool(elapsed>=seconds and len(calls)>=(1 if seconds==30 else 80)
            and (seconds==30 or audio>=1000 and p95<=1) and report['min_sampled_global_free_MiB']>=512)
        report['verdict']='COMPONENT_ENGINEERING_SOAK_ONLY_NOT_LIVE_PROMOTION' if report['gate_pass'] else 'INSUFFICIENT_COMPONENT_COVERAGE'
    except BaseException:
        report['failures'].append(traceback.format_exc());report['verdict']='ASR_COMPONENT_INTERRUPTED_OR_FAILED'
    finally:
        if process is not None and process.poll() is None:
            process.terminate();process.wait(10)
        report['elapsed_seconds']=time.monotonic()-started;report['resources']=monitor.close()
        if report['failures']:report['gate_pass']=False
        with (run/'result.json').open('x',encoding='utf-8') as f:f.write(encode(report))
        trace.event('ASR_small_soak.completed',{k:v for k,v in report.items() if k not in ('calls','global_free_samples','resources')},'orchestrator');trace.close()
        print(encode({k:report[k] for k in ('run_id','verdict','gate_pass','failures','elapsed_seconds')}),flush=True)
    return report['gate_pass']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,choices=(30,1800),required=True);p.add_argument('--preflight',type=Path)
    a=p.parse_args();raise SystemExit(0 if main(a.seconds,a.preflight.resolve() if a.preflight else None) else 1)
