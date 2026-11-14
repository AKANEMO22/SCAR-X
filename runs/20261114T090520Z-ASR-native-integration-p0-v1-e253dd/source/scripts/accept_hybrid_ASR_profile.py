"""Prepare or execute four real private selected-ASR calls after frozen jobs end."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from scarx.hybrid.store import Store,encode
from scarx.hybrid.device_budget import gpu_headroom
from scarx.evaluation.shared_native import SharedNative
from scarx.runtime.asr_profiles import PROFILES
from scarx.evaluation.integration_call_budget import charge as charge_budget
from monitor_hybrid_endurance import inspect as inspect_endurance
from monitor_hybrid_asr_soak import inspect as inspect_soak

ENDURANCE='20261114T011841Z-shared-endurance-v1-554c19'
SOAK='20261114T080220Z-whisper-small-soak-v1-292b32'
PREPARATION=ROOT/'runs/20261114T084052Z-ASR-profile-resolver-p0-v1-791db8/shadow-preparation-v1'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(2<<20):digest.update(block)
    return digest.hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def prerequisites():
    long=inspect_endurance(ROOT/'runs'/ENDURANCE)
    soak=inspect_soak(ROOT/'runs'/SOAK)
    if long['actual_process_live'] or long['final_verdict'] is None:
        raise RuntimeError('Original8h is still live/incomplete; keep its source and VLM untouched')
    if soak['actual_process_live'] or soak['final_verdict'] is None:
        raise RuntimeError('ASRcomponent job is still live/incomplete')
    audits=[]
    for folder in (ROOT/'runs').glob('*-endurance-data-audit-v1-*'):
        path=folder/'result.json'
        if not path.is_file():continue
        value=json.loads(path.read_text(encoding='utf-8'))
        if value['parent_endurance_run']==ENDURANCE and value['verdict']=='EIGHT_HOUR_ENGINEERING_DATA_AUDIT_PASS_ONLY':
            audits.append(path)
    if not audits:raise RuntimeError('Missing independent completed engineering audit; no source unlock or nativecalls')
    component=ROOT/'runs'/SOAK/'independent-soak-audit-v1.json'
    value=json.loads(component.read_text(encoding='utf-8'))
    if value['verdict']!='COMPONENT_ENGINEERING_SOAK_DATA_VERIFIED_NOT_LIVE_PROMOTION':
        raise RuntimeError('ASRcomponent data audit not qualified')
    return {'endurance_final_audit':str(sorted(audits)[-1].relative_to(ROOT)),
            'ASR_component_audit':str(component.relative_to(ROOT))}


def prepare():
    name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-ASR-native-integration-p0-v1-'+uuid.uuid4().hex[:6]
    run=ROOT/'runs'/name;run.mkdir()
    corpuspath=ROOT/'runs/20261114T015303Z-asr-dev-data-v1-ddf1fd/manifest.json'
    corpus=json.loads(corpuspath.read_text(encoding='utf-8'))
    selected={lang:next(row for row in corpus['samples'] if row['language']==lang) for lang in ('vi','en')}
    order=[{'model':model,'sample':selected[lang]} for model,lang in [('base','vi'),('small','vi'),('small','en'),('base','en')]]
    hashes={}
    for path in [Path(__file__),ROOT/'src/scarx/runtime/asr_profiles.py',
        ROOT/'experiments/hybrid-ASR-profile-native-integration-p0-v1.md',
        ROOT/'integrations/hermes/asr_profile_integration_patch.py',
        ROOT/'src/scarx/evaluation/shared_native.py',ROOT/'src/scarx/hybrid/engine.py',
        ROOT/'src/scarx/hybrid/learner.py',ROOT/'src/scarx/hybrid/config.py',
        ROOT/'src/scarx/evaluation/integration_call_budget.py']:
        rel=path.relative_to(ROOT);dest=run/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=sha(path)
    preparation=json.loads((PREPARATION/'result.json').read_text())
    candidate_hashes={}
    for filename,identity in preparation['reviewed_shadow_source'].items():
        source=PREPARATION/'generated-shadow'/filename
        assert sha(source)==identity['candidate_sha256']
        dest=run/'candidate-src'/filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        candidate_hashes[filename]=sha(source)
    resources=ROOT/'runs/20261114T081540Z-client-roots-p0-v1-da6f4c/resources-candidate.py.source'
    (run/'resource-monitor-candidate.py.source').write_bytes(resources.read_bytes())
    manifest={'run_id':name,'source_sha256':hashes,'candidate_source_sha256':candidate_hashes,
        'resource_candidate_sha256':sha(resources),'corpus_manifest_sha256':sha(corpuspath),'cases':order,
        'max_attempted_ASR_calls_total':4,'max_agent_VLM_calls_including_startup':4,
        'private_phase_max_seconds':300,'native_calls_started':False,
        'prerequisite_original8h':ENDURANCE,'prerequisite_ASRcomponent':SOAK,
        'case_scope':'Two previously usedFLEURSDEVutterances, not newheldoutsamples'}
    (run/'manifest.json').write_text(encode(manifest),encoding='utf-8')
    (ROOT/'local/current-ASR-native-integration-run.txt').write_text(name,encoding='utf-8')
    trace=Store(ROOT/'local/hybrid-v1')
    trace.event('ASR_profile.native_cases_locked',{'run_id':name,'cases':[{'model':c['model'],'sample_id':c['sample']['id']} for c in order],
        'actual_native_calls':0,'waiting_for_final_endurance_audit':True},'orchestrator');trace.close()
    print(encode({'run_id':name,'actual_native_calls':0,'stage':'source_and_cases_locked_only'}))
    return run


def charge():
    return charge_budget(ROOT/'local/ASR-native-integration-p0-v1-budget.json','ASR')


def execute(run):
    run=run.resolve()
    if not run.is_relative_to(ROOT/'runs'):raise ValueError('Expected owned locked run')
    manifests=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    required=prerequisites() # Fail before data mutation, GPU allocation or inference.
    if (run/'private-started.json').exists():raise RuntimeError('Run already attempted; no overwrite or retry')
    for rel,expected in manifests['source_sha256'].items():
        if sha(ROOT/rel)!=expected:raise RuntimeError('Integration source changed before execution: '+rel)
    for filename,digest in manifests['candidate_source_sha256'].items():
        if sha(run/'candidate-src'/filename)!=digest:raise RuntimeError('Locked candidate bytes changed')
    native=SharedNative(ROOT);native.check()
    if gpu_headroom()['free_MiB']<1800:raise MemoryError('ASRintegration launch headroom not available')
    with (run/'private-started.json').open('x',encoding='utf-8') as f:
        f.write(encode({'started_epoch':time.time(),**required}))
    directory=ROOT/'local'/run.name;directory.mkdir()
    live=ROOT/'local/hybrid-v1'
    for profile in PROFILES.values():
        dest=directory/profile.relative_path;dest.parent.mkdir(parents=True,exist_ok=True)
        os.link(live/profile.relative_path,dest)
    for rel in ('native-integrity.json','candidates/whisper-small/native-storage-receipt.json'):
        dest=directory/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((live/rel).read_bytes())
    shutil.copytree(live/'runtime/whisper.cpp-b5454-cuda12.4',directory/'runtime/whisper.cpp-b5454-cuda12.4',copy_function=os.link)
    path=directory/'resources.py';path.write_bytes((run/'resource-monitor-candidate.py.source').read_bytes())
    monitor_module=load_module('scarx.hybrid._ASR_native_P0_resources',path)
    monitor=monitor_module.ResourceMonitor(limit=None,extra_pids={native.owner.pid},include_hermes=True).start()
    trace=Store(live);private=Store(directory)
    config_module=load_module('scarx.hybrid._ASR_native_P0_config',run/'candidate-src/config.py')
    backend_module=load_module('scarx.hybrid._ASR_native_P0_backends',run/'candidate-src/backends.py')
    samples=[];actual_sample=backend_module.gpu_headroom
    def measured_headroom(device):
        value=actual_sample(device);samples.append({'epoch':time.time(),**value});return value
    backend_module.gpu_headroom=measured_headroom
    class ForegroundMonitor:
        active_backend=None
        limit=None
        def check(self):
            if self.active_backend and native.status().get('cognition',{}).get('phase')=='serving':
                self.active_backend.cancel_idle.set()
                if self.active_backend.asr is None:
                    raise InterruptedError('Foreground work arrived before private ASRlaunch')
            return monitor.check()
        def sample(self):return monitor.sample()
    foreground_monitor=ForegroundMonitor()
    backends={model:backend_module.NativeBackends(directory,private,foreground_monitor,
        config_module.HybridConfig(execution_mode='gpu',asr_model_name='openai/whisper-'+model,
                                  asr_revision=PROFILES['openai/whisper-'+model].revision))
        for model in ('base','small')}
    from scarx.hybrid.engine import HybridEngine
    engine=HybridEngine(private,backends['small'],monitor,backends['small'].config)
    engine.model.to_device('gpu');engine.state['learning_enabled']=False
    pin=engine.save(pinned=True);baseline=private.snapshot(pin)
    report={'run_id':run.name,'prerequisites':required,'calls':[],'failures':[],
        'new_learning_or_live_weights_import':False,'live_ASR_changed':False,
        'native_owner_pid':native.owner.pid,'native_worker_pid':native.worker.pid,
        'native_worker_created':native.worker_created,'global_free_samples':samples}
    started=time.monotonic()
    try:
        for case in manifests['cases']:
            if time.monotonic()-started>=300:raise TimeoutError('Private ASRintegration budget300s exceeded')
            native.check()
            wait=time.monotonic()
            while native.status().get('cognition',{}).get('phase')=='serving':
                if time.monotonic()-wait>=60:raise TimeoutError('Liveforeground retains priority')
                time.sleep(.25)
            sample=case['sample'];audio=Path(sample['audio'])
            if sha(audio)!=sample['audio_sha256']:raise ValueError('Locked audio changed')
            attempt=charge();backend=backends[case['model']]
            foreground_monitor.active_backend=backend
            call_started=time.monotonic()
            try:transcript=backend.transcribe(audio)
            finally:foreground_monitor.active_backend=None
            events=private.events(after=0,limit=200)
            result=next(e['payload'] for e in reversed(events) if e['kind']=='asr_result')
            expected=PROFILES['openai/whisper-'+case['model']]
            if (result['ASR_identity']!=expected.identity() or result['actual_neural_device']!='CUDA0'
                    or not result['execution_device_log_verified'] or not result['precision_log_verified']):
                raise ValueError('Actualcandidate ASRidentity/device evidence disagrees')
            native.check()
            if engine.model.dumps(engine.state)!=baseline:raise ValueError('Private ALIF/controller changed without learning')
            record={'model':case['model'],'attempt':attempt,'sample_id':sample['id'],'language':sample['language'],
                'actual_transcript':transcript,'seconds':time.monotonic()-call_started,
                'ASR_result':result,'source_audio_sha256':sample['audio_sha256'],
                'selected_native_sha256':sha(backend.asr_model_path()),
                'model_storage':backend.asr_storage_receipt,'new_independent_quality_sample':False}
            report['calls'].append(record)
            raw=Path(result['raw_result']).read_bytes()
            (run/f'actual-call-{attempt}.json').write_bytes(raw)
            trace.event('ASR_profile.actual_selected_call',{'run_id':run.name,**record},'orchestrator')
        if len(report['calls'])!=4:raise RuntimeError('Missing locked ASRcase')
        from scarx.hybrid.learner import LocalALIF
        clone,controller=LocalALIF.loads(baseline,device='gpu')
        if clone.dumps(controller)!=baseline:raise ValueError('Private coldGPUclone differs')
        engine.load(pin)
        if engine.model.dumps(engine.state)!=baseline:raise ValueError('Private actualrollback differs')
        if samples and min(s['free_MiB'] for s in samples)<512:raise MemoryError('ASRintegration reservebreach')
        for rel,expected in manifests['source_sha256'].items():
            if sha(ROOT/rel)!=expected:raise RuntimeError('Locked source changed during privateASR')
        native.check()
        report.update(private_cold_GPU_clone_and_rollback_exact=True,private_neural_updates=engine.model.updates,
            actual_original_VLM_identity_unchanged=True,gate_pass=True,
            verdict='PRIVATE_SELECTED_ASR_NATIVE_INTEGRATION_PASS_NOT_LIVE_PROMOTION')
    except Exception:
        report['failures'].append(traceback.format_exc());report['gate_pass']=False
        report['verdict']='PRIVATE_SELECTED_ASR_INTEGRATION_FAILED'
    finally:
        for backend in backends.values():backend.close()
        engine.close();report['resources']=monitor.close();private.close()
        report['seconds']=time.monotonic()-started
        with (run/'private-result.json').open('x',encoding='utf-8') as f:f.write(encode(report))
        trace.event('ASR_profile.private_phase_completed',{k:v for k,v in report.items() if k not in ('calls','global_free_samples','resources')},'orchestrator');trace.close()
        print(encode({k:report[k] for k in ('run_id','verdict','gate_pass','failures','seconds')}))
    return not report['failures'] and report['gate_pass']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','execute'));parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:
        if args.run is None:parser.error('--run is required')
        raise SystemExit(0 if execute(args.run) else 1)
