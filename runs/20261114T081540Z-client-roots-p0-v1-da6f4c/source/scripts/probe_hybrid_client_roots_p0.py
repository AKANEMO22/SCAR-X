"""Real process-membership probe of an isolated monitor overlay; no inference."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from scarx.hybrid.resources import ResourceMonitor
from scarx.hybrid.store import Store,encode
from scarx.evaluation.shared_native import SharedNative
from scarx.evaluation.hermes_resource_roots import verified_gateway_pid
from monitor_hybrid_endurance import inspect


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-client-roots-p0-v1-'+uuid.uuid4().hex[:6]
    run=ROOT/'runs'/name;run.mkdir()
    source=ROOT/'src/scarx/hybrid/resources.py'
    original=source.read_bytes()
    recipe=ROOT/'integrations/hermes/resource_client_roots_patch.py'
    spec=importlib.util.spec_from_file_location('resource_root_recipe',recipe)
    patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
    candidate=patch.patch_bytes(original)
    directory=ROOT/'local'/name;directory.mkdir()
    overlay=directory/'resources_candidate.py';overlay.write_bytes(candidate)
    (run/'resources-candidate.py.source').write_bytes(candidate)
    sources=[Path(__file__),recipe,ROOT/'src/scarx/evaluation/hermes_resource_roots.py',
             ROOT/'src/scarx/hybrid/resources.py',ROOT/'src/scarx/hybrid/gpu_memory.py',
             ROOT/'src/scarx/evaluation/shared_native.py',ROOT/'experiments/hybrid-resource-client-roots-p0-v1.md']
    hashes={}
    for path in sources:
        rel=path.relative_to(ROOT);dest=run/'source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=sha(path.read_bytes())
    (run/'manifest.json').write_text(encode({'source_sha256':hashes,
        'original_resource_sha256':sha(original),'candidate_sha256':sha(candidate),
        'max_seconds':30,'installed_into_active_source':False,'neural_inference_or_learning':False}),encoding='utf-8')
    report={'run_id':name,'failures':[],'installed_into_active_source':False,'new_inference_or_learning':False}
    started=time.monotonic()
    trace=Store(ROOT/'local/hybrid-v1')
    try:
        native=SharedNative(ROOT);native.check()
        home=Path(os.environ['LOCALAPPDATA'])/'hermes'
        gateway=verified_gateway_pid(home)
        if gateway is None:raise RuntimeError('Current Hermes gateway not verifiable')
        module_name='scarx.hybrid.resources_client_roots_p0'
        spec=importlib.util.spec_from_file_location(module_name,overlay)
        module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
        samples={}
        attached={}
        lab=inspect(ROOT/'runs/20261114T011841Z-shared-endurance-v1-554c19')
        if not lab['actual_process_live']:raise RuntimeError('Actual private lab root needed for attachment compatibility probe')
        for key,cls in (('original',ResourceMonitor),('candidate',module.ResourceMonitor)):
            monitor=cls(limit=None,extra_pids={native.owner.pid},include_hermes=True)
            try:
                samples[key]=monitor.sample()
                monitor.extra_pids.add(lab['pid'])
                monitor.last_client_scan=0
                attached[key]=monitor.sample()
            finally:monitor.close()
        original_sample,candidate_sample=samples['original'],samples['candidate']
        candidate_pids={row['pid'] for row in candidate_sample['processes']}
        assert native.owner.pid in candidate_sample['explicit_root_pids']
        assert native.owner.pid in candidate_sample['included_client_pids']
        assert gateway in candidate_sample['discovered_hermes_roots']
        assert gateway in candidate_pids and native.worker.pid in candidate_pids
        assert lab['pid'] in attached['candidate']['explicit_root_pids']
        assert lab['pid'] in {row['pid'] for row in attached['candidate']['processes']}
        assert verified_gateway_pid(home)==gateway
        assert sha(source.read_bytes())==sha(original),'Frozen resource source changed'
        for rel,digest in hashes.items():assert sha((ROOT/rel).read_bytes())==digest
        native.check()
        report.update(samples=samples,attached_samples=attached,validated_gateway_pid=gateway,explicit_nativeowner_pid=native.owner.pid,
            native_VLM_pid=native.worker.pid,original_dropped_explicit_root=native.owner.pid not in original_sample['included_client_pids'],
            original_current_gateway_included=gateway in {row['pid'] for row in original_sample['processes']},
            candidate_preserves_explicit_and_current_gateway_roots=True,
            existing_extra_pids_add_API_retained_after_refresh=True,attached_actual_lab_pid=lab['pid'],
            original_source_still_frozen=True,memory_delta_or_historical_peak_improvement_claim=False,
            verdict='P0_REAL_ROOT_MEMBERSHIP_VERIFIED_OVERLAY_ONLY')
    except Exception:
        report['failures'].append(traceback.format_exc());report['verdict']='CLIENT_ROOT_P0_INCOMPLETE'
    finally:
        report['seconds']=time.monotonic()-started
        if report['seconds']>30:report['failures'].append('Registered30second P0 budget exceeded');report['verdict']='CLIENT_ROOT_P0_INCOMPLETE'
        with (run/'result.json').open('x',encoding='utf-8') as f:f.write(encode(report))
        trace.event('client_roots_p0.actual_membership',{k:v for k,v in report.items() if k not in ('samples','attached_samples')},'orchestrator')
        trace.close()
        print(encode({k:v for k,v in report.items() if k not in ('samples','attached_samples')}))
    return not report['failures']


if __name__=='__main__':raise SystemExit(0 if main() else 1)
