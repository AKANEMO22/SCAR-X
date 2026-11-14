"""Read-only final artifact-to-install verification. No restart or inference."""

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(m)
    finally:sys.dont_write_bytecode=old
    return m


def main():
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);args=parser.parse_args()
    archive=args.archive.resolve();digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive.with_suffix('.zip.sha256').read_text(encoding='ascii').split()[0]!=digest:raise ValueError('ZIP hash mismatch')
    packager=load('_final_RC_packager',ROOT/'scripts/package_hybrid_rc.py')
    manager=load('_final_RC_manager',ROOT/'release/hybrid/manage.py')
    checker=load('_final_RC_source_trust',ROOT/'scripts/verify_hybrid_rc_candidate.py')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=ROOT/'local/release-verification'/stamp;package=directory/'package';package.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in packager.zip_members(bundle):
            dest=package/member.filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(bundle.read(member))
    manifest=manager.verify_package(package);checker.validate_trusted_sources(packager,package,manifest)
    root,data,python,home=manager.registered_target()
    if root!=ROOT or manager.plan_files(package,root,manifest):raise ValueError('Installed code does not match final payload')
    manager.assert_registered_owner_runtime(root,data)
    manager.validate_existing_assets(root,home,package)
    suite=ET.parse(ROOT/'local/media-admission-final-release-20261114.xml').getroot().find('testsuite')
    cases=list(suite.iter('testcase'))
    assert len(cases)==int(suite.attrib['tests']) and len(cases)==256
    assert not int(suite.attrib['failures']) and not int(suite.attrib['errors'])
    skipped=[c for c in cases if c.find('skipped') is not None]
    assert all(c.attrib['name']=='test_hybrid_alif_forward_matches_existing_native_core' for c in skipped)
    names={c.attrib['name'] for c in cases}
    assert 'test_long_chat_queries_search_every_chunk_without_rewriting_model_input' in names
    assert 'test_actual_Hermes_request_envelope_enables_one_readonly_probe' in names
    assert 'test_different_drafts_with_identical_text_cannot_steal_each_others_video' in names
    assert 'test_distinct_frame_requests_never_change_a_previous_returned_path' in names
    evidence=json.loads((ROOT/'docs/release/evidence/RC3_OPERATIONAL_SMOKE_20261114.json').read_text(encoding='utf-8'))
    assert evidence['installed_P0_candidates_exact'] and evidence['research_or_production_PASS'] is False
    sys.path.insert(0,str(ROOT/'src'))
    from scarx.evaluation.shared_native import SharedNative
    native=SharedNative(ROOT);status=native.status()
    assert status['ready'] and not status['failure'] and status['neural']['execution_device']=='gpu'
    main_sha=subprocess.check_output(['git','-C',str(ROOT.parents[2]),'rev-parse','main'],text=True).strip()
    assert main_sha=='37749041ef45d54951558fdeb0c3695616b47cb5'
    report={'status':'FINAL_RC_PAYLOAD_MATCHES_INSTALLED_CODE_AND_REGISTERED_RUNTIME',
            'version':manifest['version'],'source_git_sha':manifest['source_git_sha'],
            'archive':str(archive.relative_to(ROOT)),'archive_sha256':digest,'archive_bytes':archive.stat().st_size,
            'payload_files':len(manifest['target_before_sha256']),'pending_payload_changes':0,
            'native_asset_Hermes_dependency_pins_match':True,
            'Python_checks_passed':len(cases)-len(skipped),'native_CPU_parity_skipped':len(skipped),
            'long_input_and_compaction_evidence':'docs/release/evidence/LONG_INPUT_AND_COMPACTION_20261114.json',
            'request_owned_media_evidence':'docs/release/evidence/MEDIA_ADMISSION_REPAIRS_20261114.json',
            'gateway_pid':native.owner.pid,'native_GPU_pid':native.worker.pid,
            'live_neural_updates':status['neural']['updates'],'original_main_sha_unchanged':main_sha,
            'eight_hour_run_cancelled_not_PASS':True,'new_native_calls_or_training':0,
            'operational_evidence':'docs/release/evidence/RC3_OPERATIONAL_SMOKE_20261114.json',
            'research_or_production_PASS':False}
    path=archive.with_suffix('.installed-verification.json')
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    (directory/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
