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
    receipts={
        '1.0.0-rc.6':('local/media-admission-final-release-20261114.xml',256),
        '1.0.0-rc.7':('local/media-presentation-final-release-20261114/python.xml',263),
        '1.0.0-rc.8':('local/rc8-upgrade-final-release-20261115/python.xml',292),
        '1.0.0-rc.9':('local/feedback-atomic-wire-final-release-20261115/python.xml',301),
        '1.0.0-rc.10':('local/web-routing-final-release-20261115/python.xml',323),
        '1.0.0-rc.11':('local/web-context-release-20261115/python.xml',332),
        '1.0.0-rc.12':('local/web-context-release-20261115/python.xml',332),
        '1.0.0-rc.13':('local/web-context-release-20261115/python.xml',332),
    }
    receipt_path,expected_count=receipts[manifest['version']]
    suite=ET.parse(ROOT/receipt_path).getroot().find('testsuite')
    cases=list(suite.iter('testcase'))
    assert len(cases)==int(suite.attrib['tests']) and len(cases)==expected_count
    assert not int(suite.attrib['failures']) and not int(suite.attrib['errors'])
    skipped=[c for c in cases if c.find('skipped') is not None]
    assert all(c.attrib['name']=='test_hybrid_alif_forward_matches_existing_native_core' for c in skipped)
    names={c.attrib['name'] for c in cases}
    assert 'test_long_chat_queries_search_every_chunk_without_rewriting_model_input' in names
    assert 'test_actual_Hermes_request_envelope_enables_one_readonly_probe' in names
    assert 'test_different_drafts_with_identical_text_cannot_steal_each_others_video' in names
    assert 'test_distinct_frame_requests_never_change_a_previous_returned_path' in names
    if manifest['version'] in ('1.0.0-rc.7','1.0.0-rc.8','1.0.0-rc.9','1.0.0-rc.10','1.0.0-rc.11','1.0.0-rc.12'):
        assert 'test_non_ascii_questions_in_other_languages_are_not_forced_into_vietnamese' in names
        assert 'test_streamed_references_precede_terminal_chunk_and_raw_native_answer_is_logged_separately' in names
    if manifest['version'] in ('1.0.0-rc.8','1.0.0-rc.9','1.0.0-rc.10','1.0.0-rc.11','1.0.0-rc.12'):
        assert 'test_actual_ffmpeg_audio_decode_failure_keeps_frame_ingestion_alive' in names
        assert 'test_screen_audio_gaps_reach_model_as_coverage_limits_without_false_transcripts' in names
        assert 'test_upgrade_baseline_is_loaded_from_pinned_predecessor_not_new_worktree' in names
    if manifest['version'] in ('1.0.0-rc.9','1.0.0-rc.10','1.0.0-rc.11','1.0.0-rc.12'):
        assert 'test_actual_http_commit_before_response_cache_loss_is_retryable_without_second_learning' in names
        assert 'test_actual_sqlite_abort_returns_retryable_error_and_reverts_all_nested_commits' in names
        assert 'test_actual_socket_response_loss_after_commit_retries_without_learning_twice' in names
    if manifest['version']=='1.0.0-rc.10':
        assert 'test_bare_search_followup_uses_previous_user_topic_not_assistant_or_tool_instructions' in names
        targeted=ET.parse(ROOT/'local/web-routing-final-targeted-20261115.xml').getroot().find('testsuite')
        assert targeted.attrib['tests']=='102' and targeted.attrib['failures']==targeted.attrib['errors']=='0'
    if manifest['version'] in ('1.0.0-rc.11','1.0.0-rc.12'):
        assert 'test_public_web_context_excludes_unrelated_global_personal_memory' in names
        assert 'test_followup_keeps_actual_user_message_but_review_uses_original_task_and_sources' in names
        assert 'test_host_tool_continuation_is_not_saved_as_the_user_question' in names
    if manifest['version']=='1.0.0-rc.12':
        target=ET.parse(ROOT/'local/web-frame-time-final-20261115.xml').getroot().find('testsuite')
        assert target.attrib['tests']=='35' and target.attrib['errors']==target.attrib['failures']=='0'
    if manifest['version']=='1.0.0-rc.13':
        target=ET.parse(ROOT/'local/arithmetic-task-scope-20261115.xml').getroot().find('testsuite')
        assert target.attrib['tests']=='82' and target.attrib['errors']==target.attrib['failures']=='0'
        assert any(c.attrib['name']=='test_original_neural_project_verifier_and_historical_twenty_tasks_are_preserved'
                   for c in target.iter('testcase'))
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
            'Python_regression_receipt':receipt_path,
            'full_regression_reused_from_RC11':manifest['version'] in ('1.0.0-rc.12','1.0.0-rc.13'),
            'RC12_current_wrapper_review_feedback_targeted_checks':35 if manifest['version']=='1.0.0-rc.12' else None,
            'RC13_current_arithmetic_review_cognition_feedback_web_targeted_checks':82 if manifest['version']=='1.0.0-rc.13' else None,
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
