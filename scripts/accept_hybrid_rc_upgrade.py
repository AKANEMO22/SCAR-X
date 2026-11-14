"""Exercise an actual RC predecessor payload, source gate, CAS and code rollback.

All mutations are inside a disposable owned target. No inference, live source
installation, process stop or private checkpoint/database replacement.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    packager = load('_upgrade_packager', ROOT / 'scripts/package_hybrid_rc.py')
    manager = load('_upgrade_manager', ROOT / 'release/hybrid/manage.py')
    checker = load('_upgrade_trust', ROOT / 'scripts/verify_hybrid_rc_candidate.py')
    run = ROOT / 'local/release-upgrade-acceptance' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    package, target = run / 'package', run / 'target'
    package.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in packager.zip_members(bundle):
            destination = package / member.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(bundle.read(member))
    manifest = manager.verify_package(package)
    checker.validate_trusted_sources(packager, package, manifest)
    spec = json.loads((ROOT / 'release/hybrid/payload-files.json').read_text())
    predecessor = packager.predecessor_manifest(ROOT, spec)
    with zipfile.ZipFile(ROOT / spec['predecessor']['archive']) as bundle:
        for name in predecessor['files']:
            if name.startswith('payload/'):
                destination = target / name.removeprefix('payload/')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(bundle.read(name))
    original_run = Path((ROOT / 'local/current-small-mixed-endurance-run.txt').read_text().strip())
    frozen = json.loads((original_run / 'manifest.json').read_text())
    for name in frozen['source_sha256']:
        if 'payload/' + name not in predecessor['files']:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    copied_run = target / 'runs' / original_run.name
    copied_run.mkdir(parents=True)
    for name in ('manifest.json', 'process-identity-v1.json', 'user-cancellation-request-v1.json',
                 'user-cancellation-audit-v1.json', 'result.json'):
        shutil.copyfile(original_run / name, copied_run / name)
    local = target / 'local'
    local.mkdir()
    (local / 'current-small-mixed-endurance-run.txt').write_text(str(copied_run))
    # Show the old freeze actually blocks this otherwise valid predecessor.
    try:
        manager.protected_run_gate(target)
    except RuntimeError as exc:
        assert 'Protected source changed' in str(exc)
    else:
        raise AssertionError('Old terminal-run source freeze was not reproduced')
    changes = manager.plan_files(package, target, manifest)
    assert {name for name, _, _ in changes} == {
        'src/scarx/hybrid/screen.py', 'src/scarx/hybrid/gateway.py', 'scripts/hybrid_screen_worker.py'}
    gate = manager.protected_run_gate(target, manifest)
    assert gate['eight_hour_qualified'] is False
    backup = run / 'backup'
    backup.mkdir()
    records = manager.write_overlay(target, backup, changes)
    assert manager.plan_files(package, target, manifest) == []
    assert manager.protected_run_gate(target, manifest)['eight_hour_qualified'] is False
    with sqlite3.connect(target / 'user-memory.sqlite') as db:
        db.execute('CREATE TABLE experience(value TEXT)')
        db.execute("INSERT INTO experience VALUES('NEW_AFTER_UPDATE_KEEP_ME')")
    manager.rollback_files(target, backup)
    assert manager.plan_files(package, target, manifest) == changes
    with sqlite3.connect(target / 'user-memory.sqlite') as db:
        assert db.execute('SELECT value FROM experience').fetchone()[0] == 'NEW_AFTER_UPDATE_KEEP_ME'
    # A later source edit must block the whole plan before other targets change.
    relative, _, _ = changes[0]
    (target / relative).write_bytes(b'OWNED_SIMULATED_USER_EDIT')
    try:
        manager.plan_files(package, target, manifest)
    except RuntimeError as exc:
        assert 'differs from tested baseline' in str(exc)
    else:
        raise AssertionError('Unexpected user edit was accepted')
    for name, before, _ in changes[1:]:
        assert (target / name).read_bytes() == before
    report = {'verdict': 'PINNED_RC7_TO_RC8_SOURCE_GATE_CAS_ROLLBACK_VERIFIED_ON_OWNED_CLONE',
              'predecessor_archive_sha256': spec['predecessor']['sha256'],
              'next_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'old_permanent_source_freeze_reproduced': True,
              'changed_payload_paths': [item['path'] for item in records],
              'both_predecessor_and_next_payload_states_accepted': True,
              'unexpected_user_source_edit_rejected_before_mutation': True,
              'code_rollback_preserves_later_fixture_user_data': True,
              'cancelled_eight_hour_test_remains_not_PASS': True,
              'actual_live_install_or_neural_calls_in_this_check': False}
    (run / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (ROOT / 'local/current-release-upgrade-acceptance.txt').write_text(str(run), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
