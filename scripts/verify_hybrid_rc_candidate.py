"""Check an extracted RC and the live-install refusal without installing it.

CPU/SQLite/transport fixtures exercise real packaged SCAR-X code. Their backend
outputs are declared fixtures, not chat/vision quality or new research results.
No native model calls, live learning, service stop/start or user-state restore.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def validate_trusted_sources(packager, package, manifest):
    # A manifest and a sidecar alone are self-declared hashes, not authenticity.
    # Anchor executable contents to this reviewed checkout before executing them.
    if (package/'manage.py').read_bytes() != (ROOT/'release/hybrid/manage.py').read_bytes():
        raise ValueError('Extracted manager differs from trusted release source')
    allowed = json.loads((ROOT/'release/hybrid/payload-files.json').read_text())['files']
    if {name.removeprefix('payload/') for name in manifest['files'] if name.startswith('payload/')} != set(allowed):
        raise ValueError('Extracted executable payload differs from trusted allowlist')
    overlays, _ = packager.overlay_candidates(ROOT)
    for name in allowed:
        expected = overlays[name] if name in overlays else (ROOT/name).read_bytes()
        if (package/'payload'/name).read_bytes() != expected:
            raise ValueError('Extracted executable differs from trusted source: '+name)
    spec=json.loads((ROOT/'release/hybrid/payload-files.json').read_text(encoding='utf-8'))
    predecessor=packager.predecessor_manifest(ROOT,spec)
    if predecessor:
        expected={name:predecessor['files'].get('payload/'+name,{}).get('sha256') for name in allowed}
        if manifest.get('target_before_sha256')!=expected or manifest.get('predecessor')!=spec['predecessor']:
            raise ValueError('Upgrade baseline differs from trusted predecessor')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    archive_path = args.archive.resolve()
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_path.with_suffix('.zip.sha256').read_text().split()[0] != digest:
        raise ValueError('Archive checksum differs from sidecar')
    packager = load('_hybrid_rc_packager_check', ROOT/'scripts/package_hybrid_rc.py')
    manager = load('_hybrid_rc_manager_check', ROOT/'release/hybrid/manage.py')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = ROOT/'local/release-verification'/stamp
    package = run/'package'
    package.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for member in packager.zip_members(archive):
            destination = package/member.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
    manifest = manager.verify_package(package)
    validate_trusted_sources(packager, package, manifest)
    verified = subprocess.run([sys.executable, '-I', '-B', str(package/'manage.py'),
                               '--package', str(package), '--verify-only'],
                              capture_output=True, text=True, encoding='utf-8', timeout=30)
    if verified.returncode:
        raise RuntimeError('Extracted verifier failed: '+verified.stderr[-2000:])
    (run/'verify-only.json').write_text(verified.stdout, encoding='utf-8')
    # Exercise the actual registered protected gate, without reaching stop/write.
    blocked = subprocess.run([sys.executable, '-I', '-B', str(package/'manage.py'),
                              '--package', str(package), '--check'],
                             capture_output=True, text=True, encoding='utf-8', timeout=30)
    if blocked.returncode == 0 or 'Protected 8-hour test is still running' not in blocked.stderr:
        raise RuntimeError('This pre-terminal check expected a protected live worker; no installation was attempted: '+blocked.stderr[-1500:])
    (run/'protected-check.stderr.log').write_text(blocked.stderr, encoding='utf-8')
    config = run/'pytest.ini'
    config.write_text('[pytest]\n', encoding='utf-8')
    program = '''import sys,pathlib,json
src=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(src))
import scarx.hybrid.gateway,scarx.runtime.maintenance
import pytest
code=pytest.main(sys.argv[2:])
loaded={n:str(m.__file__) for n,m in sys.modules.items() if n=='scarx' or n.startswith('scarx.')}
assert all(pathlib.Path(p).resolve().is_relative_to(src) for p in loaded.values()), loaded
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
print(json.dumps({'all_SCARX_imports_from_payload':True,'torch_imported':False,'modules':sorted(loaded)}))
raise SystemExit(code)
'''
    tests = ['tests/test_hybrid_learning.py', 'tests/test_hybrid_gateway.py',
             'tests/test_hybrid_modes.py', 'tests/test_hybrid_cognition.py']
    command = [sys.executable, '-I', '-B', '-c', program, str(package/'payload/src'),
               '-c', str(config), '-q', '-k', 'not hybrid_alif_forward_matches_existing_native_core',
               '--junitxml='+str(run/'junit.xml'), *[str(ROOT/path) for path in tests]]
    checked = subprocess.run(command, cwd=run, capture_output=True, text=True,
                             encoding='utf-8', timeout=90)
    (run/'regression.stdout.log').write_text(checked.stdout, encoding='utf-8')
    (run/'regression.stderr.log').write_text(checked.stderr, encoding='utf-8')
    if checked.returncode:
        raise RuntimeError('Extracted regression failed; retained logs at '+str(run))
    suite = ET.parse(run/'junit.xml').getroot().find('testsuite')
    if suite is None or int(suite.attrib['tests']) == 0 or int(suite.attrib['failures']) or int(suite.attrib['errors']):
        raise RuntimeError('No passing regression receipt')
    # A read-only monitor confirms worker liveness and source custody after tests.
    pointer = (ROOT/'local/current-small-mixed-endurance-run.txt').read_text().strip()
    monitor = subprocess.run([sys.executable, str(ROOT/'scripts/monitor_hybrid_endurance.py'), pointer],
                             capture_output=True, text=True, encoding='utf-8', timeout=15)
    protected = json.loads(monitor.stdout)
    if monitor.returncode or not protected.get('actual_process_live') or not protected.get('frozen_source_matches'):
        raise RuntimeError('Protected worker/source must be inspected; no installation was attempted')
    receipt = {'status':'EXTRACTED_RC_FIXTURES_AND_PRETERMINAL_INSTALL_REFUSAL_VERIFIED',
               'archive_sha256':digest, 'source_git_sha':manifest['source_git_sha'],
               'tests':int(suite.attrib['tests']), 'failures':0, 'errors':0,
               'excluded_native_parity_test':True, 'backend_quality_measured':False,
               'isolated_ALIF_fixture_compute_executed':True,
               'native_calls':0, 'live_state_modified_or_service_restarted':False,
               'protected_run':protected, 'result_directory':str(run)}
    (run/'result.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
