"""Existing-install RC overlay. No private data is packaged or database restored."""

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import sqlite3
import sys
import tempfile
import time
import psutil
import shutil


def sha(raw):return hashlib.sha256(raw).hexdigest()


def atomic(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.scarx-rc-',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def safe_relative(name):
    p=PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or p.is_absolute()
            or any(part in ('','..','.') for part in name.split('/'))):
        raise ValueError('Unsafe release path')
    return p


def verify_package(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('format')!='scarx-hybrid-local-rc-v1' or manifest.get('model_id')!='scarx-hybrid':
        raise ValueError('Not a SCAR-X Hybrid release manifest')
    expected=set(manifest['files'])|{'MANIFEST.json'}
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError('Unexpected or missing package files')
    total=0
    for name,info in manifest['files'].items():
        safe_relative(name);path=(directory/name)
        if path.is_symlink() or not path.resolve().is_relative_to(directory):raise ValueError('Package link escapes root')
        raw=path.read_bytes();total+=len(raw)
        if len(raw)>2_000_000 or len(raw)!=info['bytes'] or sha(raw)!=info['sha256']:
            raise ValueError('Package file integrity mismatch: '+name)
        if name.startswith('payload/'):
            target=name.removeprefix('payload/')
            safe_relative(target)
            if target.split('/')[0] in ('local','runs','data','artifacts','.git','tests'):
                raise ValueError('Private/research payload forbidden')
            if any(part in ('__pycache__','.venv','venv','cache') for part in PurePosixPath(target).parts):
                raise ValueError('Runtime environment/cache is not code payload')
            if PurePosixPath(target).suffix not in ('.py','.js','.json','.yaml','.html') or PurePosixPath(target).name in ('settings.json','config.json','auth.json'):
                raise ValueError('Unexpected generated/private payload file')
    if total>20_000_000:raise ValueError('Code-only release exceeds expected size')
    if 'payload/run_hybrid.py' not in expected:raise ValueError('Hybrid entrypoint missing')
    return manifest


def registered_target(explicit=None):
    home=Path(os.environ['LOCALAPPDATA'])/'hermes'
    settings=json.loads((home/'plugins/scarx-hybrid/dashboard/settings.json').read_text(encoding='utf-8'))
    entry=Path(settings['entrypoint']).resolve();root=entry.parent
    if settings.get('gateway')!='http://127.0.0.1:8775' or entry.name!='run_hybrid.py':
        raise ValueError('Unexpected registered SCAR-X provider')
    if explicit and Path(explicit).resolve()!=root:raise ValueError('Target differs from registered installation')
    data=Path(settings['data_dir']).resolve();python=Path(settings['python']).resolve()
    if data!=root/'local/hybrid-v1' or not python.is_file() or not entry.is_file():
        raise ValueError('Existing registered runtime is required')
    return root,data,python,home


def no_live_protected_run(root):
    pointer=root/'local/current-small-mixed-endurance-run.txt'
    if not pointer.is_file():raise RuntimeError('Missing tested installation gate; this RC updates the registered tested deployment only')
    run=Path(pointer.read_text(encoding='utf-8').strip()).resolve()
    if not run.is_relative_to(root/'runs'):raise ValueError('Protected run path outside installation')
    m=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    identity=json.loads((run/'process-identity-v1.json').read_text(encoding='utf-8'))
    try:
        process=psutil.Process(m['pid'])
        live=process.is_running() and abs(process.create_time()-identity['process_created_epoch'])<.01
    except psutil.NoSuchProcess:live=False
    if live:raise RuntimeError('Protected 8-hour test is still running; no stop, restart or source change is allowed')
    return run,m


def protected_run_gate(root):
    run,m=no_live_protected_run(root)
    result=json.loads((run/'result.json').read_text(encoding='utf-8'))
    audit=json.loads((run/'small-mixed-audit-v1.json').read_text(encoding='utf-8'))
    if (m['duration_seconds']!=28800 or not result.get('gate_pass') or result['failures']
            or result['elapsed_seconds']<28800 or len(result['video_cycles'])<80 or len(result['checkpoints'])<8
            or not audit.get('engineering_data_verified')
            or audit['verdict']!='SMALL_MIXED_ENGINEERING_DATA_VERIFIED_ONLY'
            or not audit['ASR_integrity'] or not audit['source_archive_valid']
            or not audit['private_checkpoint_and_receipt_agree']):
        raise RuntimeError('Completed test/independent data audit not qualified')
    for name,digest in m['source_sha256'].items():
        safe_relative(name)
        if not (root/name).resolve().is_relative_to(root):raise ValueError('Protected source outside root')
        if sha((root/name).read_bytes())!=digest:raise RuntimeError('Protected source changed since tested baseline: '+name)
    return {'run_id':run.name,'result_sha256':sha((run/'result.json').read_bytes()),
            'audit_sha256':sha((run/'small-mixed-audit-v1.json').read_bytes())}


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def plan_files(package,root,manifest):
    changes=[]
    for name,info in manifest['files'].items():
        if not name.startswith('payload/'):continue
        relative=name.removeprefix('payload/');target=(root/relative)
        if not target.resolve().is_relative_to(root):raise ValueError('Target path escapes installation')
        before=target.read_bytes() if target.is_file() else None
        after=(package/name).read_bytes()
        if before==after:continue
        expected=manifest['target_before_sha256'].get(relative)
        if (sha(before) if before is not None else None)!=expected:
            raise RuntimeError('Target source differs from tested baseline: '+relative)
        changes.append((relative,before,after))
    return changes


def validate_existing_assets(root,home,package):
    pins=json.loads((package/'ASSETS.json').read_text(encoding='utf-8'))
    for relative in pins.get('runtime_paths_required',[]):
        safe_relative(relative);path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise RuntimeError('Existing runtime prerequisite missing: '+relative)
    search=json.loads((root/'local/searxng/installation.json').read_text(encoding='utf-8'))
    if search.get('commit')!=pins['runtime_releases']['SearXNG_commit']:
        raise RuntimeError('Native SearXNG revision differs from tested pin')
    if not Path(search['source']).resolve().is_relative_to(root/'local/searxng'):
        raise RuntimeError('Native SearXNG source is outside registered installation')
    for relative,entry in pins['installation_files'].items():
        safe_relative(relative);path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size!=entry['bytes']:
            raise RuntimeError('Missing pinned runtime asset: '+relative)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(2<<20):digest.update(block)
        if digest.hexdigest()!=entry['sha256']:raise RuntimeError('Runtime asset checksum mismatch: '+relative)
    for relative,entry in pins['hermes_integration_files'].items():
        safe_relative(relative);path=(home/relative).resolve()
        if not path.is_relative_to(home) or not path.is_file() or sha(path.read_bytes())!=entry['sha256']:
            raise RuntimeError('Compatible existing Hermes integration required: '+relative)
    import importlib.metadata as metadata
    versions=json.loads((package/'DEPENDENCIES.json').read_text(encoding='utf-8'))
    for name,version in versions['packages'].items():
        if metadata.version(name)!=version:raise RuntimeError('Runtime dependency version mismatch: '+name)
    for name,entry in pins.get('system_tools',{}).items():
        found=shutil.which(name)
        if not found:raise RuntimeError('Required system media tool missing: '+name)
        digest=hashlib.sha256()
        with Path(found).open('rb') as stream:
            while block:=stream.read(2<<20):digest.update(block)
        if digest.hexdigest()!=entry['sha256']:raise RuntimeError('System media tool differs from tested pin: '+name)


def snapshot_backup(data,backup):
    database=(data/'hybrid.sqlite').resolve()
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
        pointer=json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()[0])
        digest,payload=db.execute('SELECT digest,payload FROM checkpoints WHERE id=?',(pointer['id'],)).fetchone()
    if len(payload)>16_000_000 or sha(payload)!=digest:raise RuntimeError('Active private checkpoint integrity failure')
    (backup/'private-full-checkpoint.bin').write_bytes(payload)
    # Pin the existing snapshot; never replace the DB or rewind its journal.
    with sqlite3.connect(database) as db:
        db.execute("INSERT OR REPLACE INTO objects(kind,id,payload) VALUES('checkpoint_pin',?,?)",(pointer['id'],json.dumps({'id':pointer['id']})))
    return {'checkpoint_id':pointer['id'],'private_checkpoint_sha256':digest}


def assert_registered_owner_runtime(root,data):
    lock=data/'runtime.lock'
    if not lock.exists():return
    owner=json.loads(lock.read_text(encoding='utf-8'))
    process=psutil.Process(owner['pid'])
    if abs(process.create_time()-owner['created'])>=.01:raise RuntimeError('Service owner PID was reused')
    if Path(process.exe()).resolve()!=Path(psutil.Process().exe()).resolve():
        raise RuntimeError('Service owner differs from registered runtime interpreter')
    if not any(Path(arg).resolve()==root/'run_hybrid.py' for arg in process.cmdline()[1:] if not arg.startswith('--')):
        raise RuntimeError('Service owner command is not the registered Hybrid entrypoint')


def write_overlay(root,backup,changes):
    records=[];written=[]
    try:
        for relative,before,after in changes:
            target=root/relative;current=target.read_bytes() if target.is_file() else None
            if current!=before:raise RuntimeError('Concurrent source edit: '+relative)
            if before is not None:
                saved=backup/'files'/relative;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(before)
            record={'path':relative,'before_sha256':sha(before) if before is not None else None,'after_sha256':sha(after)}
            records.append(record)
        atomic(backup/'files.json',json.dumps(records,indent=2).encode())
        for relative,before,after in changes:
            target=root/relative
            if (target.read_bytes() if target.is_file() else None)!=before:raise RuntimeError('Concurrent source edit during overlay')
            atomic(target,after);written.append((relative,before,after))
    except Exception:
        for relative,before,after in reversed(written):
            target=root/relative
            if target.is_file() and target.read_bytes()==after:
                if before is None:target.unlink()
                else:atomic(target,before)
        raise
    return records


def rollback_files(root,backup,apply=True):
    records=json.loads((backup/'files.json').read_text(encoding='utf-8'))
    # Validate every target before changing any, preserving later user edits.
    for item in records:
        safe_relative(item['path']);target=(root/item['path']).resolve()
        if not target.is_relative_to(root):raise ValueError('Rollback target escapes root')
        if not target.is_file() or sha(target.read_bytes())!=item['after_sha256']:
            raise RuntimeError('Rollback refuses a later source edit: '+item['path'])
        if item['before_sha256'] is not None and sha((backup/'files'/item['path']).read_bytes())!=item['before_sha256']:
            raise RuntimeError('Rollback backup integrity failure')
    if not apply:return
    for item in reversed(records):
        target=root/item['path']
        if item['before_sha256'] is None:target.unlink()
        else:atomic(target,(backup/'files'/item['path']).read_bytes())


def wait_ready(service,seconds=90):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        state=service.status()
        if state and state.get('failure'):raise RuntimeError('Gateway qualification failed: '+str(state['failure']))
        if state and state.get('ready'):return state
        time.sleep(.25)
    raise TimeoutError('Gateway did not become ready')


def main(args):
    package=Path(args.package).resolve() if args.package else Path(__file__).resolve().parent
    manifest=verify_package(package)
    if args.verify_only:
        print(json.dumps({'package_verified':True,'version':manifest['version'],'payload_files':len([n for n in manifest['files'] if n.startswith('payload/')])}));return
    root,data,python,home=registered_target(args.target_root)
    # Validate source bytes before importing any target helper/package code.
    changes=plan_files(package,root,manifest)
    if not args.rollback and not changes:
        print(json.dumps({'already_matches_payload':True,'version':manifest['version']}));return
    sys.path.insert(0,str(root/'src'))
    from scarx.runtime.maintenance import MaintenanceLease
    from scarx.hybrid.ownership import OwnerLock
    service=load_module('_scarx_rc_registered_service',root/'scripts/hybrid_service.py')
    assert_registered_owner_runtime(root,data)
    if args.rollback:
        pointer=json.loads((data/'release-backups/latest.json').read_text(encoding='utf-8'))
        backup=Path(pointer['backup']).resolve()
        if not backup.is_relative_to(data/'release-backups'):raise ValueError('Invalid rollback location')
        no_live_protected_run(root)
        rollback_files(root,backup,apply=False)
        lease=MaintenanceLease(data)
        try:
            before=service.status()
            if before and before.get('cognition',{}).get('phase') in ('serving','observing'):raise RuntimeError('User work active; defer rollback')
            service.stop();owner=OwnerLock(data)
            try:rollback_files(root,backup)
            finally:owner.close()
        finally:lease.close()
        service.start();state=wait_ready(service)
        print(json.dumps({'code_rollback':True,'private_database_and_current_checkpoint_retained':True,'ready':state['ready']}));return
    gate=protected_run_gate(root)  # Before stop/lease/write, even for apply.
    if args.check:
        print(json.dumps({'preflight_source_gate':True,'changed_files':[r for r,_,_ in changes],'test_gate':gate,'applied':False}));return
    validate_existing_assets(root,home,package)
    from scarx.hybrid.config import load_runtime_config
    config=load_runtime_config(data)
    if config.execution_mode!='gpu' or config.asr_model_name!='openai/whisper-small':
        raise RuntimeError('This RC targets the current tested GPU/small-ASR profile; configuration was not changed')
    current=service.status()
    if current and current.get('cognition',{}).get('phase') in ('serving','observing'):raise RuntimeError('User work active; defer update')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=data/'release-backups'/stamp;backup.mkdir(parents=True)
    lease=MaintenanceLease(data);records=None
    try:
        service.stop();owner=OwnerLock(data)
        try:
            private=snapshot_backup(data,backup)
            records=write_overlay(root,backup,changes)
            receipt={'version':manifest['version'],'source_git_sha':manifest['source_git_sha'],
                'test_gate':gate,'files':records,'private_backup':private,'database_replaced':False}
            atomic(backup/'receipt.json',json.dumps(receipt,indent=2).encode())
            atomic(data/'release-backups/latest.json',json.dumps({'backup':str(backup)}).encode())
        finally:owner.close()
    except Exception:
        if records is not None:
            recovery_owner=OwnerLock(data)
            try:rollback_files(root,backup)
            finally:recovery_owner.close()
        # write_overlay restores its own writes on exceptions. Restart the
        # previous service only when every code byte is back at its beforestate.
        safe=all((root/relative).is_file() and (root/relative).read_bytes()==before
                 if before is not None else not (root/relative).exists()
                 for relative,before,_ in changes)
        lease.close()
        if safe and current:
            service.start();wait_ready(service)
        raise
    finally:lease.close()
    # A failed startup is reported explicitly; the intact local code/state
    # backup supports --rollback. No forced process kill or database restore.
    service.start();state=wait_ready(service)
    print(json.dumps({'updated':True,'version':manifest['version'],'ready':state['ready'],
        'gateway_pid':state['pid'],'code_backup':str(backup),'current_private_state_retained':True,
        'release_kind':'RC; operational acceptance still required'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path);p.add_argument('--target-root',type=Path)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--rollback',action='store_true')
    main(p.parse_args())
