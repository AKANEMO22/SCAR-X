"""Activate a qualified on-disk one-file routing hotfix, preserving live state.

This checkout is also the registered serving source. Source was edited/tested
before restart; rollback bytes therefore come from the pinned previous release,
not a fabricated claim that the previous file was still present on disk.
"""
from datetime import datetime, timezone
import hashlib, importlib.util, io, json
from pathlib import Path
import sqlite3, subprocess, sys, time, zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.runtime.maintenance import MaintenanceLease


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def snapshot():
    path=ROOT/'local/hybrid-v1/hybrid.sqlite'
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        pointer=json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()[0])
        digest,payload=db.execute('SELECT digest,payload FROM checkpoints WHERE id=?',(pointer['id'],)).fetchone()
    if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Checkpoint integrity')
    return payload


def main():
    archive=Path(sys.argv[1]).resolve()
    manager=load('_dated_manager',ROOT/'release/hybrid/manage.py')
    packager=load('_dated_packager',ROOT/'scripts/package_hybrid_rc.py')
    trust=load('_dated_trust',ROOT/'scripts/verify_hybrid_rc_candidate.py')
    service=load('_dated_service',ROOT/'scripts/hybrid_service.py')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=ROOT/'local/dated-web-activation'/stamp;package=run/'package';package.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for info in packager.zip_members(z):
            target=package/info.filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
    manifest=manager.verify_package(package);trust.validate_trusted_sources(packager,package,manifest)
    if manifest['version']!='1.0.0-rc.15':raise ValueError('Expected qualified dated-webRC15')
    root,data,python,home=manager.registered_target()
    if root!=ROOT or manager.plan_files(package,root,manifest):raise ValueError('Serving disk differs from tested package')
    manager.assert_registered_owner_runtime(root,data);manager.validate_existing_assets(root,home,package)
    manager.protected_run_gate(root,manifest)
    previous=packager.predecessor_manifest(ROOT,json.loads((ROOT/'release/hybrid/payload-files.json').read_text(encoding='utf-8')))
    changed=[name for name in manifest['target_before_sha256'] if manifest['files']['payload/'+name]['sha256']!=previous['files']['payload/'+name]['sha256']]
    relative='src/scarx/hybrid/web_routing.py'
    if changed!=[relative]:raise ValueError('Hotfix scope must be exactly one public-routing module')
    old_archive=ROOT/manifest['predecessor']['archive']
    with zipfile.ZipFile(old_archive) as z:old_bytes=z.read('payload/'+relative)
    new_bytes=(ROOT/relative).read_bytes()
    if hashlib.sha256(old_bytes).hexdigest()!=manifest['target_before_sha256'][relative]:raise ValueError('Pinned rollback source')
    before=snapshot();(run/'private-before.npz').write_bytes(before)
    status=service.status()
    if status.get('cognition',{}).get('phase') in ('serving','observing'):raise RuntimeError('Foreground retains priority')
    backup=data/'release-backups'/stamp;backup.mkdir()
    lease=MaintenanceLease(data)
    stopped=False
    try:
        service.stop()
        stopped=True
        manager.snapshot_backup(data,backup)
        rollback=backup/'files'/relative;rollback.parent.mkdir(parents=True);rollback.write_bytes(old_bytes)
        records=[{'path':relative,'before_sha256':hashlib.sha256(old_bytes).hexdigest(),'after_sha256':hashlib.sha256(new_bytes).hexdigest()}]
        manager.atomic(backup/'files.json',json.dumps(records,indent=2).encode())
        manager.atomic(backup/'receipt.json',json.dumps({'version':manifest['version'],
            'activation_method':'restart of already edited and qualified registered source',
            'rollback_source':'exact pinned RC14 archive; not the on-disk candidate at activation',
            'source_git_sha':manifest['source_git_sha'],'files':records},indent=2).encode())
        manager.atomic(data/'release-backups/latest.json',json.dumps({'backup':str(backup)}).encode())
    except Exception:
        lease.close()
        if stopped:service.start()
        raise
    finally:lease.close()
    service.start();after_status=manager.wait_ready(service)
    after=snapshot();(run/'private-after.npz').write_bytes(after)
    with np.load(io.BytesIO(before),allow_pickle=False) as a,np.load(io.BytesIO(after),allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        arrays_equal=all(np.array_equal(a[k],b[k]) and a[k].dtype==b[k].dtype for k in a.files if k!='metadata')
        ma,mb=json.loads(a['metadata'].tobytes()),json.loads(b['metadata'].tobytes())
        learned_equal={k:v for k,v in ma.items() if k!='controller'}=={k:v for k,v in mb.items() if k!='controller'}
    if not arrays_equal or not learned_equal or after_status['neural']['execution_device']!='gpu':raise RuntimeError('State/device preservation failed')
    report={'version':manifest['version'],'source_git_sha':manifest['source_git_sha'],
        'archive':str(archive.relative_to(ROOT)),'activation_method':'qualified source restart',
        'changed_payload_files':changed,'full_saved_arrays_equal':arrays_equal,'learned_metadata_RNG_replay_equal':learned_equal,
        'updates':mb['updates'],'hidden_updates':mb['hidden_updates'],'gateway_pid':after_status['pid'],
        'GPU_placement':after_status['execution_placement'],'rollback_backup':str(backup),
        'model_weights_or_learning_rule_changed':False,'actual_Hermes_acceptance_pending':True,'research_quality_PASS':False}
    (run/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'local/current-dated-web-activation.txt').write_text(str(run),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
