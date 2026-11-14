"""Install a trusted local RC, preserve private state and record real startup.

No training, new architecture, weight conversion or research benchmark.
"""

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from urllib.request import urlopen
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def snapshot():
    database=(ROOT/'local/hybrid-v1/hybrid.sqlite').resolve()
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
        pointer=json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()[0])
        digest,payload=db.execute('SELECT digest,payload FROM checkpoints WHERE id=?',(pointer['id'],)).fetchone()
        cursor=db.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
    if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Private active snapshot integrity mismatch')
    return payload,cursor


def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);args=p.parse_args()
    archive=args.archive.resolve();sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive.with_suffix('.zip.sha256').read_text().split()[0]!=sha:raise ValueError('Archive checksum mismatch')
    packager=load('_rc_activate_packager',ROOT/'scripts/package_hybrid_rc.py')
    manager=load('_rc_activate_manager',ROOT/'release/hybrid/manage.py')
    checker=load('_rc_activate_trust',ROOT/'scripts/verify_hybrid_rc_candidate.py')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=ROOT/'local/release-activation'/stamp;package=run/'package';package.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in packager.zip_members(bundle):
            dest=package/member.filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(bundle.read(member))
    manifest=manager.verify_package(package);checker.validate_trusted_sources(packager,package,manifest)
    before,cursor=snapshot();(run/'private-before.npz').write_bytes(before)
    command=[sys.executable,'-B',str(package/'manage.py'),'--package',str(package)]
    checked=subprocess.run([*command,'--check'],capture_output=True,text=True,encoding='utf-8',timeout=30)
    (run/'preflight.stdout.log').write_text(checked.stdout,encoding='utf-8')
    (run/'preflight.stderr.log').write_text(checked.stderr,encoding='utf-8')
    if checked.returncode:raise RuntimeError('RC preflight refused; inspect '+str(run))
    with (run/'install.stdout.log').open('w',encoding='utf-8') as out,(run/'install.stderr.log').open('w',encoding='utf-8') as err:
        applied=subprocess.run(command,stdout=out,stderr=err,timeout=180)
    if applied.returncode:raise RuntimeError('RC activation failed; preserve backup/logs at '+str(run))
    result=json.loads((run/'install.stdout.log').read_text())
    with urlopen('http://127.0.0.1:8775/api/status?after=2147483647',timeout=10) as response:status=json.load(response)
    after,_=snapshot();(run/'private-after.npz').write_bytes(after)
    with np.load(io.BytesIO(before),allow_pickle=False) as old,np.load(io.BytesIO(after),allow_pickle=False) as new:
        if set(old.files)!=set(new.files):raise RuntimeError('Saved state fields changed during activation')
        arrays_equal=all(old[key].dtype==new[key].dtype and np.array_equal(old[key],new[key]) for key in old.files if key!='metadata')
        old_meta=json.loads(old['metadata'].tobytes());new_meta=json.loads(new['metadata'].tobytes())
        learned_equal={k:v for k,v in old_meta.items() if k!='controller'}=={k:v for k,v in new_meta.items() if k!='controller'}
    expected=('VLM_vision','ASR','ALIF_forward_learning')
    if (not status.get('ready') or status.get('failure') or not arrays_equal or not learned_equal
            or not all(status['execution_placement'][key]=='gpu' for key in expected)):
        raise RuntimeError('Postactivation state/device check failed; inspect private receipts at '+str(run))
    backup=Path(result['code_backup']).resolve()
    if not backup.is_relative_to(ROOT/'local/hybrid-v1/release-backups') or (backup/'private-full-checkpoint.bin').read_bytes()!=before:
        raise RuntimeError('Full private backup differs from preactivation checkpoint')
    for name,info in manifest['files'].items():
        if name.startswith('payload/') and hashlib.sha256((ROOT/name.removeprefix('payload/')).read_bytes()).hexdigest()!=info['sha256']:
            raise RuntimeError('Installed source differs from payload')
    receipt={'status':'RC2_INSTALLED_DEVICE_AND_PRIVATE_LEARNED_STATE_PRESERVED',
             'archive':str(archive.relative_to(ROOT)),'archive_sha256':sha,
             'source_git_sha':manifest['source_git_sha'],'version':manifest['version'],
             'gateway_pid':status['pid'],'updates':new_meta['updates'],'hidden_updates':new_meta['hidden_updates'],
             'all_saved_arrays_equal':arrays_equal,'learned_metadata_RNG_replay_equal':learned_equal,
             'full_private_backup_matches':True,'execution_placement':status['execution_placement'],
             'startup_event_cursor':cursor,'code_backup':str(backup),'native_startup_parity_calls_expected':3,
             'eight_hour_qualified':False,'Hermes_acceptance_pending':True,
             'research_or_production_PASS':False,'private_log_directory':str(run)}
    (run/'result.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
