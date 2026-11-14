"""Atomic per-file exact-source activation; own writerlock, private state backups."""

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from scarx.hybrid.ownership import OwnerLock
from scarx.runtime.asr_profiles import PROFILES
from scarx.evaluation.integration_call_budget import LIMITS
from accept_hybrid_ASR_profile import prerequisites


def digest(raw):return hashlib.sha256(raw).hexdigest()


def load_recipe(name,filename):
    path=ROOT/'integrations/hermes'/filename
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def atomic_write(path,raw):
    fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.ASR-profile-',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def main(run,apply):
    run=run.resolve()
    if not run.is_relative_to(ROOT/'runs'):raise ValueError('Expected owned integrationrun')
    evidence=prerequisites()
    private=json.loads((run/'private-audit-v1.json').read_text(encoding='utf-8'))
    if private['verdict']!='ACTUAL_SELECTED_ASR_PRIVATE_P0_DATA_AUDIT_PASS_ONLY' or private['actual_ASR_calls']!=4:
        raise RuntimeError('Private nativeASR acceptance missing/failed')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    recipe=load_recipe('ASR_exactsource_activation','asr_profile_integration_patch.py')
    root_recipe=load_recipe('clientroots_exactsource_activation','resource_client_roots_patch.py')
    before={};after={}
    for filename,expected in recipe.EXPECTED.items():
        target=ROOT/'src/scarx/hybrid'/filename
        raw=target.read_bytes()
        if digest(raw)!=expected:raise RuntimeError('Current source changed; refuse overwriting '+filename)
        changed=recipe.patch_bytes(filename,raw)
        if digest(changed)!=manifest['candidate_source_sha256'][filename]:raise RuntimeError('Reviewed candidate source mismatch')
        before[target]=raw;after[target]=changed
    resource=ROOT/'src/scarx/hybrid/resources.py'
    before[resource]=resource.read_bytes();after[resource]=root_recipe.patch_bytes(before[resource])
    live=ROOT/'local/hybrid-v1';config=live/'config.json';raw=config.read_bytes()
    settings=json.loads(raw.decode('utf-8-sig'))
    if settings.get('execution_mode')!='gpu':raise ValueError('This profileactivation only qualifies currentGPU mode')
    profile=PROFILES['openai/whisper-small']
    selected={**settings,'asr_model_name':profile.name,'asr_revision':profile.revision}
    before[config]=raw;after[config]=(json.dumps(selected,ensure_ascii=False,indent=2)+'\n').encode()
    budget=json.loads((ROOT/'local/ASR-native-integration-p0-v1-budget.json').read_text(encoding='utf-8'))
    if budget.get('limits')!=LIMITS or budget['attempted']['ASR']!=4 or budget['attempted']['VLM']!=0:
        raise ValueError('Integration budget state does not permit activation/parity stage')
    if not apply:
        print(json.dumps({'reviewed_exactsource_targets':len(after),'ASR_model':profile.identity(),
            'applied':False,'live_service_needs_graceful_stop_before_apply':True}));return
    lock=OwnerLock(live) # Only succeeds after the real gateway has released this writerlock.
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=live/'profile-backups'/stamp;backup.mkdir(parents=True)
    written=[]
    try:
        for target,raw in before.items():
            expected_root=ROOT/'src/scarx/hybrid' if target!=config else live
            if not target.resolve().is_relative_to(expected_root.resolve()):raise ValueError('Activation path escapes intended owned directory')
            destination=backup/target.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
        database=(live/'hybrid.sqlite').resolve()
        with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
            active=json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()[0])
            stored=db.execute('SELECT digest,payload FROM checkpoints WHERE id=?',(active['id'],)).fetchone()
        if digest(stored[1])!=stored[0]:raise RuntimeError('Private livecheckpointbackup hash invalid')
        (backup/'private-full-live-checkpoint.bin').write_bytes(stored[1])
        for target in after:
            if target.read_bytes()!=before[target]:raise RuntimeError('Concurrent source/configedit; activation refused')
        for target,raw in after.items():
            if target.read_bytes()!=before[target]:raise RuntimeError('Concurrent write during source activation')
            atomic_write(target,raw);written.append(target)
        receipt={'installed_UTC':stamp,'run_id':run.name,'private_backup':str(backup),
            'prerequisites':evidence,'targets':{str(path.relative_to(ROOT)):{'before_sha256':digest(before[path]),'after_sha256':digest(raw)} for path,raw in after.items()},
            'saved_private_live_checkpoint_sha256':stored[0],'selected_ASR':profile.identity(),
            'service_restarted_here':False,'model_or_controller_weights_modified':False}
        (backup/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        (run/'source-activation-v1.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps({'source_and_config_applied':True,'targets':len(written),'backup':str(backup),
            'service_restarted_here':False,'private_checkpoint_hash':stored[0]}))
    except Exception:
        for target in reversed(written):
            if target.read_bytes()==after[target]:atomic_write(target,before[target])
        raise
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();main(args.run,args.apply)
