"""Owned-file activation transaction fixtures; no live files or neural workers."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


@pytest.fixture
def world(tmp_path,monkeypatch):
    installer=module('ASR_activation_fixture',ROOT/'scripts/install_hybrid_ASR_profile.py')
    recipe=module('ASR_activation_reviewed_recipe',ROOT/'integrations/hermes/asr_profile_integration_patch.py')
    resource=module('ASR_activation_reviewed_roots',ROOT/'integrations/hermes/resource_client_roots_patch.py')
    owned=tmp_path/'owned';live=owned/'local/hybrid-v1';live.mkdir(parents=True)
    run=owned/'runs/locked';run.mkdir(parents=True)
    identities={};originals={}
    # Importable candidate behavior is tested elsewhere. Here only exact
    # reviewedbytes are copied to owned paths for filetransaction behavior.
    for filename in (*recipe.EXPECTED,'resources.py'):
        path=owned/'src/scarx/hybrid'/filename;path.parent.mkdir(parents=True,exist_ok=True)
        original=ROOT/'runs/20261114T011841Z-shared-endurance-v1-554c19/source/src/scarx/hybrid'/filename
        raw=original.read_bytes();path.write_bytes(raw);originals[path]=raw
        if filename in recipe.EXPECTED:identities[filename]=hashlib.sha256(recipe.patch_bytes(filename,raw)).hexdigest()
    config=live/'config.json';config.write_text(json.dumps({'execution_mode':'gpu','gpu_device':0}));originals[config]=config.read_bytes()
    (run/'manifest.json').write_text(json.dumps({'candidate_source_sha256':identities}))
    (run/'private-audit-v1.json').write_text(json.dumps({'verdict':'ACTUAL_SELECTED_ASR_PRIVATE_P0_DATA_AUDIT_PASS_ONLY','actual_ASR_calls':4}))
    (owned/'local/ASR-native-integration-p0-v1-budget.json').write_text(json.dumps({'limits':{'ASR':4,'VLM':4},'attempted':{'ASR':4,'VLM':0}}))
    payload=b'owned private backup transport fixture, not a neuralcheckpoint'
    with sqlite3.connect(live/'hybrid.sqlite') as db:
        db.executescript('CREATE TABLE objects(kind TEXT,id TEXT,payload TEXT);CREATE TABLE checkpoints(id TEXT,digest TEXT,payload BLOB);')
        db.execute('INSERT INTO objects VALUES(?,?,?)',('runtime','active_checkpoint',json.dumps({'id':'owned'})))
        db.execute('INSERT INTO checkpoints VALUES(?,?,?)',('owned',hashlib.sha256(payload).hexdigest(),payload))
    monkeypatch.setattr(installer,'ROOT',owned)
    monkeypatch.setattr(installer,'prerequisites',lambda:{'scope':'unit fixture, not actual8hqualification'})
    monkeypatch.setattr(installer,'load_recipe',lambda name,filename:recipe if filename=='asr_profile_integration_patch.py' else resource)
    return installer,run,live,originals


def test_live_writer_lock_prevents_any_source_activation(world):
    installer,run,live,originals=world
    writer=installer.OwnerLock(live)
    try:
        with pytest.raises(RuntimeError,match='owns this data'):installer.main(run,True)
        assert all(path.read_bytes()==raw for path,raw in originals.items())
    finally:writer.close()


def test_mid_transaction_failure_restores_only_own_written_sources_and_keeps_backup(world,monkeypatch):
    installer,run,live,originals=world
    write=installer.atomic_write;calls=[]
    def fail_once(path,raw):
        calls.append(path)
        if len(calls)==3:raise OSError('Controlled unitfixture file replacement failure')
        return write(path,raw)
    monkeypatch.setattr(installer,'atomic_write',fail_once)
    with pytest.raises(OSError,match='Controlled'):installer.main(run,True)
    assert all(path.read_bytes()==raw for path,raw in originals.items())
    assert not (live/'runtime.lock').exists()
    assert list((live/'profile-backups').glob('*/private-full-live-checkpoint.bin'))
    assert not (run/'source-activation-v1.json').exists()
