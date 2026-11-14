"""High-impact release safety: source CAS, code rollback and private data exclusion."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import shutil
import zipfile
import psutil
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def manager():return load('hybrid_rc_manager_fixture',ROOT/'release/hybrid/manage.py')


@pytest.fixture
def packager():return load('hybrid_rc_packager_fixture',ROOT/'scripts/package_hybrid_rc.py')


def test_payload_policy_excludes_private_data_and_case_aliases(packager):
    for path in ['local/hybrid.sqlite','src/scarx/.env','runs/result.json','artifacts/private.bin',
                 'scripts/../local/config.json','integrations/hermes/hybrid_plugin/dashboard/settings.json',
                 'src/scarx/__pycache__/private.py','C:/private.py']:
        with pytest.raises(ValueError):packager.validate_payload_path(path)
    packager.validate_payload_path('src/scarx/hybrid/gateway.py')


@pytest.mark.parametrize('names',[['../escape.py'],['payload/a.py','PAYLOAD/A.PY'],['C:/private.py']])
def test_zip_reader_rejects_traversal_or_windows_case_aliases(packager,names):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive:
        for name in names:archive.writestr(name,b'owned fixture')
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
        with pytest.raises(ValueError):packager.zip_members(archive)


def test_package_tamper_and_unexpected_private_file_are_rejected(manager,tmp_path):
    payload=tmp_path/'payload';payload.mkdir();file=payload/'run_hybrid.py';file.write_bytes(b'owned source')
    manifest={'format':'scarx-hybrid-local-rc-v1','model_id':'scarx-hybrid',
              'files':{'payload/run_hybrid.py':{'sha256':manager.sha(file.read_bytes()),'bytes':len(file.read_bytes())}}}
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    assert manager.verify_package(tmp_path)
    file.write_bytes(b'changed source')
    with pytest.raises(ValueError,match='integrity'):manager.verify_package(tmp_path)
    file.write_bytes(b'owned source');(tmp_path/'private.env').write_bytes(b'PRIVATE_SENTINEL_NEVER_PACKAGE')
    with pytest.raises(ValueError,match='Unexpected'):manager.verify_package(tmp_path)


def test_live_protected_test_blocks_before_any_files_change(manager,tmp_path):
    run=tmp_path/'runs/protected';run.mkdir(parents=True);local=tmp_path/'local';local.mkdir()
    (local/'current-small-mixed-endurance-run.txt').write_text(str(run))
    (run/'manifest.json').write_text(json.dumps({'pid':os.getpid()}))
    (run/'process-identity-v1.json').write_text(json.dumps({'process_created_epoch':psutil.Process().create_time()}))
    marker=tmp_path/'keep.py';marker.write_bytes(b'original')
    with pytest.raises(RuntimeError,match='still running'):manager.protected_run_gate(tmp_path)
    assert marker.read_bytes()==b'original'


def test_mid_overlay_failure_restores_code_and_keeps_database(manager,tmp_path,monkeypatch):
    root=tmp_path/'target';root.mkdir();backup=tmp_path/'backup';backup.mkdir()
    (root/'a.py').write_bytes(b'old-A');(root/'b.py').write_bytes(b'old-B')
    with sqlite3.connect(root/'user.sqlite') as db:db.execute('CREATE TABLE memory(value TEXT)');db.execute("INSERT INTO memory VALUES('KEEP_USER_DATA')")
    original=manager.atomic;writes=[]
    def fail_once(path,raw):
        writes.append(path)
        if path==root/'b.py':raise OSError('controlled update failure')
        return original(path,raw)
    monkeypatch.setattr(manager,'atomic',fail_once)
    with pytest.raises(OSError):manager.write_overlay(root,backup,[('a.py',b'old-A',b'new-A'),('b.py',b'old-B',b'new-B')])
    assert (root/'a.py').read_bytes()==b'old-A' and (root/'b.py').read_bytes()==b'old-B'
    with sqlite3.connect(root/'user.sqlite') as db:assert db.execute('SELECT value FROM memory').fetchone()[0]=='KEEP_USER_DATA'


def test_code_rollback_retains_new_user_experiences(manager,tmp_path):
    root=tmp_path/'target';root.mkdir();backup=tmp_path/'backup';backup.mkdir();(root/'a.py').write_bytes(b'old')
    manager.write_overlay(root,backup,[('a.py',b'old',b'new')])
    with sqlite3.connect(root/'user.sqlite') as db:db.execute('CREATE TABLE memory(value TEXT)');db.execute("INSERT INTO memory VALUES('NEW_AFTER_RELEASE')")
    manager.rollback_files(root,backup)
    assert (root/'a.py').read_bytes()==b'old'
    with sqlite3.connect(root/'user.sqlite') as db:assert db.execute('SELECT value FROM memory').fetchone()[0]=='NEW_AFTER_RELEASE'


def test_rollback_refuses_later_edit_before_touching_any_other_file(manager,tmp_path):
    root=tmp_path/'target';root.mkdir();backup=tmp_path/'backup';backup.mkdir()
    (root/'a.py').write_bytes(b'A');(root/'b.py').write_bytes(b'B')
    manager.write_overlay(root,backup,[('a.py',b'A',b'new-A'),('b.py',b'B',b'new-B')])
    (root/'b.py').write_bytes(b'USER_LATER_EDIT')
    with pytest.raises(RuntimeError,match='later source edit'):manager.rollback_files(root,backup)
    assert (root/'a.py').read_bytes()==b'new-A' and (root/'b.py').read_bytes()==b'USER_LATER_EDIT'


def test_checkpoint_backup_pins_existing_payload_without_rewinding_journal(manager,tmp_path):
    data=tmp_path/'data';data.mkdir();backup=tmp_path/'backup';backup.mkdir();payload=b'owned checkpoint transport fixture; not a neuralquality test'
    with sqlite3.connect(data/'hybrid.sqlite') as db:
        db.executescript('CREATE TABLE objects(kind TEXT,id TEXT,payload TEXT,PRIMARY KEY(kind,id));CREATE TABLE checkpoints(id TEXT,digest TEXT,payload BLOB);CREATE TABLE events(id INTEGER,kind TEXT);')
        db.execute('INSERT INTO objects VALUES(?,?,?)',('runtime','active_checkpoint',json.dumps({'id':'owned'})))
        db.execute('INSERT INTO checkpoints VALUES(?,?,?)',('owned',manager.sha(payload),payload));db.execute("INSERT INTO events VALUES(1,'KEEP_JOURNAL')")
    result=manager.snapshot_backup(data,backup)
    assert result['private_checkpoint_sha256']==manager.sha(payload)
    assert (backup/'private-full-checkpoint.bin').read_bytes()==payload
    with sqlite3.connect(data/'hybrid.sqlite') as db:
        assert db.execute('SELECT kind FROM events').fetchone()[0]=='KEEP_JOURNAL'
        assert json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime'").fetchone()[0])=={'id':'owned'}
        assert db.execute("SELECT COUNT(*) FROM objects WHERE kind='checkpoint_pin'").fetchone()[0]==1


@pytest.mark.parametrize('tamper',['audit','candidate','baseline'])
def test_overlay_requires_exact_qualified_recipe_and_audit(packager,tmp_path,tamper):
    targets=packager.overlay_candidates(ROOT)[1]
    for target,entry in targets.items():
        audit=entry['audit_reference']
        candidate=str(Path(audit).parent/('candidate-store.py' if target.endswith('store.py') else 'candidate-engine.py'))
        recipe='integrations/hermes/'+('full_query_retrieval_patch.py' if target.endswith('store.py') else 'verifier_capability_guard_patch.py')
        archived=str(Path(audit).parent/'source'/target)
        for relative in (target,audit,candidate,recipe,archived):
            destination=tmp_path/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,destination)
    assert packager.overlay_candidates(tmp_path)[0]==packager.overlay_candidates(ROOT)[0]
    entry=targets['src/scarx/hybrid/store.py']
    if tamper=='audit':
        path=tmp_path/entry['audit_reference'];report=json.loads(path.read_text())
        report['verdict']='NOT_VERIFIED';path.write_text(json.dumps(report))
    elif tamper=='candidate':
        path=tmp_path/Path(entry['audit_reference']).parent/'candidate-store.py'
        path.write_bytes(path.read_bytes()+b'\n# changed after P0\n')
    else:
        path=tmp_path/'src/scarx/hybrid/store.py';path.write_bytes(path.read_bytes()+b'\n# concurrent edit\n')
    with pytest.raises(ValueError):packager.overlay_candidates(tmp_path)


@pytest.mark.parametrize('tamper',['manage.py','payload/src/scarx/hybrid/engine.py'])
def test_self_consistent_forged_manifest_is_not_executed(manager,packager,tmp_path,tamper):
    checker=load('hybrid_rc_candidate_trust_fixture',ROOT/'scripts/verify_hybrid_rc_candidate.py')
    allowed=json.loads((ROOT/'release/hybrid/payload-files.json').read_text())['files']
    overlays,_=packager.overlay_candidates(ROOT)
    for name in allowed:
        path=tmp_path/'payload'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(overlays[name] if name in overlays else (ROOT/name).read_bytes())
    (tmp_path/'manage.py').write_bytes((ROOT/'release/hybrid/manage.py').read_bytes())
    changed=tmp_path/tamper;changed.write_bytes(changed.read_bytes()+b'\n# UNREVIEWED EXECUTABLE FIXTURE\n')
    manifest={'format':'scarx-hybrid-local-rc-v1','model_id':'scarx-hybrid','files':{
        p.relative_to(tmp_path).as_posix():{'sha256':manager.sha(p.read_bytes()),'bytes':p.stat().st_size}
        for p in tmp_path.rglob('*') if p.is_file()}}
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    # Checksums agree with the forgery; the separately trusted source rejects it.
    manager.verify_package(tmp_path)
    with pytest.raises(ValueError,match='trusted'):
        checker.validate_trusted_sources(packager,tmp_path,manifest)


@pytest.mark.parametrize('tamper',[None,'claim_PASS','source','identity'])
def test_explicit_user_cancellation_allows_RC_without_endurance_PASS(manager,tmp_path,tamper):
    pid=2147483647
    assert not psutil.pid_exists(pid)
    run=tmp_path/'runs/cancelled';run.mkdir(parents=True)
    (tmp_path/'local').mkdir();(tmp_path/'local/current-small-mixed-endurance-run.txt').write_text(str(run))
    (tmp_path/'core.py').write_bytes(b'unchanged core')
    manifest={'pid':pid,'source_sha256':{'core.py':manager.sha(b'unchanged core')}}
    (run/'manifest.json').write_text(json.dumps(manifest))
    (run/'process-identity-v1.json').write_text(json.dumps({'process_created_epoch':123.0}))
    request={'action':'cancel_8h_for_product_release','run_id':run.name,'pid':pid,
             'process_created_epoch':123.0,'eight_hour_PASS_claimed':False}
    result={'gate_pass':False,'formal_P1_P2_P3_or_lifelong_pass':False}
    if tamper=='claim_PASS':result['gate_pass']=True
    request_raw=json.dumps(request).encode();result_raw=json.dumps(result).encode()
    (run/'user-cancellation-request-v1.json').write_bytes(request_raw)
    (run/'result.json').write_bytes(result_raw)
    audit={'run_id':run.name,'pid':pid,'process_created_epoch':123.0,
           'verdict':'USER_CANCELLED_NOT_EIGHT_HOUR_PASS','actual_worker_terminal':True,
           'source_matches_before_activation':True,'residual_private_worker_pids':[],
           'request_sha256':manager.sha(request_raw),'result_sha256':manager.sha(result_raw),
           'engineering_eight_hour_qualified':False}
    if tamper=='identity':audit['process_created_epoch']=456.0
    if tamper=='source':(tmp_path/'core.py').write_bytes(b'concurrent change')
    (run/'user-cancellation-audit-v1.json').write_text(json.dumps(audit))
    if tamper is None:
        gate=manager.protected_run_gate(tmp_path)
        assert gate['release_gate']=='user_cancelled_8h_for_RC' and gate['eight_hour_qualified'] is False
    else:
        with pytest.raises(RuntimeError):manager.protected_run_gate(tmp_path)
