"""Consolidation integrity, not a quality/training experiment."""
from pathlib import Path
import hashlib,importlib.util,json,subprocess,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_catalog_has_protected_active_experiment_and_scoped_results():
    d=json.loads((ROOT/'docs/research/experiment-catalog.json').read_text(encoding='utf-8'))
    ids={v['id']:v for v in d['entries']}
    assert len(ids)==len(d['entries'])==120
    assert ids['R13-1-ACTIVE']['status']=='IN_PROGRESS_EXCLUDED'
    assert ids['R13-1-ACTIVE']['reports']==[]
    assert ids['CODEX-R8-CONFIRMATION']['status']=='P1_FAIL_RESEARCH_ONLY'
    assert ids['CAPACITY-H64-H128']['status']=='CAPACITY_ONLY_SIGNAL'
    assert ids['CODEX-ACR']['status']=='ACR_SPECIALIZED_GAIN'
    assert ids['GEMINI-R13-SNAPSHOT']['status']=='ARCHIVE_ONLY_INVALID_CONTROLS'

def test_catalog_and_original_source_digests_are_valid():
    subprocess.run([sys.executable,str(ROOT/'scripts/research/query_catalog.py'),'verify'],cwd=ROOT,check=True,capture_output=True)
    d=json.loads((ROOT/'docs/research/archive/source-index.json').read_text(encoding='utf-8'))
    assert len(d['files'])==227 and len(d['branches'])==27
    assert all('/original/' in v['local_path'] for v in d['files'])

def test_cleanup_target_guard_blocks_primary_and_active_worktree(tmp_path):
    file=ROOT/'scripts/research/archive_workspace.py'
    spec=importlib.util.spec_from_file_location('archive_workspace_guard',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    repo=tmp_path/'scar-x';repo.mkdir();active=tmp_path/'scar-x-gemini-r7';active.mkdir();(active/'.git').write_text('gitdir: placeholder')
    with pytest.raises(ValueError):module.safe_target(repo,repo,{active})
    with pytest.raises(ValueError):module.safe_target(repo,active,{active})
    outside=tmp_path/'unrelated-font-project';outside.mkdir();(outside/'.git').write_text('gitdir: placeholder')
    with pytest.raises(ValueError):module.safe_target(repo,outside,{active})
    ended=tmp_path/'scar-x-old-round';ended.mkdir();(ended/'.git').write_text('gitdir: placeholder')
    assert module.safe_target(repo,ended,{active})==ended.resolve()

def test_production_source_unchanged_from_precleanup_main():
    paths=['src/scarx/neural','src/scarx/app','src/scarx/learning','integrations','run_app.py','capabilities.json','project.json']
    r=subprocess.run(['git','diff','--exit-code','e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541','--',*paths],cwd=ROOT,capture_output=True)
    assert r.returncode==0,r.stdout.decode(errors='replace')
    # New isolated research registrations may be added to configs/, but none
    # of the existing serving configurations may be modified or deleted.
    r=subprocess.run(['git','diff','--name-only','--diff-filter=MD','e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541','--','configs'],cwd=ROOT,capture_output=True)
    assert r.returncode==0 and not r.stdout.strip()

def test_combined_snapshot_formats_roundtrip_without_kind_collision():
    from scarx.neural.core import Config
    from scarx.evaluation.capacity import CapacityACR
    from scarx.evaluation.literature_eprop import LiteratureEPropALIF
    from scarx.evaluation.literature_braintrace import BrainTraceDRTRL
    for cls,h in ((CapacityACR,128),(LiteratureEPropALIF,48),(BrainTraceDRTRL,48)):
        model=cls(Config(inputs=8,outputs=2,hidden=h,fan_in=8,plastic_rows=12))
        raw=model.dumps();assert cls.loads(raw).dumps()==raw
