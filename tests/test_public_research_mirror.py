from pathlib import Path
import importlib.util
import pytest


def module():
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    spec=importlib.util.spec_from_file_location('isolated_public_mirror',scripts/'public_research_mirror.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_evidence_publishing_cannot_write_to_a_main_checkout(tmp_path,monkeypatch):
    m=module();project=tmp_path/'scar-x';project.mkdir();(project/'.git').mkdir();(project/'run_app.py').write_text('')
    research=project/'local/worktrees/research';run=research/'runs/20261111-data-audit-abcdef';run.mkdir(parents=True)
    (run/'manifest.json').write_text('{}');(run/'result.json').write_text('{"measured":true}')
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'main\n')
    result=m.publish(run)
    assert result['mirror'] is None and result['sqlite_written'] is False
    assert not (project/'runs').exists() and not (project/'local/experiment-status').exists()


def test_publishing_refuses_to_replace_an_unowned_historical_run(tmp_path,monkeypatch):
    m=module();project=tmp_path/'scar-x';project.mkdir();(project/'.git').mkdir();(project/'run_app.py').write_text('')
    research=project/'local/worktrees/research';run=research/'runs/20261111-data-audit-abcdef';run.mkdir(parents=True)
    (run/'manifest.json').write_text('{}')
    target=project/'runs'/run.name;target.mkdir(parents=True);(target/'result.json').write_text('original')
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'research-observer\n')
    with pytest.raises(ValueError,match='unowned'):m.publish(run)
    assert (target/'result.json').read_text()=='original'
