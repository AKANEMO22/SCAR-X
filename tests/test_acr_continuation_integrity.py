"""Protect an interrupted study against silent update/evaluation/budget changes."""
import ast,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

def job(filename):
    tree=ast.parse((ROOT/'scripts'/filename).read_text(encoding='utf-8'))
    return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='job')

@pytest.mark.parametrize('name',['run_acr_resume.py','run_acr_resume_v2.py'])
def test_resumed_training_and_evaluation_body_is_original(name):
    original=job('run_acr_dev.py');resumed=job(name)
    guards=[n for n in resumed.body if isinstance(n,ast.If) and ast.unparse(n.test)=='folder.exists()']
    assert len(guards)==1
    resumed.body.remove(guards[0])
    assert ast.dump(original,include_attributes=False)==ast.dump(resumed,include_attributes=False)

def test_actual_continuation_manifest_preserves_budget_and_all_reused_files():
    run=ROOT/'runs/20261112T082034Z-acr-dev-v1-resume-ac9bba'
    m=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    total=m['limit_seconds']+m['prior_elapsed_lower_bound']+m['shutdown_allowance_seconds']+m['additional_preflight_allowance_seconds']
    assert abs(total-9000)<1e-8
    assert m['reused_completed_cells']==55 and len(m['reused_files_sha256'])==220
    assert m['confirmation'] is False
