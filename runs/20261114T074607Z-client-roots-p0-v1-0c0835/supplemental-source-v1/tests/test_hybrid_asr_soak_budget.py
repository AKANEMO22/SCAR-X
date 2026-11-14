import json
import pytest
from scarx.evaluation.asr_soak_budget import charge_attempt


def test_budget_lock_is_closed_and_released_on_success_and_exhaustion(tmp_path):
    ledger=tmp_path/'budget.json'
    assert charge_attempt(ledger)==1
    assert charge_attempt(ledger)==2
    assert not ledger.with_suffix('.lock').exists()
    ledger.write_text(json.dumps({'attempted':255,'cap':256}))
    assert charge_attempt(ledger)==256
    with pytest.raises(RuntimeError):charge_attempt(ledger)
    assert json.loads(ledger.read_text())['attempted']==256
    assert not ledger.with_suffix('.lock').exists()


def test_budget_corruption_is_not_silently_reset(tmp_path):
    ledger=tmp_path/'budget.json';ledger.write_text(json.dumps({'attempted':-1,'cap':256}))
    with pytest.raises(ValueError):charge_attempt(ledger)
    assert json.loads(ledger.read_text())['attempted']==-1
    assert not ledger.with_suffix('.lock').exists()
