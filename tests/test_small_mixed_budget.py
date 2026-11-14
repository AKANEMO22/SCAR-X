import json
import pytest
from scarx.evaluation.mixed_call_budget import charge, LIMITS


def test_phase_ledgers_keep_failed_attempts_and_do_not_reset_through_another_phase(tmp_path):
    path = tmp_path / "ledger.json"
    for count in range(12):
        assert charge(path, "preflight", "ASR") == count + 1
    with pytest.raises(RuntimeError, match="exhausted"):
        charge(path, "preflight", "ASR")
    assert charge(path, "long", "ASR") == 1
    assert charge(path, "preflight", "VLM") == 1
    assert json.loads(path.read_text())["attempted"]["preflight"]["ASR"] == 12


def test_corrupt_or_concurrent_ledger_is_not_replaced(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text('{"limits":{}}')
    original = path.read_bytes()
    with pytest.raises(ValueError, match="no reset"):
        charge(path, "preflight", "ASR")
    assert path.read_bytes() == original
    lock = path.with_suffix(".lock")
    lock.write_text("unit owned other writer")
    with pytest.raises(FileExistsError):
        charge(path, "preflight", "ASR")
    assert lock.read_text() == "unit owned other writer"
