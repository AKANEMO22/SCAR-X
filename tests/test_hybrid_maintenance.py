import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from scarx.runtime.maintenance import MaintenanceLease, check_startup


def test_live_maintenance_blocks_a_distinct_autostart_process_until_released(tmp_path):
    lease = MaintenanceLease(tmp_path)
    program = "from scarx.runtime.maintenance import check_startup; import sys; check_startup(sys.argv[1])"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    blocked = subprocess.run([sys.executable, "-c", program, str(tmp_path)],
                             capture_output=True, text=True, timeout=10, env=env)
    assert blocked.returncode != 0 and "maintenance transaction" in blocked.stderr
    with pytest.raises(RuntimeError, match="Another live process"):
        MaintenanceLease(tmp_path)
    lease.close()
    allowed = subprocess.run([sys.executable, "-c", program, str(tmp_path)],
                             capture_output=True, text=True, timeout=10, env=env)
    assert allowed.returncode == 0


def test_reused_pid_is_not_a_live_owner_and_only_exact_stale_lease_is_replaced(tmp_path):
    path = tmp_path / ".startup-maintenance.lock"
    path.write_text(json.dumps({"pid": os.getpid(), "created": 1, "nonce": "a" * 32}))
    other = tmp_path / "keep.txt"
    other.write_text("keep")
    check_startup(tmp_path)
    lease = MaintenanceLease(tmp_path)
    assert json.loads(path.read_text())["nonce"] == lease.nonce
    lease.close()
    assert other.read_text() == "keep"


def test_corrupt_or_partial_lease_fails_closed_without_deleting_it(tmp_path):
    path = tmp_path / ".startup-maintenance.lock"
    path.write_bytes(b"{")
    with pytest.raises(RuntimeError, match="Invalid"):
        check_startup(tmp_path)
    with pytest.raises(RuntimeError, match="Invalid"):
        MaintenanceLease(tmp_path)
    assert path.read_bytes() == b"{"


def test_closing_a_lease_never_removes_a_replacement_owner(tmp_path):
    lease = MaintenanceLease(tmp_path)
    path = tmp_path / ".startup-maintenance.lock"
    value = json.loads(path.read_text())
    value["nonce"] = "b" * 32
    path.write_text(json.dumps(value))
    lease.close()
    assert path.exists() and json.loads(path.read_text())["nonce"] == "b" * 32
