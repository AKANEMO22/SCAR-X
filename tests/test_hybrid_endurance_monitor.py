import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "endurance_readonly_monitor",
    Path(__file__).resolve().parents[1] / "scripts/monitor_hybrid_endurance.py")
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def test_reused_pid_is_not_treated_as_the_original_running_lab(tmp_path, monkeypatch):
    run = tmp_path / "runs/owned"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"pid": 1234, "source_sha256": {}}))
    (run / "process-identity-v1.json").write_text(json.dumps({"process_created_epoch": 111}))
    data = tmp_path / "local/hybrid-v1"
    data.mkdir(parents=True)
    with sqlite3.connect(data / "hybrid.sqlite") as db:
        db.execute("CREATE TABLE events(id INTEGER,kind TEXT,payload TEXT)")
    process = SimpleNamespace(is_running=lambda: True, create_time=lambda: 222,
                              cmdline=lambda: ["python", "hybrid_shared_endurance.py"])
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    monkeypatch.setattr(monitor.psutil, "Process", lambda pid: process)
    result = monitor.inspect(run)
    assert result["actual_process_live"] is False
    assert result["state"] == "pid_missing_no_final_receipt"
    assert result["inspection_does_not_restart"]


def test_partial_heartbeat_write_is_retried_without_inventing_terminal_state(monkeypatch):
    attempts = iter(["{", '{"elapsed_seconds":31}'])
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: None)
    path = SimpleNamespace(read_text=lambda **kwargs: next(attempts))
    assert monitor.read_heartbeat(path)["elapsed_seconds"] == 31
    broken = SimpleNamespace(read_text=lambda **kwargs: "{")
    assert monitor.read_heartbeat(broken) is None
