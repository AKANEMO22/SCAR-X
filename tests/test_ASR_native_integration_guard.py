import importlib.util
import json
from pathlib import Path
import pytest


def test_running_original_endurance_prevents_GPU_allocation_or_attempted_calls(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/accept_hybrid_ASR_profile.py'
    spec=importlib.util.spec_from_file_location('ASR_native_guard_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'owned';run=root/'runs/locked';run.mkdir(parents=True)
    (run/'manifest.json').write_text(json.dumps({'run_id':'locked'}))
    monkeypatch.setattr(module,'ROOT',root)
    monkeypatch.setattr(module,'inspect_endurance',lambda run:{'actual_process_live':True,'final_verdict':None})
    monkeypatch.setattr(module,'inspect_soak',lambda run:{'actual_process_live':False,'final_verdict':'component_only'})
    def forbidden(*args,**kwargs):raise AssertionError('Live job guard must stop before model/process/budget action')
    monkeypatch.setattr(module,'SharedNative',forbidden)
    monkeypatch.setattr(module,'charge',forbidden)
    monkeypatch.setattr(module,'gpu_headroom',forbidden)
    with pytest.raises(RuntimeError,match='still live'):module.execute(run)
    assert not (run/'private-started.json').exists()
    assert not (root/'local').exists()
