"""Declared audit-event fixtures only; no subprocess, network or GPU calls."""
import hashlib
from pathlib import Path
import pytest
from scarx.evaluation.bridge_effect_guard import StickyEffectGuard
from scarx.evaluation.bridge_environment_metrics import GPU_QUERY_ARGS,parse_gpu_memory


def test_gethostname_is_only_a_local_metadata_query():
    guard=StickyEffectGuard();guard.hook('socket.gethostname',());guard.check()
    assert not guard.violations and len(guard.allowed_queries)==1


@pytest.mark.parametrize('event,args',[
    ('socket.connect',(None,('example.invalid',443))),
    ('socket.getaddrinfo',('example.invalid',443,0,0,0)),
    ('subprocess.Popen',('unknown.exe',['unknown.exe'],None,None)),
])
def test_swallowed_permission_error_still_stops_at_next_boundary(event,args):
    guard=StickyEffectGuard()
    try:guard.hook(event,args)
    except PermissionError:pass # Simulate the SDK fallback, without executing it.
    with pytest.raises(RuntimeError):guard.check()
    assert guard.violations[0]['event']==event


def test_only_pinned_source_executable_and_exact_ver_command_can_match(tmp_path):
    source=tmp_path/'platform.py';source.write_text('def _syscmd_ver(): pass\n')
    executable=tmp_path/'cmd.exe';executable.write_bytes(b'declared unit fixture, never executed')
    guard=StickyEffectGuard(source,hashlib.sha256(source.read_bytes()).hexdigest(),
                            executable,hashlib.sha256(executable.read_bytes()).hexdigest())
    frames=[{'filename':str(source),'function':'_syscmd_ver','line':1}]
    command=str(executable.resolve())+' /c "ver"'
    args=(str(executable),command,None,None)
    assert guard._ver_probe_allowed(args,frames)
    assert not guard._ver_probe_allowed((str(executable),command+' & other.exe',None,None),frames)
    assert not guard._ver_probe_allowed(args,[{'filename':str(source),'function':'untrusted','line':1}])
    source.write_text('changed source')
    assert not guard._ver_probe_allowed(args,frames)


def test_telemetry_exception_is_limited_to_exact_query_and_pinned_source(tmp_path):
    source=tmp_path/'monitor.py';source.write_text('def sample_physical_gpu(): pass\n')
    executable=tmp_path/'nvidia-smi.exe';executable.write_bytes(b'unit fixture; not executed')
    guard=StickyEffectGuard(monitor_source=source,monitor_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
        gpu_executable=executable,gpu_sha=hashlib.sha256(executable.read_bytes()).hexdigest())
    frames=[{'filename':str(source),'function':'sample_physical_gpu','line':1}]
    argv=[str(executable.resolve()),*GPU_QUERY_ARGS]
    assert guard._gpu_probe_allowed((str(executable),argv,None,None),frames)
    assert not guard._gpu_probe_allowed((str(executable),argv+['--query-compute-apps=pid'],None,None),frames)
    assert parse_gpu_memory('2854, 6141\n')['free_bytes']==2854*1024**2
    for text in ('-1, 6141','7000, 6141','0, 0','1,2\n3,4'):
        with pytest.raises(ValueError):parse_gpu_memory(text)


def test_missing_probe_pin_cannot_turn_a_swallowed_error_into_a_guard_bypass(tmp_path):
    source=tmp_path/'platform.py';source.write_text('declared source')
    missing=tmp_path/'missing-cmd.exe'
    guard=StickyEffectGuard(source,hashlib.sha256(source.read_bytes()).hexdigest(),missing,'wrong')
    try:guard.hook('subprocess.Popen',(str(missing),str(missing)+' /c "ver"',None,None))
    except PermissionError:pass
    with pytest.raises(RuntimeError):guard.check()
