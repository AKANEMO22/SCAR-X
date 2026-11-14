"""Capture fault isolation; fixtures do not measure language/video intelligence."""

import ast
import builtins
import json
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace

from PIL import Image
import pytest

from scarx.hybrid.screen import ScreenMemory
from scarx.hybrid.store import Store


@pytest.fixture
def screen(tmp_path):
    store = Store(tmp_path)
    observations = []
    engine = SimpleNamespace(
        store=store, foreground=threading.Event(),
        monitor=SimpleNamespace(check=lambda: None),
        backend=SimpleNamespace(
            transcribe=lambda _: 'engineering fixture text',
            describe=lambda *a, **kw: {'text': 'engineering fixture frame'},
        ),
        observe=lambda text, **kw: observations.append({'text': text, **kw}),
    )
    memory = ScreenMemory(engine)
    owned = tmp_path / 'archive' / ('d' * 32)
    owned.mkdir(parents=True)
    state = {'id': 'd' * 32, 'kind': 'screen', 'status': 'sharing',
             'capture_frontier': 0, 'vision_frontier': 0, 'asr_frontier': 0,
             'audio_available': True, 'asr_status': 'waiting', 'latest_frame': None,
             'error': None}
    memory.heartbeats[state['id']] = threading.Event()
    try:
        yield memory, state, owned, observations
    finally:
        store.close()


def row(name='audio.wav', start=0, end=10):
    return {'kind': 'audio', 'path': name, 'start': start, 'end': end}


def test_actual_ffmpeg_audio_decode_failure_keeps_frame_ingestion_alive(screen):
    memory, state, owned, observed = screen
    assert shutil.which('ffmpeg'), 'Qualified runtime requires real FFmpeg for this check'
    (owned / 'corrupt.wav').write_bytes(b'owned invalid audio fixture, not a WAV')
    Image.new('RGB', (24, 24), 'red').save(owned / 'frame.jpg')
    rows = [row('corrupt.wav'), {'kind': 'frame', 'time': 12, 'path': 'frame.jpg'}]
    (owned / 'capture.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    process = SimpleNamespace(poll=lambda: 0, returncode=0,
                              terminate=lambda: pytest.fail('Audio failure must not terminate capture'))
    memory._ingest(state, process, owned)
    assert state['status'] == 'stopped' and state['error'] is None
    assert state['capture_frontier'] == state['vision_frontier'] == 12
    assert state['asr_frontier'] == 0 and state['degraded']
    assert [item['modality'] for item in observed] == ['sampled_frame']
    event = next(e for e in memory.store.events() if e['kind'] == 'screen_asr_failed')
    assert event['payload']['transcript_not_created']
    assert event['payload']['observation_is_training_label'] is False


def test_next_audio_chunk_can_recover_without_bridging_a_failed_interval(screen, monkeypatch):
    memory, state, owned, observed = screen
    monkeypatch.setattr('scarx.hybrid.screen.subprocess.run', lambda *a, **kw: None)
    calls = []
    def transcribe(_):
        calls.append(True)
        if len(calls) == 1:
            raise TimeoutError('owned timeout fixture')
        return 'engineering fixture text'
    memory.engine.backend.transcribe = transcribe
    previous = memory._ingest_audio(state, row(), owned, None)
    assert previous is None and state['asr_frontier'] == 0
    previous = memory._ingest_audio(state, row('next.wav', 10, 20), owned, previous)
    assert previous == ((owned / 'next.wav').resolve(), 10)
    assert state['asr_frontier'] == 20 and state['asr_status'] == 'ready'
    assert state['consecutive_audio_errors'] == 0 and state['degraded']
    assert observed[0]['start'] == 10 and observed[0]['end'] == 20
    assert [(x['start'], x['end']) for x in state['asr_errors']] == [(0, 10)]


def test_three_audio_failures_pause_only_audio_and_bound_retry_work(screen, monkeypatch):
    memory, state, owned, observed = screen
    calls = []
    def fail(*a, **kw):
        calls.append(True)
        raise subprocess.CalledProcessError(1, ['ffmpeg'])
    monkeypatch.setattr('scarx.hybrid.screen.subprocess.run', fail)
    previous = None
    for i in range(5):
        previous = memory._ingest_audio(state, row(start=i * 10, end=(i + 1) * 10), owned, previous)
    assert len(calls) == 3 and state['asr_status'] == 'paused_after_errors'
    assert state['asr_frontier'] == 0 and not observed
    assert len(state['asr_errors']) == 3 and len(state['audio_skipped_intervals']) == 2
    memory.store.put('stream', state['id'], state)
    assert memory.store.get('stream', state['id'])['asr_status'] == 'paused_after_errors'


def test_foreground_preemption_records_gap_without_penalizing_audio_or_retrying(screen, monkeypatch):
    memory, state, owned, observed = screen
    memory.engine.foreground.set()
    monkeypatch.setattr('scarx.hybrid.screen.subprocess.run', lambda *a, **kw: pytest.fail('Foreground has priority'))
    assert memory._ingest_audio(state, row(), owned, None) is None
    assert state['audio_skipped_intervals'][0]['reason'] == 'foreground_preempted'
    assert not state.get('asr_errors') and state['asr_frontier'] == 0 and not observed


@pytest.mark.parametrize('from_monitor', [False, True])
def test_resource_breach_is_fatal_and_is_not_relabelled_as_an_audio_fault(screen, monkeypatch, from_monitor):
    memory, state, owned, observed = screen
    def fail(*a, **kw):
        if from_monitor:
            raise RuntimeError('ASR failure followed by resource breach')
        raise MemoryError('physical reserve breached')
    monkeypatch.setattr('scarx.hybrid.screen.subprocess.run', fail)
    if from_monitor:
        def resource_breach():
            raise MemoryError('whole-system budget exceeded')
        memory.engine.monitor.check = resource_breach
    with pytest.raises(MemoryError):
        memory._ingest_audio(state, row(), owned, None)
    assert not state.get('asr_errors') and not observed


def test_audio_path_outside_owned_capture_remains_a_fatal_contract_error(screen):
    memory, state, owned, _ = screen
    with pytest.raises(ValueError, match='Invalid capture source'):
        memory._ingest_audio(state, row('../../outside.wav'), owned, None)


def test_loopback_device_unavailable_does_not_suppress_visual_observations(screen):
    memory, state, owned, observed = screen
    Image.new('RGB', (24, 24), 'blue').save(owned / 'frame.jpg')
    rows = [{'kind': 'audio_unavailable', 'error': 'owned device unavailable'},
            {'kind': 'frame', 'time': 4, 'path': 'frame.jpg'}]
    (owned / 'capture.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    memory._ingest(state, SimpleNamespace(poll=lambda: 0, returncode=0), owned)
    assert state['status'] == 'stopped' and state['vision_frontier'] == 4
    assert state['asr_status'] == 'unavailable' and not state['audio_available']
    assert state['degraded'] and [item['modality'] for item in observed] == ['sampled_frame']


def test_worker_reports_optional_audio_import_failure_without_opening_capture(monkeypatch):
    # Execute the shipped nested audio routine, not main()/WindowsCapture.
    source = Path(__file__).resolve().parents[1] / 'scripts/hybrid_screen_worker.py'
    routine = next(n for n in ast.walk(ast.parse(source.read_text()))
                   if isinstance(n, ast.FunctionDef) and n.name == 'audio')
    original_import = builtins.__import__
    def no_audio_import(name, *args, **kwargs):
        if name == 'pyaudiowpatch':
            raise ImportError('optional audio package unavailable')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_audio_import)
    emitted = []
    scope = {'emit': lambda kind, **data: emitted.append((kind, data))}
    exec(compile(ast.Module(body=[routine], type_ignores=[]), str(source), 'exec'), scope)
    scope['audio']()
    assert len(emitted) == 1 and emitted[0][0] == 'audio_unavailable'
    assert emitted[0][1]['microphone_used'] is False
