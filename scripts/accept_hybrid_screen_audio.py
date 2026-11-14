"""Two real native frame calls across a real FFmpeg audio-decode failure.

Owned journal producer, not Windows screen capture or a model-quality benchmark.
No ASR, learning, new models, web or private-window access. Old source is replayed
only on a disposable engineering fixture, never installed into the live gateway.
"""

import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.device_budget import gpu_headroom
from scarx.hybrid.screen import ScreenMemory
from scarx.hybrid.store import Store


def main():
    run = ROOT / 'local/screen-audio-resilience' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True)
    baseline = subprocess.check_output(['git', 'show', '1e8f4534:src/scarx/hybrid/screen.py'], cwd=ROOT)
    old_path = run / 'baseline-screen.py'
    old_path.write_bytes(baseline)
    spec = importlib.util.spec_from_file_location('scarx.hybrid._screen_audio_baseline', old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    native = SharedNative(ROOT)
    before = native.status()['neural']['updates']
    producer = run / 'journal-producer.py'
    producer.write_text('''import json,sys,time
from pathlib import Path
root=Path(sys.argv[1])
with (root/'capture.jsonl').open('w',buffering=1) as f:
 for row in [{'kind':'audio','path':'corrupt.wav','start':0,'end':1},
             {'kind':'frame','path':'red.jpg','time':1}]:f.write(json.dumps(row)+'\\n')
 deadline=time.monotonic()+20
 while not (root/'first-frame-observed').exists():
  if time.monotonic()>deadline:raise TimeoutError('Fixture first-frame acknowledgement absent')
  time.sleep(.05)
 f.write(json.dumps({'kind':'frame','path':'blue.jpg','time':10})+'\\n')
''', encoding='utf-8')
    results = []
    for label, cls in [('baseline', old.ScreenMemory), ('candidate', ScreenMemory)]:
        directory = run / label
        store = Store(directory)
        owned = directory / 'archive' / ('e' * 32)
        owned.mkdir(parents=True)
        (owned / 'corrupt.wav').write_bytes(b'owned malformed audio fixture; not a WAV')
        for color in ('red', 'blue'):
            Image.new('RGB', (128, 128), color).save(owned / (color + '.jpg'))
        calls, observed, headroom = [], [], []
        started = time.monotonic()
        def check():
            native.check()
            if time.monotonic() - started > 60:
                raise TimeoutError('Bounded functional acceptance expired')
            value = gpu_headroom()
            headroom.append(value['free_MiB'])
            if value['free_MiB'] < 512:
                raise MemoryError('Physical GPU reserve breached')
        def describe(path, background=False):
            assert background and len(calls) < 2
            result = native.generate({'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()}},
                {'type': 'text', 'text': 'Describe the visible color briefly. Do not infer speech.'},
            ]}], 'max_tokens': 32})
            calls.append(result)
            return {'text': result['choices'][0]['message']['content']}
        def observe(text, **metadata):
            observed.append({'text': text, **metadata})
            store.observe(text, **metadata)
            (owned / 'first-frame-observed').touch()
        def no_transcript(_):
            raise AssertionError('Corrupt audio must fail FFmpeg before any ASR call')
        engine = SimpleNamespace(store=store, monitor=SimpleNamespace(check=check),
                                 foreground=threading.Event(), observe=observe,
                                 backend=SimpleNamespace(describe=describe, transcribe=no_transcript))
        memory = cls(engine)
        state = {'id': 'e' * 32, 'kind': 'screen', 'status': 'starting',
                 'capture_frontier': 0, 'vision_frontier': 0, 'asr_frontier': 0,
                 'latest_frame': None, 'audio_available': True,
                 'asr_status': 'waiting', 'error': None}
        memory.heartbeats[state['id']] = threading.Event()
        with (directory / 'producer.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, str(producer), str(owned)],
                                       stdout=log, stderr=log,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            memory._ingest(state, process, owned)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            store.close()
        result = {'arm': label, 'state': state, 'observations': observed,
                  'actual_native_outputs': calls, 'actual_native_calls': len(calls),
                  'GPU_free_MiB_point_samples': headroom,
                  'owned_producer_terminal': process.poll() is not None,
                  'seconds': time.monotonic() - started}
        (directory / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results.append(result)
    old, new = results
    assert old['state']['status'] == 'failed' and old['actual_native_calls'] == 0
    assert new['state']['status'] == 'stopped' and new['actual_native_calls'] == 2
    assert new['state']['capture_frontier'] == new['state']['vision_frontier'] == 10
    assert new['state']['asr_frontier'] == 0 and new['state']['degraded']
    assert all(item['modality'] == 'sampled_frame' for item in new['observations'])
    assert len(new['state']['asr_errors']) == 1
    assert native.status()['neural']['updates'] == before
    report = {'verdict': 'REAL_FFMPEG_AUDIO_FAILURE_ISOLATED_WHILE_NATIVE_FRAME_ANALYSIS_CONTINUES',
              'scope': 'owned journal fixture; real FFmpeg decode failure and real VLM; not a WGC/loopback or quality test',
              'baseline_failed_whole_capture': True, 'candidate_visual_frames_observed': 2,
              'actual_native_VLM_calls': 2, 'actual_ASR_calls': 0,
              'failed_audio_interval': [0, 1], 'false_transcripts': 0,
              'live_SNN_updates_before_and_after': before,
              'GPU_free_point_sample_min_MiB': min(new['GPU_free_MiB_point_samples']),
              'not_continuous_whole_system_peak_measurement': True,
              'all_owned_fixture_processes_terminal': all(x['owned_producer_terminal'] for x in results),
              'actual_frame_answers': [{'question': 'Describe the visible color briefly. Do not infer speech.',
                                        'answer': item['text'], 'source_time': item['start']}
                                       for item in new['observations']],
              'source_sha256': {'baseline-screen.py': hashlib.sha256(baseline).hexdigest(),
                                'candidate-screen.py': hashlib.sha256((ROOT/'src/scarx/hybrid/screen.py').read_bytes()).hexdigest()},
              'new_learning_or_production_PASS': False}
    (run / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (ROOT / 'local/current-screen-audio-resilience.txt').write_text(str(run), encoding='utf-8')
    live = Store(ROOT / 'local/hybrid-v1')
    live.event('work_trace.activity', {'kind': 'actual_functional_acceptance',
               'evidence': str((run/'result.json').relative_to(ROOT)), **report}, 'orchestrator')
    live.close()
    print(json.dumps(report))


if __name__ == '__main__':
    main()
