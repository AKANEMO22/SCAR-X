"""Bounded native frozen semantic features, no SNN/VLM weight learning."""
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import secrets, subprocess, sys, time, uuid
from urllib.request import Request, urlopen
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.store import Store, encode


def main():
    prior = ROOT / 'runs/20261115T093544Z-semantic-encoder-v1-84f936'
    prior_result = json.loads((prior / 'result.json').read_text(encoding='utf-8'))
    assert prior_result['attempts'] == 1 and not prior_result['completed']
    second = ROOT / 'runs/20261115T093631Z-semantic-encoder-v2-5bbf9f'
    second_result = json.loads((second / 'result.json').read_text(encoding='utf-8'))
    assert not second_result['completed'] and second_result['attempts'] == 0
    remaining_seconds = 600 - prior_result['elapsed_seconds'] - second_result['elapsed_seconds']
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-encoder-recovery-p0-' + uuid.uuid4().hex[:6])
    run.mkdir()
    with (ROOT / 'local/semantic-encoder-recovery-p0-execution.json').open('x', encoding='utf-8') as f:
        json.dump({'run': str(run)}, f)
    (ROOT / 'local/current-semantic-encoder.txt').write_text(str(run), encoding='utf-8')
    source = Path((ROOT / 'local/current-semantic-ALIF-data.txt').read_text(encoding='utf-8').strip())
    data_bytes = (source / 'selected-data.json').read_bytes()
    data = json.loads(data_bytes)
    assert len(data) == 600
    native = SharedNative(ROOT)
    args = native.worker.cmdline().copy()
    if '--mmproj' in args:
        index = args.index('--mmproj')
        del args[index:index + 2]
    for flag, value in (('-c', '512'), ('-b', '512'), ('-ub', '512')):
        if flag in args:
            args[args.index(flag) + 1] = value
        else:
            args += [flag, value]
    token = secrets.token_hex(24)
    for flag, value in (('--port', '18786'), ('--api-key', token), ('--alias', 'scarx-semantic-encoder')):
        if flag in args:
            args[args.index(flag) + 1] = value
    if '--mmap' in args:
        args.remove('--mmap')
    args += ['--embeddings', '--pooling', 'last', '--no-mmap']
    assert '--lora' not in args
    manifest = {
        'data_run': source.name, 'data_sha256': hashlib.sha256(data_bytes).hexdigest(),
        'source_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in (
            'scripts/encode_semantic_alif_banking.py', 'experiments/semantic-alif-local-prototypes-v1.md')},
        'native_exe_sha256': hashlib.sha256(Path(args[0]).read_bytes()).hexdigest(),
        'native_model_sha256': hashlib.sha256(Path(args[args.index('-m') + 1]).read_bytes()).hexdigest(),
        'max_calls': 609, 'max_seconds': remaining_seconds, 'combined_max_calls': 610,
        'prior_charged_calls': 1, 'prior_run': prior.name, 'text_context_tokens': 512,
        'second_closed_run': second.name, 'model_memory_mapping': False,
        'pooling': 'last', 'precision': 'original BF16/F32',
        'encoder_inputs': 'raw public utterance only; never labels/intent names/split/IDs',
        'SNN_updates': 0, 'VLM_updates': 0, 'live_adapter_changed': False,
    }
    (run / 'manifest.json').write_text(encode(manifest), encoding='utf-8')
    for name in ('scripts/encode_semantic_alif_banking.py', 'experiments/semantic-alif-local-prototypes-v1.md'):
        (run / Path(name).name).write_bytes((ROOT / name).read_bytes())
    process = None
    created = None
    started = time.monotonic()
    trace = Store(ROOT / 'local/hybrid-v1')
    records, vectors, failures = [], [], []
    def guard():
        if time.monotonic() - started > remaining_seconds:
            raise TimeoutError('600second encoder budget')
        native.check()
        output = subprocess.check_output([str(Path(os.environ['WINDIR']) / 'System32/nvidia-smi.exe'),
            '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW)
        free = int(output.strip()) * 1024**2
        if free < 512 * 1024**2:
            raise MemoryError('512MiB discrete GPU reserve')
        if process:
            memory = psutil.Process(process.pid).memory_info()
            report['last_worker_memory'] = {'RSS': memory.rss, 'private_commit': getattr(memory, 'private', None)}
            if memory.rss > 2_000_000_000:
                raise MemoryError('Encoder worker2GB resident cap: ' + str(memory.rss))
        return free
    def call(path, body=None):
        with urlopen(Request('http://127.0.0.1:18786' + path,
            data=None if body is None else encode(body).encode(), headers={
                'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}), timeout=45) as r:
            raw = r.read(250_001)
        if len(raw) > 250_000:
            raise ValueError('Embedding response byte cap')
        return json.loads(raw)
    def embedding(text, identifier):
        if len(records) >= 609:
            raise RuntimeError('610calls precharge cap')
        guard()
        attempt = {'attempt': len(records) + 1, 'id': identifier, 'epoch': time.time(),
                   'input_sha256': hashlib.sha256(text.encode()).hexdigest()}
        records.append(attempt)
        with (run / 'attempts.jsonl').open('a', encoding='utf-8') as f:
            f.write(encode(attempt) + '\n')
        before = time.monotonic()
        raw = call('/v1/embeddings', {'model': 'scarx-semantic-encoder', 'input': text,
                                    'encoding_format': 'float'})
        vector = np.asarray(raw['data'][0]['embedding'], dtype=np.float32)
        if vector.shape != (1024,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
            raise ValueError('Expected finite nonzero1024D native embedding')
        attempt.update(seconds=time.monotonic() - before, norm=float(np.linalg.norm(vector)),
                       output_sha256=hashlib.sha256(vector.tobytes()).hexdigest())
        with (run / 'completed.jsonl').open('a', encoding='utf-8') as f:
            f.write(encode(attempt) + '\n')
        return vector
    report = {'completed': False, 'SNN_updates': 0, 'VLM_updates': 0, 'failures': failures}
    try:
        guard()
        with (run / 'native.stdout.log').open('wb') as out, (run / 'native.stderr.log').open('wb') as err:
            process = subprocess.Popen(args, cwd=ROOT, stdout=out, stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        created = psutil.Process(process.pid).create_time()
        deadline = time.monotonic() + 90
        while True:
            if process.poll() is not None:
                raise RuntimeError('Owned embedding worker exited during load')
            try:
                if call('/health').get('status') == 'ok':
                    break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError('Embedding worker readiness')
                time.sleep(.25)
        a = embedding('An extra fee was charged on my card payment.', 'E0-card')
        b = embedding('What exchange rate applies to a cash withdrawal abroad?', 'E0-cash')
        if np.array_equal(a, b):
            raise ValueError('Encoder returned identical vectors for distinct inputs')
        np.savez(run / 'E0.npz', a=a, b=b)
        trace.event('work_trace.activity', {'title': 'Semantic ALIF: native encoder E0',
            'run_id': run.name, 'finite_1024D': True, 'actual_calls': len(records),
            'SNN_updates': 0, 'model': 'frozen original Qwen0.8B CUDA'}, 'orchestrator')
        resources = []
        for index, row in enumerate(data):
            while native.status().get('cognition', {}).get('phase') == 'serving':
                guard()
                time.sleep(.25)
            vectors.append(embedding(row['text'], row['id']))
            if index % 20 == 0 or index == len(data) - 1:
                resources.append({'completed': index + 1, 'free_VRAM_bytes': guard(),
                    'worker_RSS': psutil.Process(process.pid).memory_info().rss})
                trace.event('work_trace.activity', {'title': 'Đang mã hóa câu hỏi thật cho SNN/RAG',
                    'run_id': run.name, 'completed': index + 1, 'total': 600,
                    'public_question': row['text'], 'split': row['split'], 'SNN_updates': 0,
                    'encoder': 'same frozen Qwen0.8B for all controls', 'resources': resources[-1]}, 'orchestrator')
        matrix = np.stack(vectors).astype(np.float32)
        np.save(run / 'vectors.npy', matrix, allow_pickle=False)
        (run / 'data.json').write_bytes(data_bytes)
        report.update(completed=True, verdict='FROZEN_SEMANTIC_FEATURES_EXTRACTED_NOT_SNN_LEARNING',
            shape=list(matrix.shape), vectors_sha256=hashlib.sha256((run / 'vectors.npy').read_bytes()).hexdigest(),
            resources=resources)
    except Exception as exc:
        failures.append(type(exc).__name__ + ': ' + str(exc))
        report['verdict'] = 'SEMANTIC_ENCODER_STOPPED_NOT_PASS'
        if vectors:
            np.save(run / 'partial-vectors.npy', np.stack(vectors), allow_pickle=False)
    finally:
        if process is not None and process.poll() is None:
            owned = psutil.Process(process.pid)
            if owned.create_time() == created and Path(owned.exe()).resolve() == Path(args[0]).resolve():
                process.terminate()
                process.wait(timeout=15)
        report.update(attempts=len(records), completed_vectors=len(vectors),
            elapsed_seconds=time.monotonic() - started,
            owned_worker_cleanup_complete=process is None or process.poll() is not None)
        (run / 'result.json').write_text(encode(report), encoding='utf-8')
        trace.event('work_trace.activity', {'title': report['verdict'], 'run_id': run.name,
            'attempts': len(records), 'SNN_updates': 0, 'failures': failures}, 'orchestrator')
        trace.close()
    print(encode({'run': str(run), **report}), flush=True)
    if not report['completed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
