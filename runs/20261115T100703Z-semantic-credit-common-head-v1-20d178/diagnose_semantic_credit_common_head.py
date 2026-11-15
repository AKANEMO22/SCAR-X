"""GPU-only feature extraction and common-head fitting; no ALIF updates/TEST."""
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import subprocess, sys, time, uuid
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.evaluation.semantic_credit_alif import CreditALIF
from scarx.hybrid.store import Store, encode


def main():
    source = Path((ROOT / 'local/current-semantic-credit-dev.txt').read_text(encoding='utf-8').strip())
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    feature_dir = ROOT / 'runs' / manifest['feature_run']
    with np.load(feature_dir / 'features.npz', allow_pickle=False) as f:
        x, labels, dev = (f[n].copy() for n in ('x', 'labels', 'dev'))
    stages = json.loads((ROOT / 'runs' / manifest['schedule_source'] / 'schedules.json').read_text(encoding='utf-8'))['260113']
    train = np.array(stages[-1]['controller']['unique_seen'])
    data = json.loads((feature_dir / 'data.json').read_text(encoding='utf-8'))
    assert len(train) == 224 and len(dev) == 120
    assert all(data[i]['split'] == 'TRAIN' for i in train) and all(data[i]['split'] == 'DEV' for i in dev)
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-credit-common-head-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    with (ROOT / 'local/semantic-credit-common-head-v1-execution.json').open('x', encoding='utf-8') as f:
        json.dump({'run': str(run)}, f)
    (ROOT / 'local/current-semantic-credit-common-head.txt').write_text(str(run), encoding='utf-8')
    sources = ('scripts/diagnose_semantic_credit_common_head.py', 'experiments/semantic-credit-common-head-diagnostic-v1.md')
    (run / 'manifest.json').write_text(encode({'source_run': source.name, 'source_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sources},
        'ALIF_updates': 0, 'TEST_calls': 0, 'device': 'CUDA0', 'max_forwards': 1100, 'max_seconds': 60}), encoding='utf-8')
    for name in sources:
        (run / Path(name).name).write_bytes((ROOT / name).read_bytes())
    trace = Store(ROOT / 'local/hybrid-v1')
    journal = (run / 'attempts.jsonl').open('x', encoding='utf-8')
    attempts, started, last = 0, time.monotonic(), 0.
    arrays, scores, correct, resources = {}, {}, {}, []
    report = {'completed': False, 'ALIF_updates': 0, 'CPU_forward_or_fitting': False, 'TEST_calls': 0,
              'confirmation_or_quality_PASS': False, 'failures': []}
    def charge(kind):
        nonlocal attempts, last
        if kind != 'forward' or attempts >= 1100 or time.monotonic() - started > 60:
            raise TimeoutError('GPU commonhead diagnostic quota')
        attempts += 1
        journal.write(encode({'forward': attempts, 'epoch': time.time()}) + '\n')
        journal.flush()
        if time.monotonic() - last > 5:
            free = int(subprocess.check_output([str(Path(os.environ['WINDIR']) / 'System32/nvidia-smi.exe'),
                '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW).strip()) * 1024**2
            rss = psutil.Process().memory_info().rss
            resources.append({'RSS': rss, 'GPU_free': free})
            if rss > 2_000_000_000 or free < 512 * 1024**2:
                raise MemoryError('GPU commonhead physical guard')
            last = time.monotonic()
    try:
        for name in ('hidden_eta05', 'hidden_eta20', 'frozen'):
            if name == 'frozen':
                model = CreditALIF(260113, 'frozen')
            else:
                model = CreditALIF.loads((ROOT / 'artifacts' / source.name / f'{name}-stage19.npz').read_bytes())[0]
            model.budget = type('Budget', (), {'charge': staticmethod(charge)})()
            xp = model.model.xp
            matrices = []
            for indices in (train, dev):
                fs = []
                for index in indices:
                    _, features = model.forward(x[index])
                    fs.append(features.copy())
                matrices.append(xp.stack(fs))
            a, b = matrices
            mean = a.mean(axis=0)
            def code(matrix):
                matrix = matrix - mean
                matrix /= xp.maximum(xp.linalg.norm(matrix, axis=1, keepdims=True), np.float32(1e-8))
                return xp.column_stack((matrix, xp.ones(len(matrix), dtype=xp.float32)))
            train_code, dev_code = code(a), code(b)
            # All fitting and prediction explicitly CuPy CUDA; no CPU optimizer.
            head = xp.linalg.solve(train_code.T @ train_code + xp.eye(129, dtype=xp.float32),
                                   train_code.T @ xp.eye(6, dtype=xp.float32)[xp.asarray(labels[train])])
            prediction = xp.argmax(dev_code @ head, axis=1)
            correct[name] = model.model.host(prediction == xp.asarray(labels[dev]))
            scores[name] = float(correct[name].mean())
            for key, value in (('TRAIN', a), ('DEV', b), ('head', head), ('mean', mean), ('prediction', prediction)):
                arrays[name + '_' + key] = model.model.host(value)
            trace.event('work_trace.activity', {'title': 'Common-head trênGPU: kiểm chất lượng hidden',
                'run_id': run.name, 'arm': name, 'DEV_accuracy': scores[name], 'ALIF_updates': 0,
                'forwards': attempts, 'readout_fit_device': 'CUDA0', 'TEST_calls': 0}, 'orchestrator')
        effects = {}
        for name in ('hidden_eta05', 'hidden_eta20'):
            diff = correct[name].astype(float) - correct['frozen']
            rng = np.random.default_rng(260216)
            samples = [float(diff[np.concatenate([c * 20 + rng.integers(20, size=20) for c in range(6)])].mean()) for _ in range(10000)]
            effects[name] = {'gain_vs_frozen': float(diff.mean()), 'CI95': np.quantile(samples, [.025, .975]).tolist()}
        if time.monotonic() - started > 60:
            raise TimeoutError('Commonhead diagnostic totalwall')
        report.update(completed=True, verdict='COMMON_HEAD_DEV_DIAGNOSIS_NOT_CONFIRMATION', scores=scores, effects=effects)
    except Exception as exc:
        report['failures'].append(type(exc).__name__ + ': ' + str(exc))
        report['verdict'] = 'COMMON_HEAD_GPU_DIAGNOSTIC_STOPPED_NOT_PASS'
    finally:
        np.savez(run / 'measurements.npz', **arrays)
        report.update(forward_attempts=attempts, elapsed_seconds=time.monotonic() - started, resources=resources,
                      source_charged_forwards=9126, combined_forwards=9126 + attempts)
        (run / 'result.json').write_text(encode(report), encoding='utf-8')
        journal.close()
        trace.event('work_trace.activity', {'title': report['verdict'], 'run_id': run.name, **report}, 'orchestrator')
        trace.close()
    print(encode({'run': str(run), **report}), flush=True)
    if not report['completed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
