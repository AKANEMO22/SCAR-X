"""Read-only GPU representations/common readouts and stable-weight intervention."""
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import subprocess, sys, time, uuid
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.evaluation.semantic_credit_alif import CreditALIF
from scarx.evaluation.semantic_metaplastic_alif import ProtectedALIF
from scarx.hybrid.store import Store, encode
from diagnose_semantic_credit_common_head import gpu_spd_solve


def main():
    source = Path((ROOT / 'local/current-semantic-metaplasticity-DEV.txt').read_text(encoding='utf-8').strip())
    source_receipt = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    assert source_receipt['completed'] and not source_receipt['protected_all_seed_acquisition_retention_gate']
    old = Path((ROOT / 'local/current-semantic-credit-dev.txt').read_text(encoding='utf-8').strip())
    feature_dir = Path((ROOT / 'local/current-semantic-features.txt').read_text(encoding='utf-8').strip())
    with np.load(feature_dir / 'features.npz', allow_pickle=False) as f:
        x, labels, dev = (f[n].copy() for n in ('x', 'labels', 'dev'))
    plans = json.loads((Path((ROOT / 'local/current-semantic-local-ALIF.txt').read_text(encoding='utf-8').strip()) / 'schedules.json').read_text(encoding='utf-8'))
    data = json.loads((feature_dir / 'data.json').read_text(encoding='utf-8'))
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-three-seed-diagnosis-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    with (ROOT / 'local/semantic-three-seed-diagnosis-v1-execution.json').open('x', encoding='utf-8') as f:
        json.dump({'run': str(run)}, f)
    (ROOT / 'local/current-semantic-three-seed-diagnosis.txt').write_text(str(run), encoding='utf-8')
    sources = ('scripts/diagnose_semantic_three_seed.py', 'scripts/diagnose_semantic_credit_common_head.py',
               'experiments/semantic-credit-three-seed-diagnosis-v1.md')
    (run / 'manifest.json').write_text(encode({'source_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sources': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sources},
        'source_run': source.name, 'old_credit_source': old.name, 'feature_run': feature_dir.name,
        'max_forwards': 4600, 'max_seconds': 90, 'ALIF_updates': 0, 'TEST_calls': 0}), encoding='utf-8')
    for name in sources:
        (run / Path(name).name).write_bytes((ROOT / name).read_bytes())
    started, attempts, last = time.monotonic(), 0, 0.
    journal = (run / 'attempts.jsonl').open('x', encoding='utf-8')
    trace = Store(ROOT / 'local/hybrid-v1')
    arrays, scores, correctness, resources, residuals = {}, {}, {}, [], {}
    report = {'completed': False, 'ALIF_updates': 0, 'CPU_forward_or_fitting': False, 'TEST_calls': 0,
              'quality_or_fresh_confirmation': False, 'failures': []}
    def forward(model, vector):
        nonlocal attempts, last
        if attempts >= 4600 or time.monotonic() - started > 90 or model.device != 'gpu':
            raise RuntimeError('Read-only GPUdiagnostic bound/device')
        attempts += 1
        journal.write(encode({'forward': attempts, 'epoch': time.time()}) + '\n')
        journal.flush()
        if time.monotonic() - last > 5:
            rss = psutil.Process().memory_info().rss
            free = int(subprocess.check_output([str(Path(os.environ['WINDIR']) / 'System32/nvidia-smi.exe'),
                '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW).strip()) * 1024**2
            resources.append({'RSS': rss, 'GPU_free': free})
            if rss > 2_000_000_000 or free < 512 * 1024**2:
                raise MemoryError('GPU read-only physical guard')
            last = time.monotonic()
        xp = model.xp
        p, features, _, _, _ = model._forward_native(xp.repeat(xp.asarray(vector, dtype=xp.float32)[None, :], 8, axis=0))
        return p.copy(), features.copy()
    try:
        for seed in (260113, 260137, 260171):
            train = np.asarray(plans[str(seed)][-1]['controller']['unique_seen'])
            assert len(train) == 224 and all(data[i]['split'] == 'TRAIN' for i in train)
            scores[seed], residuals[seed] = {}, {}
            for rule in ('protected', 'norm_matched', 'unprotected', 'frozen'):
                if rule == 'frozen':
                    model = CreditALIF(seed, 'frozen').model
                elif rule == 'unprotected' and seed == 260113:
                    model = CreditALIF.loads((ROOT / 'artifacts' / old.name / 'hidden_eta20-stage19.npz').read_bytes())[0].model
                else:
                    model = ProtectedALIF.loads((ROOT / source_receipt['artifacts'] / f'{seed}-{rule}-stage19.npz').read_bytes())[0].model
                xp = model.xp
                parts, original_dev = [], []
                for selected in (train, dev):
                    values = []
                    for index in selected:
                        p, feature = forward(model, x[index])
                        values.append(feature)
                        if data[index]['split'] == 'DEV':
                            original_dev.append(model.host(p))
                    parts.append(xp.stack(values))
                a, b = parts
                mean = a.mean(axis=0)
                def code(matrix):
                    matrix = matrix - mean
                    matrix /= xp.maximum(xp.linalg.norm(matrix, axis=1, keepdims=True), np.float32(1e-8))
                    return xp.column_stack((matrix, xp.ones(len(matrix), dtype=xp.float32)))
                aa, bb = code(a), code(b)
                head, residual = gpu_spd_solve(xp, aa.T @ aa + xp.eye(aa.shape[1], dtype=xp.float32),
                                             aa.T @ xp.eye(6, dtype=xp.float32)[xp.asarray(labels[train])])
                pred = model.host(xp.argmax(bb @ head, axis=1))
                scores[seed][rule] = float(np.mean(pred == labels[dev]))
                residuals[seed][rule] = residual
                correctness.setdefault(rule, []).append(pred == labels[dev])
                for key, value in (('TRAIN', a), ('DEV', b), ('head', head), ('mean', mean)):
                    arrays[f'{seed}_{rule}_{key}'] = model.host(value)
                arrays[f'{seed}_{rule}_prediction'] = pred
                arrays[f'{seed}_{rule}_original_DEV'] = np.stack(original_dev)
                if rule == 'protected':
                    model.core.w_in[:] = model.core.stable_in
                    stable_p = np.stack([model.host(forward(model, x[i])[0]) for i in dev])
                    arrays[f'{seed}_stable_input_DEV'] = stable_p
                    scores[seed]['stable_input_fixed_head'] = float(np.mean(stable_p.argmax(axis=1) == labels[dev]))
                trace.event('work_trace.activity', {'title': 'Kiểm representation trên3seed · GPU', 'run_id': run.name,
                    'seed': seed, 'rule': rule, 'common_head_DEV_accuracy': scores[seed][rule],
                    'GPU_solver_residual': residual, 'ALIF_updates': 0, 'forwards': attempts}, 'orchestrator')
            # Shared raw-feature readout fit on the identical224seenTRAIN (CUDA).
            a, b = xp.asarray(x[train]), xp.asarray(x[dev])
            mean = a.mean(axis=0)
            aa, bb = code(a), code(b)
            head, residual = gpu_spd_solve(xp, aa.T @ aa + xp.eye(65, dtype=xp.float32),
                                         aa.T @ xp.eye(6, dtype=xp.float32)[xp.asarray(labels[train])])
            pred = model.host(xp.argmax(bb @ head, axis=1))
            arrays[f'{seed}_rawPCA64_prediction'] = pred
            arrays[f'{seed}_rawPCA64_head'] = model.host(head)
            correctness.setdefault('rawPCA64', []).append(pred == labels[dev])
            scores[seed]['rawPCA64'] = float(np.mean(pred == labels[dev]))
            residuals[seed]['rawPCA64'] = residual
        effects = {}
        for candidate in ('protected', 'unprotected'):
            for control in ('frozen', 'norm_matched', 'rawPCA64'):
                diff = np.asarray(correctness[candidate], dtype=float) - np.asarray(correctness[control])
                rng, values = np.random.default_rng(260216), []
                for _ in range(10000):
                    seeds = rng.integers(3, size=3)
                    items = np.concatenate([c * 20 + rng.integers(20, size=20) for c in range(6)])
                    values.append(float(diff[np.ix_(seeds, items)].mean()))
                effects[candidate + '-vs-' + control] = {'gain': float(diff.mean()), 'CI95': np.quantile(values, [.025, .975]).tolist()}
        if time.monotonic() - started > 90:
            raise TimeoutError('Read-only diagnosis totalwall')
        report.update(completed=True, verdict='THREE_SEED_REPRESENTATION_DIAGNOSIS_NOT_CONFIRMATION',
            scores=scores, effects=effects, GPU_ridge_residuals=residuals,
            mean_common_head_accuracy={k: float(np.mean(v)) for k, v in correctness.items()})
    except Exception as exc:
        report['failures'].append(type(exc).__name__ + ': ' + str(exc))
        report['verdict'] = 'THREE_SEED_GPU_DIAGNOSIS_STOPPED_NOT_PASS'
    finally:
        np.savez(run / 'measurements.npz', **arrays)
        report.update(forward_attempts=attempts, elapsed_seconds=time.monotonic() - started, resources=resources)
        (run / 'result.json').write_text(encode(report), encoding='utf-8')
        journal.close()
        trace.event('work_trace.activity', {'title': report['verdict'], 'run_id': run.name, **report}, 'orchestrator')
        trace.close()
    print(encode({'run': str(run), **report}), flush=True)
    if not report['completed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
