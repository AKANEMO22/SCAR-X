"""Read-only matched-head/causal diagnosis, no TEST or neural updates."""
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import subprocess, sys, time, uuid
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.evaluation.semantic_prototype_alif import SemanticALIF
from scarx.hybrid.store import Store, encode


def main():
    source = Path((ROOT / 'local/current-semantic-local-ALIF.txt').read_text(encoding='utf-8').strip())
    original = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    assert original['completed'] and original['charged_counts'] == {'forward': 33546, 'update': 4374}
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-representation-diagnostic-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    with (ROOT / 'local/semantic-representation-diagnostic-v1-execution.json').open('x', encoding='utf-8') as f:
        json.dump({'run': str(run)}, f)
    (ROOT / 'local/current-semantic-representation-diagnostic.txt').write_text(str(run), encoding='utf-8')
    feature_dir = ROOT / 'runs' / json.loads((source / 'manifest.json').read_text(encoding='utf-8'))['features_run']
    with np.load(feature_dir / 'features.npz', allow_pickle=False) as f:
        x, labels, dev = (f[key].copy() for key in ('x', 'labels', 'dev'))
    rows = json.loads((feature_dir / 'data.json').read_text(encoding='utf-8'))
    assert all(rows[i]['split'] == 'DEV' for i in dev)
    schedules = json.loads((source / 'schedules.json').read_text(encoding='utf-8'))
    curves = json.loads((source / 'curves.json').read_text(encoding='utf-8'))
    artifact = ROOT / original['artifacts']
    manifest = {'source_run': source.name, 'source_counts': original['charged_counts'],
        'source_elapsed_seconds': original['elapsed_seconds'], 'max_additional_forwards': 4500,
        'max_seconds': 180, 'updates': 0, 'TEST_calls': 0,
        'source_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'scripts/diagnose_semantic_alif_representation.py', 'experiments/semantic-alif-representation-diagnostic-v1.md')}}
    (run / 'manifest.json').write_text(encode(manifest), encoding='utf-8')
    for name in manifest['sources']:
        (run / Path(name).name).write_bytes((ROOT / name).read_bytes())
    count, started, last_resource = 0, time.monotonic(), 0.
    resources = []
    trace = Store(ROOT / 'local/hybrid-v1')
    journal = (run / 'attempts.jsonl').open('x', encoding='utf-8')
    arrays, scores, features, restored_pools = {}, {}, {}, []
    report = {'completed': False, 'failures': [], 'SNN_updates': 0, 'TEST_calls': 0,
              'whole_or_module_quality_PASS': False}
    def forward(model, vector, ticks=8):
        nonlocal count, last_resource
        if count >= 4500 or time.monotonic() - started > 180:
            raise TimeoutError('Registered read-only diagnostic quota')
        count += 1
        journal.write(encode({'forward': count, 'ticks': ticks, 'epoch': time.time()}) + '\n')
        journal.flush()
        if time.monotonic() - last_resource > 5:
            memory = psutil.Process().memory_info()
            free = int(subprocess.check_output([str(Path(os.environ['WINDIR']) / 'System32/nvidia-smi.exe'),
                '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW).strip()) * 1024**2
            resources.append({'RSS': memory.rss, 'private_commit': getattr(memory, 'private', None), 'GPU_free': free})
            if memory.rss > 2_000_000_000 or free < 512 * 1024**2:
                raise MemoryError('Read-only diagnostic physical guard')
            last_resource = time.monotonic()
        xp = model.model.xp
        model.model._forward_native(xp.repeat(xp.asarray(vector, dtype=xp.float32)[None, :], ticks, axis=0))
        return model.model.host(model.model.core.features).copy()
    def pooled(matrix):
        return np.asarray(matrix)[:, :126].reshape(-1, 6, 21).max(axis=2)
    try:
        for seed in (260113, 260137, 260171):
            scores[seed] = {}
            path = artifact / f'{seed}-prototype-stage19.npz'
            model, _ = SemanticALIF.loads(path.read_bytes())
            for name, ticks in (('ticks32', 32), ('no_recurrence_ticks8', 8)):
                if name.startswith('no_recurrence'):
                    model.model.core.w_rec.fill(0)
                matrix = np.stack([forward(model, x[i], ticks) for i in dev])
                score = pooled(matrix)
                arrays[f'{seed}_{name}_scores'] = score
                scores[seed][name] = {'accuracy': float(np.mean(score.argmax(axis=1) == labels[dev])),
                    'class_score_tie_rate': float(np.mean(np.sum(score == score.max(axis=1, keepdims=True), axis=1) > 1))}
            model, _ = SemanticALIF.loads(path.read_bytes())
            weights = model.model.host(model.model.core.w_in).copy()
            analog = (x[dev] @ weights[:126].T).reshape(-1, 6, 21).max(axis=2)
            arrays[f'{seed}_analog_cosine_control'] = analog
            scores[seed]['analog_cosine_NOT_SNN'] = float(np.mean(analog.argmax(axis=1) == labels[dev]))
            for c in range(6):
                trajectory = np.asarray(curves[str(seed)]['prototype'])
                stage = c + int(np.argmax(trajectory[c:, c]))
                past, _ = SemanticALIF.loads((artifact / f'{seed}-prototype-stage{stage:02d}.npz').read_bytes())
                model, _ = SemanticALIF.loads(path.read_bytes())
                model.model.core.w_in[c * 21:(c + 1) * 21] = past.model.core.w_in[c * 21:(c + 1) * 21]
                indices = dev[labels[dev] == c]
                score = pooled([forward(model, x[i]) for i in indices])
                restored_pools.append({'seed': seed, 'intent': c, 'past_best_stage': stage,
                    'past_best_accuracy': float(trajectory[stage, c]), 'final_accuracy': float(trajectory[-1, c]),
                    'old_target_pool_with_final_competitors_accuracy': float(np.mean(score.argmax(axis=1) == c))})
            train = np.array(schedules[str(seed)][-1]['controller']['unique_seen'])
            assert len(train) == 224 and all(rows[i]['split'] == 'TRAIN' for i in train)
            for arm in ('prototype', 'shuffled', 'frozen'):
                model = SemanticALIF(seed, 'frozen') if arm == 'frozen' else SemanticALIF.loads((artifact / f'{seed}-{arm}-stage19.npz').read_bytes())[0]
                train_features = np.stack([forward(model, x[i]) for i in train])
                dev_features = np.stack([forward(model, x[i]) for i in dev])
                mean = train_features.mean(axis=0)
                def code(matrix):
                    matrix = matrix - mean
                    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), np.float32(1e-8))
                    return np.column_stack((matrix, np.ones(len(matrix), dtype=np.float32)))
                a, b = code(train_features), code(dev_features)
                y = np.eye(6, dtype=np.float32)[labels[train]]
                head = np.linalg.solve(a.T @ a + np.eye(129, dtype=np.float32), a.T @ y)
                prediction = np.argmax(b @ head, axis=1)
                arrays[f'{seed}_{arm}_train_features'] = train_features
                arrays[f'{seed}_{arm}_DEV_features'] = dev_features
                arrays[f'{seed}_{arm}_head'] = head
                arrays[f'{seed}_{arm}_DEV_predictions'] = prediction
                features.setdefault(arm, []).append(prediction == labels[dev])
                scores[seed]['common_head_' + arm] = float(np.mean(prediction == labels[dev]))
                if arm == 'prototype':
                    pool = pooled(dev_features)
                    scores[seed]['original_ticks8_tie_rate'] = float(np.mean(np.sum(pool == pool.max(axis=1, keepdims=True), axis=1) > 1))
                trace.event('work_trace.activity', {'title': 'Đo representation bằng cùng readout',
                    'run_id': run.name, 'seed': seed, 'arm': arm,
                    'DEV_accuracy': scores[seed]['common_head_' + arm], 'forwards': count,
                    'SNN_updates': 0, 'diagnostic_not_fresh_confirmation': True}, 'orchestrator')
        effects = {}
        for arm in ('frozen', 'shuffled'):
            difference = np.asarray(features['prototype'], dtype=np.float32) - np.asarray(features[arm])
            rng = np.random.default_rng(260216)
            samples = []
            for _ in range(10000):
                seeds = rng.integers(3, size=3)
                tasks = np.concatenate([c * 20 + rng.integers(20, size=20) for c in range(6)])
                samples.append(difference[np.ix_(seeds, tasks)].mean())
            effects[arm] = {'gain': float(difference.mean()), 'CI95': np.quantile(samples, [.025, .975]).tolist()}
        if time.monotonic() - started > 180:
            raise TimeoutError('Diagnostic totalwall including data analysis')
        report.update(completed=True, verdict='READ_ONLY_DIAGNOSIS_MEASURED_NOT_NEW_QUALITY_PASS',
                      scores=scores, old_pool_interventions=restored_pools, common_head_effects=effects)
    except Exception as exc:
        report['failures'].append(type(exc).__name__ + ': ' + str(exc))
        report['verdict'] = 'REPRESENTATION_DIAGNOSTIC_STOPPED_NOT_PASS'
    finally:
        journal.close()
        np.savez(run / 'measurements.npz', **arrays)
        report.update(forward_attempts=count, elapsed_seconds=time.monotonic() - started, resources=resources,
            combined_forward_attempts=33546 + count, combined_update_attempts=4374,
            combined_active_seconds=original['elapsed_seconds'] + time.monotonic() - started)
        (run / 'result.json').write_text(encode(report), encoding='utf-8')
        trace.event('work_trace.activity', {'title': report['verdict'], 'run_id': run.name, **report}, 'orchestrator')
        trace.close()
    print(encode({'run': str(run), **report}), flush=True)
    if not report['completed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
