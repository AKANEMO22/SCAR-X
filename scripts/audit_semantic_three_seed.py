"""Check saved GPU-fitted coefficients/predictions; no fitting or neural work."""
from datetime import datetime, timezone
import hashlib, json, sys, uuid
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.hybrid.store import Store, encode


def main():
    source = Path((ROOT / 'local/current-semantic-three-seed-diagnosis.txt').read_text(encoding='utf-8').strip())
    original = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-three-seed-data-audit-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    report = {'source_run': source.name, 'GPU_or_neural_calls': 0, 'model_fitting': False,
              'quality_confirmation': False, 'checks': [], 'failures': []}
    try:
        assert original['completed'] and original['forward_attempts'] == 4488 and original['elapsed_seconds'] <= 90
        for name, digest in manifest['sources'].items():
            assert hashlib.sha256((source / Path(name).name).read_bytes()).hexdigest() == digest
        attempts = [json.loads(line) for line in (source / 'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['forward'] for r in attempts] == list(range(1, 4489))
        feature_dir = ROOT / 'runs' / manifest['feature_run']
        with np.load(feature_dir / 'features.npz', allow_pickle=False) as f:
            x, y, dev = (f[k].copy() for k in ('x', 'labels', 'dev'))
        schedules = json.loads((Path((ROOT / 'local/current-semantic-local-ALIF.txt').read_text(encoding='utf-8').strip()) / 'schedules.json').read_text(encoding='utf-8'))
        with np.load(source / 'measurements.npz', allow_pickle=False) as f:
            saved = {k: f[k].copy() for k in f.files}
        correctness, max_residual = {}, 0.
        for seed in (260113, 260137, 260171):
            train = np.asarray(schedules[str(seed)][-1]['controller']['unique_seen'])
            for rule in ('protected', 'norm_matched', 'unprotected', 'frozen', 'rawPCA64'):
                if rule == 'rawPCA64':
                    a, b, head = x[train], x[dev], saved[f'{seed}_{rule}_head']
                    mean = a.mean(axis=0)
                else:
                    a, b, head, mean = (saved[f'{seed}_{rule}_{k}'] for k in ('TRAIN', 'DEV', 'head', 'mean'))
                    np.testing.assert_allclose(mean, a.mean(axis=0), rtol=1e-5, atol=1e-7)
                def code(matrix):
                    matrix = matrix - mean
                    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), np.float32(1e-8))
                    return np.column_stack((matrix, np.ones(len(matrix), dtype=np.float32)))
                aa, bb = code(a), code(b)
                # Validate already fitted weights; never solve/refit on CPU.
                lhs = aa.T @ aa + np.eye(aa.shape[1], dtype=np.float32)
                rhs = aa.T @ np.eye(6, dtype=np.float32)[y[train]]
                residual = float(np.linalg.norm(lhs @ head - rhs) / max(np.linalg.norm(rhs), 1e-8))
                max_residual = max(max_residual, residual)
                assert residual < 3e-5
                pred = np.argmax(bb @ head, axis=1)
                np.testing.assert_array_equal(pred, saved[f'{seed}_{rule}_prediction'])
                correct = pred == y[dev]
                correctness.setdefault(rule, []).append(correct)
                assert abs(float(correct.mean()) - original['scores'][str(seed)][rule]) < 1e-12
            assert abs(float(np.mean(saved[f'{seed}_stable_input_DEV'].argmax(axis=1) == y[dev])) -
                       original['scores'][str(seed)]['stable_input_fixed_head']) < 1e-12
        report['checks'].append('all15GPU-fitted heads satisfy same ridge equations; predictions/residuals verified without refitting')
        report['max_CPU_validation_residual'] = max_residual
        for key, effect in original['effects'].items():
            candidate, control = key.split('-vs-')
            difference = np.asarray(correctness[candidate], dtype=float) - np.asarray(correctness[control])
            rng, values = np.random.default_rng(260216), []
            for _ in range(10000):
                seeds = rng.integers(3, size=3)
                items = np.concatenate([c * 20 + rng.integers(20, size=20) for c in range(6)])
                values.append(float(difference[np.ix_(seeds, items)].mean()))
            np.testing.assert_allclose(np.quantile(values, [.025, .975]), effect['CI95'], rtol=1e-12, atol=1e-12)
            assert abs(float(difference.mean()) - effect['gain']) < 1e-12
        report['checks'].append('all6paired seed/text DEV confidence intervals independently recomputed')
        report.update(verdict='THREE_SEED_DATA_AND_COEFFICIENT_AUDIT_PASS_NOT_QUALITY_PASS',
            common_head_means=original['mean_common_head_accuracy'], effects=original['effects'])
    except Exception as exc:
        report['failures'].append(type(exc).__name__ + ': ' + str(exc))
        report['verdict'] = 'THREE_SEED_DATA_AUDIT_STOPPED_NOT_PASS'
    (run / 'result.json').write_text(encode(report), encoding='utf-8')
    (run / 'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT / 'local/current-semantic-three-seed-audit.txt').write_text(str(run), encoding='utf-8')
    trace = Store(ROOT / 'local/hybrid-v1')
    trace.event('work_trace.activity', {'title': 'Audit hệ số/representation3seed', 'run_id': run.name, **report}, 'orchestrator')
    trace.close()
    print(encode({'run': str(run), **report}), flush=True)
    if report['failures']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
