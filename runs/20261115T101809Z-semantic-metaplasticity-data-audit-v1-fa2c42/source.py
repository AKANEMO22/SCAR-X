"""Saved-array rescore, matched-norm/persistence audit; no model fitting/forward."""
from collections import Counter
from datetime import datetime, timezone
import hashlib, io, json, sys, uuid
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.hybrid.store import Store, encode


def checkpoint(path):
    with np.load(path, allow_pickle=False) as outer:
        rule = json.loads(outer['metadata'].tobytes())['rule']
        with np.load(io.BytesIO(outer['inner'].tobytes()), allow_pickle=False) as credit:
            meta = json.loads(credit['metadata'].tobytes())
            with np.load(io.BytesIO(credit['model'].tobytes()), allow_pickle=False) as model:
                arrays = {k: model[k].copy() for k in model.files if k != 'metadata'}
    return rule, meta, arrays


def main():
    source = Path((ROOT / 'local/current-semantic-metaplasticity-DEV.txt').read_text(encoding='utf-8').strip())
    original = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-metaplasticity-data-audit-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    report = {'source_run': source.name, 'new_training_or_GPU_forward': False, 'TEST_calls': 0,
              'whole_or_module_quality_PASS': False, 'checks': [], 'failures': []}
    try:
        assert original['completed']
        for name, digest in manifest['sources'].items():
            assert hashlib.sha256((source / Path(name).name).read_bytes()).hexdigest() == digest
        attempts = [json.loads(line) for line in (source / 'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        counts = Counter(r['kind'] for r in attempts)
        assert dict(counts) == original['charged_counts']
        assert counts['update'] == 3846 and counts['forward'] <= 30000
        for kind in counts:
            assert [r['number'] for r in attempts if r['kind'] == kind] == list(range(1, counts[kind] + 1))
        assert original['elapsed_seconds'] <= 240
        report['checks'].append('source hashes and persistent3846update ledger match; no quota refund')
        losses = [json.loads(line) for line in (source / 'losses.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(losses) == 3840
        feature_dir = ROOT / 'runs' / manifest['feature_run']
        with np.load(feature_dir / 'features.npz', allow_pickle=False) as f:
            y, dev = f['labels'].copy(), f['dev'].copy()
        data = json.loads((feature_dir / 'data.json').read_text(encoding='utf-8'))
        schedules = json.loads((ROOT / 'runs' / manifest['schedule_source'] / 'schedules.json').read_text(encoding='utf-8'))
        assert all(data[r['sample_index']]['split'] == 'TRAIN' and int(y[r['sample_index']]) == r['target'] and r['device'] == 'gpu' for r in losses)
        max_error = 0.
        for seed in manifest['seeds']:
            for rule in ('protected', 'norm_matched', 'unprotected'):
                records = [r for r in losses if r['seed'] == seed and r['rule'] == rule]
                if seed == 260113 and rule == 'unprotected':
                    assert not records
                    continue
                assert len(records) == 480
                assert [r['sample_index'] for r in records] == [i for stage in schedules[str(seed)] for key in ('current', 'replay_samples') for i in stage[key]]
            p = [r for r in losses if r['seed'] == seed and r['rule'] == 'protected']
            n = [r for r in losses if r['seed'] == seed and r['rule'] == 'norm_matched']
            for a, b in zip(p, n):
                assert a['sample_index'] == b['sample_index'] and a['target'] == b['target'] and a['stage'] == b['stage']
                error = abs(a['weight_delta_L2'] - b['weight_delta_L2'])
                assert error <= max(1e-6, 1e-4 * a['weight_delta_L2']) and abs(error - b['norm_match_error']) < 1e-12
                max_error = max(max_error, error)
        report['checks'].append('3840TRAIN-onlyGPU updates;1440protected/norm-matched pairs verified')
        report['max_norm_match_error'] = max_error
        artifact = ROOT / original['artifacts']
        checkpoints = 0
        for seed in manifest['seeds']:
            for rule in ('protected', 'norm_matched', 'unprotected'):
                if seed == 260113 and rule == 'unprotected':
                    continue
                for stage in range(20):
                    stored_rule, meta, arrays = checkpoint(artifact / f'{seed}-{rule}-stage{stage:02d}.npz')
                    assert stored_rule == rule and meta['updates'] == meta['hidden_updates'] == 24 * (stage + 1)
                    assert meta['controller'] == schedules[str(seed)][stage]['controller']
                    assert all(np.isfinite(a).all() for a in arrays.values())
                    assert arrays['importance'].dtype == np.float32 and np.all(arrays['importance'] >= 0)
                    if stage == 0:
                        recurrence = arrays['core.w_rec'].copy()
                    np.testing.assert_array_equal(recurrence, arrays['core.w_rec'])
                    np.testing.assert_array_equal(arrays['core.w_out'], arrays['core.stable_out'])
                    np.testing.assert_array_equal(arrays['f_mean'], np.zeros(128, dtype=np.float32))
                    np.testing.assert_array_equal(arrays['f_var'], np.ones(128, dtype=np.float32))
                    if rule == 'protected':
                        assert arrays['importance'][:, :64].sum() > 0
                        assert np.linalg.norm(arrays['core.w_in'] - arrays['core.stable_in']) > 0
                    else:
                        assert np.count_nonzero(arrays['importance']) == 0
                    checkpoints += 1
        report['checks'].append('160full checkpoints have learnedimportance/anchor where enabled; recurrence/fixedheads/statistics unchanged')
        with np.load(source / 'DEV-predictions.npz', allow_pickle=False) as f:
            saved = {key: f[key].copy() for key in f.files}
        curves = json.loads((source / 'curves.json').read_text(encoding='utf-8'))
        summary = {}
        for seed in manifest['seeds']:
            summary[str(seed)] = {}
            for rule in ('protected', 'norm_matched', 'unprotected'):
                curve = []
                for stage in range(20):
                    p = saved[f'{seed}_{rule}_stage{stage:02d}']
                    assert p.shape == (120, 6) and np.isfinite(p).all()
                    curve.append([float(np.mean(p.argmax(axis=1)[y[dev] == c] == c)) for c in range(6)])
                np.testing.assert_array_equal(curve, curves[str(seed)][rule])
                trajectory = np.asarray(curve)
                forget = np.array([max(0, trajectory[c:, c].max() - trajectory[-1, c]) for c in range(6)])
                summary[str(seed)][rule] = {'accuracy': float(trajectory[-1].mean()), 'mean_forgetting': float(forget.mean()), 'worst_forgetting': float(forget.max())}
                expected = original['summary'][str(seed)][rule]
                assert expected['final_DEV_accuracy'] == summary[str(seed)][rule]['accuracy']
                np.testing.assert_array_equal(forget, expected['per_intent_forgetting'])
        report['checks'].append('all180DEVcurves independently rescored; inheritedseed260113 baseline explicitly tracked')
        # Paired seed/text uncertainty on DEV; not fresh test confirmation.
        effects = {}
        protection = np.stack([saved[f'{s}_protected_stage19'].argmax(axis=1) == y[dev] for s in manifest['seeds']])
        for rule in ('unprotected', 'norm_matched'):
            control = np.stack([saved[f'{s}_{rule}_stage19'].argmax(axis=1) == y[dev] for s in manifest['seeds']])
            difference = protection.astype(float) - control
            rng, sample_values = np.random.default_rng(260216), []
            for _ in range(10000):
                seeds = rng.integers(3, size=3)
                items = np.concatenate([c * 20 + rng.integers(20, size=20) for c in range(6)])
                sample_values.append(float(difference[np.ix_(seeds, items)].mean()))
            effects[rule] = {'accuracy_gain': float(difference.mean()), 'CI95': np.quantile(sample_values, [.025, .975]).tolist()}
        report.update(verdict='METAPLASTICITY_RAW_DATA_AUDIT_PASS_NOT_QUALITY_PASS', summary=summary, effects=effects,
            protected_all_seed_gate=original['protected_all_seed_acquisition_retention_gate'],
            checkpoints_verified=checkpoints, CPU_training=False)
    except Exception as exc:
        report['failures'].append(type(exc).__name__ + ': ' + str(exc))
        report['verdict'] = 'METAPLASTICITY_DATA_AUDIT_STOPPED_NOT_PASS'
    (run / 'result.json').write_text(encode(report), encoding='utf-8')
    (run / 'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT / 'local/current-semantic-metaplasticity-audit.txt').write_text(str(run), encoding='utf-8')
    trace = Store(ROOT / 'local/hybrid-v1')
    trace.event('work_trace.activity', {'title': 'Audit protection/norm-control từ dữ liệu thật', 'run_id': run.name, **report}, 'orchestrator')
    trace.close()
    print(encode({'run': str(run), **report}), flush=True)
    if report['failures']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
