"""Independent raw-data rescore and local-weight replay; no CUDA/model API calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib, io, json, re, sys, uuid
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.hybrid.store import Store, encode


def nested(path):
    with np.load(path, allow_pickle=False) as outer:
        meta = json.loads(outer['metadata'].tobytes())
        counts = outer['counts'].copy()
        with np.load(io.BytesIO(outer['model'].tobytes()), allow_pickle=False) as inner:
            arrays = {key: inner[key].copy() for key in inner.files if key != 'metadata'}
    return meta, counts, arrays


def main():
    source = Path((ROOT / 'local/current-semantic-local-ALIF.txt').read_text(encoding='utf-8').strip())
    result = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-local-data-audit-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    report = {'source_run': source.name, 'verdict': 'AUDIT_INCOMPLETE', 'GPU_or_API_calls': 0,
              'new_training': False, 'whole_assistant_superiority': False, 'checks': [], 'failures': []}
    checks = report['checks']
    try:
        for name, digest in manifest['sources'].items():
            assert hashlib.sha256((source / Path(name).name).read_bytes()).hexdigest() == digest
        checks.append('archived source/protocol hashes match pre-run manifest')
        attempts = [json.loads(line) for line in (source / 'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        counts = Counter(row['kind'] for row in attempts)
        assert dict(counts) == result['charged_counts']
        for kind in counts:
            assert [r['number'] for r in attempts if r['kind'] == kind] == list(range(1, counts[kind] + 1))
        assert counts['update'] <= 4500 and counts['forward'] <= 40000
        assert result['elapsed_seconds'] <= 1200
        checks.append('persistent counters contiguous within unchanged global quotas')
        if not result['completed']:
            raise ValueError('Source run stopped; not a completed quality/retention study')
        features = ROOT / 'runs' / manifest['features_run']
        assert hashlib.sha256((features / 'features.npz').read_bytes()).hexdigest() == manifest['features_sha256']
        rows = json.loads((features / 'data.json').read_text(encoding='utf-8'))
        normalize = lambda s: re.sub(r'\W+', ' ', s.casefold()).strip()
        sets = {split: {normalize(r['text']) for r in rows if r['split'] == split} for split in ('TRAIN', 'DEV', 'TEST')}
        assert not (sets['TRAIN'] & sets['DEV'] or sets['TRAIN'] & sets['TEST'] or sets['DEV'] & sets['TEST'])
        assert len(sets['TEST']) == 240
        checks.append('TRAIN/DEV/TEST text clusters disjoint;240unique officialTEST rows')
        with np.load(features / 'features.npz', allow_pickle=False) as data:
            x, raw, labels, train, dev, mean, basis = (data[k].copy() for k in ('x', 'raw', 'labels', 'train', 'dev', 'mean', 'basis'))
        np.testing.assert_allclose(mean, raw[train].mean(axis=0), rtol=1e-6, atol=1e-7)
        expected = (raw - mean) @ basis
        expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-8)
        np.testing.assert_array_equal(expected, x)
        # TRAIN subspace check without relying on sign convention of individual PCs.
        _, _, v = np.linalg.svd(raw[train] - mean, full_matrices=False)
        np.testing.assert_allclose(basis @ basis.T, v[:64].T @ v[:64], rtol=2e-4, atol=2e-5)
        checks.append('PCA/normalization reproduced from TRAIN only; no TEST fit')
        schedules = json.loads((source / 'schedules.json').read_text(encoding='utf-8'))
        losses = [json.loads(line) for line in (source / 'losses.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(losses) == 4320
        assert all(r['id'] == rows[r['sample_index']]['id'] and rows[r['sample_index']]['split'] == 'TRAIN' and
                   r['target'] == int(labels[r['sample_index']]) for r in losses)
        assert all(r['device'] == 'gpu' and np.isfinite(r['loss']) and r['weight_delta_L2'] >= 0 for r in losses)
        checks.append('all4320 formal updates use only TRAIN and report actual GPU/deltas')
        artifact = ROOT / result['artifacts']
        local_replay = []
        for seed in manifest['seeds']:
            stages = schedules[str(seed)]
            assert len(stages) == 20
            for arm in ('prototype', 'readout_only', 'shuffled'):
                records = [r for r in losses if r['seed'] == seed and r['arm'] == arm]
                assert len(records) == 480
                expected_stream = [i for st in stages for key in ('current', 'replay_samples') for i in st[key]]
                assert [r['sample_index'] for r in records] == expected_stream
                meta, count, final = nested(artifact / f'{seed}-{arm}-stage19.npz')
                assert meta['updates'] == 480 and meta['controller'] == stages[-1]['controller']
                assert meta['hidden_updates'] == (0 if arm == 'readout_only' else 480)
                assert final['core.w_in'].dtype == np.float32 and final['core.w_rec'].dtype == np.float32
                rng = np.random.default_rng(seed)
                indices = np.stack([rng.choice(128, 8, replace=False) for _ in range(128)]).astype(np.int32)
                weights = rng.normal(.15, .45, (128, 64)).astype(np.float32)
                weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-8)
                recurrent = rng.normal(0, .12 / np.sqrt(8), (128, 8)).astype(np.float32)
                np.testing.assert_array_equal(final['core.indices'], indices)
                np.testing.assert_array_equal(final['core.w_rec'], recurrent)
                np.testing.assert_array_equal(final['f_mean'], np.zeros(128, dtype=np.float32))
                np.testing.assert_array_equal(final['f_var'], np.ones(128, dtype=np.float32))
                if arm == 'readout_only':
                    np.testing.assert_array_equal(weights, final['core.w_in'])
                    assert count.sum() == 0
                    continue
                counters = np.zeros(128, dtype=np.int64)
                signal = np.random.default_rng(seed + 93_000_000)
                for r in records:
                    target = int(signal.integers(6)) if arm == 'shuffled' else r['target']
                    assert r['plasticity_label'] == target
                    pool = np.arange(target * 21, (target + 1) * 21)
                    used, unused = pool[counters[pool] > 0], pool[counters[pool] == 0]
                    vector = x[r['sample_index']]
                    if len(used):
                        drive = weights[used] @ vector
                        arg = int(np.argmax(drive))
                        winner, sim = int(used[arg]), float(drive[arg])
                    else:
                        winner, sim = None, -1.
                    chosen = int(unused[0]) if len(unused) and (winner is None or sim < .95) else winner
                    assert chosen == r['chosen_neuron']
                    before = weights[chosen].copy()
                    n = int(counters[chosen])
                    value = vector.copy() if n == 0 else before + np.float32(1 / (n + 1)) * (vector - before)
                    value /= max(np.linalg.norm(value), np.float32(1e-8))
                    weights[chosen] = value
                    counters[chosen] += 1
                    assert abs(float(np.linalg.norm(value - before)) - r['weight_delta_L2']) < 3e-5
                np.testing.assert_array_equal(counters, count)
                np.testing.assert_allclose(weights, final['core.w_in'], rtol=3e-5, atol=3e-5)
                local_replay.append({'seed': seed, 'arm': arm, 'updates_reconstructed': 480,
                    'max_weight_error': float(np.max(np.abs(weights - final['core.w_in'])))})
        checks.append('2880hidden updates independently replayed; recurrent weights/statistics unchanged')
        report['hidden_rule_replay'] = local_replay
        test = np.array([i for i, r in enumerate(rows) if r['split'] == 'TEST'])
        target = labels[test]
        with np.load(source / 'predictions.npz', allow_pickle=False) as f:
            predictions = {k: f[k].copy() for k in f.files}
        accuracies = {arm: float(np.mean([predictions[f'{s}_{arm}'].argmax(axis=1) == target
                      for s in manifest['seeds']])) for arm in manifest['arms']}
        assert accuracies == result['TEST_accuracy']
        curves = json.loads((source / 'curves.json').read_text(encoding='utf-8'))
        for arm in manifest['arms']:
            for seed_index, seed in enumerate(manifest['seeds']):
                score_curve = []
                for stage in range(20):
                    p = np.load(source / f'DEV-{seed}-{arm}-stage{stage:02d}.npy', allow_pickle=False)
                    score_curve.append([float(np.mean(p.argmax(axis=1)[labels[dev] == c] == c)) for c in range(6)])
                np.testing.assert_array_equal(np.asarray(score_curve), curves[str(seed)][arm])
                trajectory = np.asarray(score_curve)
                forgotten = [max(0, float(trajectory[c:, c].max() - trajectory[-1, c])) for c in range(6)]
                expected = result['DEV_forgetting20'][arm][seed_index]
                np.testing.assert_array_equal(forgotten, expected['per_intent'])
        checks.append('TEST accuracies and all2400per-intent stage scores independently rescored')
        rng = np.random.default_rng(260216)
        seed_draws, task_draws = [], []
        for _ in range(10000):
            seed_draws.append(rng.integers(3, size=3))
            task_draws.append(np.concatenate([c * 40 + rng.integers(40, size=40) for c in range(6)]))
        seed_draws, task_draws = np.asarray(seed_draws), np.asarray(task_draws)
        proto = np.stack([predictions[f'{s}_prototype'].argmax(axis=1) == target for s in manifest['seeds']])
        for name, item in result['comparisons'].items():
            if name == 'hidden_reset':
                key = 'reset_prototype'
            elif name in manifest['arms']:
                key = name
            else:
                key = 'control_' + name.replace('/', '_')
            values = [predictions[f'{s}_{key}'] for s in manifest['seeds']]
            control = np.stack([(v.argmax(axis=1) if v.ndim == 2 else v) == target for v in values])
            diff = proto.astype(float) - control
            samples = diff[seed_draws[:, :, None], task_draws[:, None, :]].mean(axis=(1, 2))
            ci = np.quantile(samples, [.025, .975])
            np.testing.assert_allclose(ci, item['CI95'], atol=1e-12, rtol=1e-12)
            assert abs(float(diff.mean()) - item['gain']) < 1e-12
            assert abs(float(control.mean()) - item['control_accuracy']) < 1e-12
        checks.append('all13 paired stratified seed/text confidence intervals independently recomputed')
        # Dataset/encoder limitation and inductive bias are part of the verdict.
        acquired = accuracies['prototype'] >= .65
        retained = all(r['mean'] <= .03 + 1e-9 and r['worst'] <= .05 + 1e-9 for r in result['DEV_forgetting20']['prototype'])
        gains = all(r['gain'] >= .05 and r['CI95'][0] > 0 for r in result['comparisons'].values())
        assert result['module_gates_pass'] == (acquired and retained and gains)
        assert result['acquisition_gate'] == acquired and result['retention_gate'] == retained
        assert result['all_control_and_attribution_gain_gate'] == gains
        report.update(verdict='RAW_REPLAY_AND_METRIC_AUDIT_PASS_NOT_QUALITY_PASS', TEST_accuracy=accuracies,
                      module_gates_pass=result['module_gates_pass'], completed_source=True)
    except Exception as exc:
        report['failures'].append(type(exc).__name__ + ': ' + str(exc))
        report['verdict'] = 'AUDIT_STOPPED_NOT_PASS'
    (run / 'result.json').write_text(encode(report), encoding='utf-8')
    (run / 'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT / 'local/current-semantic-local-audit.txt').write_text(str(run), encoding='utf-8')
    trace = Store(ROOT / 'local/hybrid-v1')
    trace.event('work_trace.activity', {'title': 'Kiểm toán ALIF từ raw data', 'run_id': run.name, **report}, 'orchestrator')
    trace.close()
    print(encode({'run': str(run), **report}), flush=True)
    if report['failures']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
