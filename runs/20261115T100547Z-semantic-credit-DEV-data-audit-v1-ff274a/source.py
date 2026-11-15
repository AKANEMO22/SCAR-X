"""Recover/rescore saved DEV components after terminal JSON bool failure.

No training, GPUforward or TEST scoring; original failed invocation is retained.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib, io, json, sys, uuid
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.hybrid.store import Store, encode


def unpack(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(f['metadata'].tobytes())
        with np.load(io.BytesIO(f['model'].tobytes()), allow_pickle=False) as inner:
            arrays = {k: inner[k].copy() for k in inner.files if k != 'metadata'}
    return meta, arrays


def main():
    source = Path((ROOT / 'local/current-semantic-credit-dev.txt').read_text(encoding='utf-8').strip())
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-credit-DEV-data-audit-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    result = {'source_run': source.name, 'GPU_or_API_calls': 0, 'SNN_updates': 0,
              'original_terminal_report_failed': True, 'original_exit_code': 1,
              'original_exception': 'TypeError: Object of type bool is not JSON serializable (numpy.bool_)',
              'wholeprotocol_PASS': False, 'quality_confirmation': False, 'failures': [], 'checks': []}
    try:
        assert not (source / 'result.json').exists()
        for name, digest in manifest['sources'].items():
            assert hashlib.sha256((source / Path(name).name).read_bytes()).hexdigest() == digest
        rows = [json.loads(line) for line in (source / 'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        counts = Counter(r['kind'] for r in rows)
        assert counts == {'update': 1446, 'forward': 9126}
        for kind in counts:
            assert [r['number'] for r in rows if r['kind'] == kind] == list(range(1, counts[kind] + 1))
        result['charged_counts'] = dict(counts)
        result['first_to_last_attempt_seconds'] = rows[-1]['epoch'] - rows[0]['epoch']
        assert result['first_to_last_attempt_seconds'] < 180
        result['checks'].append('source/ledger match;1446updates/9126forwards charged, no rerun')
        feature = ROOT / 'runs' / manifest['feature_run']
        with np.load(feature / 'features.npz', allow_pickle=False) as f:
            dev, y = f['dev'], f['labels']
        data = json.loads((feature / 'data.json').read_text(encoding='utf-8'))
        losses = [json.loads(line) for line in (source / 'losses.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(losses) == 1440
        assert all(data[r['sample_index']]['split'] == 'TRAIN' and r['target'] == int(y[r['sample_index']])
                   and r['device'] == 'gpu' for r in losses)
        trajectories = json.loads((source / 'curves.json').read_text(encoding='utf-8'))
        with np.load(source / 'DEV-predictions.npz', allow_pickle=False) as f:
            saved = {key: f[key].copy() for key in f.files}
        summary = {}
        artifact = ROOT / 'artifacts' / source.name
        for name in ('hidden_eta05', 'hidden_eta20', 'readout_only'):
            score_curves = []
            for stage in range(20):
                p = saved[f'{name}_DEV_stage{stage:02d}']
                assert p.dtype == np.float32 and p.shape == (120, 6) and np.isfinite(p).all()
                np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
                score_curves.append([float(np.mean(p.argmax(axis=1)[y[dev] == c] == c)) for c in range(6)])
                meta, arrays = unpack(artifact / f'{name}-stage{stage:02d}.npz')
                assert meta['updates'] == 24 * (stage + 1)
                assert meta['hidden_updates'] == (0 if name == 'readout_only' else meta['updates'])
                assert all(np.isfinite(v).all() for v in arrays.values())
                if stage == 0:
                    reference_rec = arrays['core.w_rec'].copy()
                np.testing.assert_array_equal(reference_rec, arrays['core.w_rec'])
                np.testing.assert_array_equal(arrays['f_mean'], np.zeros(128, dtype=np.float32))
                np.testing.assert_array_equal(arrays['f_var'], np.ones(128, dtype=np.float32))
                if name != 'readout_only':
                    np.testing.assert_array_equal(arrays['core.w_out'], arrays['core.stable_out'])
                else:
                    np.testing.assert_array_equal(arrays['core.w_in'], arrays['core.stable_in'])
            np.testing.assert_array_equal(score_curves, trajectories[name])
            trajectory = np.asarray(score_curves)
            forgotten = np.asarray([max(0, trajectory[c:, c].max() - trajectory[-1, c]) for c in range(6)])
            accuracy = float(trajectory[-1].mean())
            summary[name] = {'final_DEV_accuracy': accuracy, 'mean_forgetting': float(forgotten.mean()),
                'worst_forgetting': float(forgotten.max()), 'per_intent_forgetting': forgotten.tolist(),
                'pilot_gate': bool(accuracy >= .65 and forgotten.mean() <= .03 + 1e-9 and forgotten.max() <= .05 + 1e-9)}
        result['checks'].append('all60DEV matrices/checkpoints rescored; fixed recurrence/statistics/head checks match')
        result.update(verdict='SAVED_DEV_COMPONENTS_RECOVERED_ORIGINAL_PROTOCOL_NOT_PASS',
            summary=summary, any_hidden_pilot_gate=any(summary[k]['pilot_gate'] for k in ('hidden_eta05', 'hidden_eta20')),
            frozen_DEV_accuracy=float(np.mean(saved['frozen_DEV'].argmax(axis=1) == y[dev])),
            TEST_calls=0, CPU_training=False, retain_original_failure=True)
    except Exception as exc:
        result['failures'].append(type(exc).__name__ + ': ' + str(exc))
        result['verdict'] = 'DEV_DATA_AUDIT_STOPPED_NOT_PASS'
    (run / 'result.json').write_text(encode(result), encoding='utf-8')
    (run / 'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT / 'local/current-semantic-credit-dev-audit.txt').write_text(str(run), encoding='utf-8')
    trace = Store(ROOT / 'local/hybrid-v1')
    trace.event('work_trace.activity', {'title': 'Kiểm toán DEV GPU: giữ lỗi xuất report gốc',
        'run_id': run.name, **result}, 'orchestrator')
    trace.close()
    print(encode({'run': str(run), **result}), flush=True)
    if result['failures']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
