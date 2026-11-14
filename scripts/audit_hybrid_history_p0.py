"""Recalculate recorded P0 math/ledger and threshold evidence; no new learning."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.evaluation.history_ablation_p0 import P0Budget


def main():
    run = Path((ROOT / 'local/current-history-ablation-p0-v1.txt').read_text().strip())
    manifest = json.loads((run / 'manifest.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    diagnostic = json.loads((run / 'parity-failure-diagnostic/result.json').read_text())
    assert result['status'] == 'P0_STOPPED_AT_REGISTERED_GATE_NOT_SUCCESS'
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((run / 'fixtures.npz').read_bytes()).hexdigest() == manifest['fixture_sha256']
    ledger = P0Budget(ROOT / 'local/history-ablation-p0-v1-budget.jsonl').records()
    assert (run / 'charged-attempts-with-diagnostic.jsonl').read_bytes() == (ROOT / 'local/history-ablation-p0-v1-budget.jsonl').read_bytes()
    counts = Counter((r['arm'], r['seed']) for r in ledger)
    assert len(ledger) == 177 and max(counts.values()) == 87
    assert ledger[-1]['epoch'] - ledger[0]['epoch'] <= 300
    assert {r['seed'] for r in ledger} == {260313}
    steps = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    assert len(steps) == 57 and max(s['CPU_GPU_max_array_abs_error'] for s in steps) < 3e-5
    audited = 0
    for step in steps:
        if not step.get('arrays'):
            continue
        with np.load(run / step['arrays'], allow_pickle=False) as z:
            assert step['arm'] == 'current_hidden'
            energy = .99 * z['energy'] + .01 * z['direction'] ** 2
            trace = .9 * z['trace'] + .1 * z['direction']
            delta = np.clip(.008 * trace / (np.sqrt(energy) + 1e-4), -.01, .01)
            for key, value in [('next_energy', energy), ('next_trace', trace), ('delta', delta)]:
                np.testing.assert_allclose(z[key], value, rtol=3e-5, atol=3e-5)
            effective = (np.clip(z['weights'] + delta, -3, 3) - z['weights']).astype(np.float64)
            assert abs(np.linalg.norm(effective) - step['actual_norm']) < 3e-5
        audited += 1
    assert audited == 17
    tick_records = []
    for path in sorted((run / 'parity-failure-diagnostic').glob('tick-*.npz')):
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['cpu_spikes'], (z['cpu_margin'] >= 0).astype(np.float32))
            np.testing.assert_array_equal(z['gpu_spikes'], (z['gpu_margin'] >= 0).astype(np.float32))
            changed = np.flatnonzero(z['cpu_spikes'] != z['gpu_spikes'])
            if len(changed):
                tick_records.append({'tick': int(path.stem.split('-')[-1]), 'neurons': changed.tolist(),
                                     'CPU_margin': z['cpu_margin'][changed].tolist(),
                                     'GPU_margin': z['gpu_margin'][changed].tolist()})
    assert tick_records == [{'tick': 7, 'neurons': [91], 'CPU_margin': [0.0],
                            'GPU_margin': [-1.7881393432617188e-07]}]
    probabilities = np.asarray(diagnostic['CPU_GPU_probabilities'], np.float64)
    losses = -np.log(probabilities[:, 1])
    np.testing.assert_allclose(losses, diagnostic['CPU_GPU_losses'], rtol=1e-12, atol=1e-12)
    assert abs(losses[0] - losses[1]) > manifest['parity_rtol_atol']
    report = {'verdict': 'P0_RECORDED_PARITY_GATE_FAILURE_CONFIRMED_NOT_PASS',
              'actual_attempted_model_calls_including_diagnostic': len(ledger),
              'maximum_attempts_arm_seed': max(counts.values()),
              'charged_call_span_seconds': ledger[-1]['epoch'] - ledger[0]['epoch'],
              'charged_seeds': sorted({r['seed'] for r in ledger}),
              'intended_two_seed_full_matrix_completed': False,
              'completed_control_cells': [c['arm'] for c in result['cells']],
              'audited_hidden_math_records': audited,
              'completed_GPU_steps_before_failure': len(steps),
              'first_spike_divergence_in_failed_step': tick_records[0],
              'recomputed_loss_gap': float(abs(losses[0] - losses[1])),
              'current_only_and_matched_learning_trajectories_executed': False,
              'unchanged_threshold_not_loosened_after_failure': True,
              'resource_scope': result['resource_scope'],
              'peak_research_process_RSS_bytes': result['research_process_peak_RSS'],
              'sampled_physical_free_GPU_min_MiB': min(v['free_MiB'] for v in result['GPU_point_samples']),
              'new_forward_backward_or_learning_in_this_audit': 0,
              'P0_quality_retention_or_production_PASS': False,
              'interpretation': 'FP32 near-threshold hard spike divergence observed; not evidence CPU or GPU semantics is superior, nor proof of a hidden-history benefit'}
    target = run / 'independent-data-audit.json'
    with target.open('x', encoding='utf-8') as handle:json.dump(report, handle, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
