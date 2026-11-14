"""Registered numerical P0 only; no DEV, language data or live model update."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.store import Store
from scarx.hybrid.device_budget import gpu_headroom
from scarx.evaluation.history_ablation_p0 import (
    ARMS, SEEDS, BASE_SHA, P0Budget, HistoryALIF, copied_learn_source,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def live_snapshot():
    with sqlite3.connect((ROOT / 'local/hybrid-v1/hybrid.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        pointer = json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()[0])
        digest, payload = db.execute('SELECT digest,payload FROM checkpoints WHERE id=?', (pointer['id'],)).fetchone()
    assert hashlib.sha256(payload).hexdigest() == digest
    return payload


def main():
    budget = P0Budget(ROOT / 'local/history-ablation-p0-v1-budget.jsonl')
    if budget.path.exists():
        raise RuntimeError('Registered P0 already has charged attempts; inspect before any rerun, never reset')
    suite = ET.parse(ROOT / 'local/history-ablation-p0-pure-fixed-20261115.xml').getroot().find('testsuite')
    assert suite.attrib['tests'] == '8' and suite.attrib['failures'] == suite.attrib['errors'] == '0'
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-history-ablation-p0-v1')
    run.mkdir(exist_ok=False)
    checkpoints = ROOT / 'artifacts' / run.name
    checkpoints.mkdir(exist_ok=False)
    sources = [Path(__file__), ROOT / 'src/scarx/evaluation/history_ablation_p0.py',
               ROOT / 'src/scarx/hybrid/learner.py', ROOT / 'src/scarx/hybrid/cuda_alif.py',
               ROOT / 'src/scarx/neural/core.py', ROOT / 'tests/test_history_ablation_p0.py',
               ROOT / 'experiments/hybrid-update-history-p0-v1.md']
    pins = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        target = run / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        pins[relative.as_posix()] = sha(path)
    (run / 'copied-learn.py.source').write_text(copied_learn_source(), encoding='utf-8')
    fixture = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 17)
        fixture[str(seed)] = rng.normal(0, .2, (20, 8, 64)).astype(np.float32)
    np.savez(run / 'fixtures.npz', **fixture)
    manifest = {'run_id': run.name, 'source_sha256': pins, 'base_sha256': BASE_SHA,
                'fixture_sha256': sha(run / 'fixtures.npz'), 'arms': ARMS, 'seeds': SEEDS,
                'max_attempts_total': 600, 'max_attempts_arm_seed': 100, 'max_seconds': 300,
                'intended_attempts': 490, 'parity_rtol_atol': 3e-5,
                'matched_relative_norm_tolerance': .01, 'production_mode_switch': False,
                'research_only_not_DEV': True}
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (ROOT / 'local/current-history-ablation-p0-v1.txt').write_text(str(run), encoding='utf-8')
    trace = Store(ROOT / 'local/hybrid-v1')
    trace.event('work_trace.activity', {'kind': 'P0_manifest_locked', **manifest}, 'orchestrator')
    before_live = live_snapshot()
    private = ROOT / 'local/history-ablation-p0-v1'
    private.mkdir(exist_ok=False)
    (private / 'private-live-before.npz').write_bytes(before_live)
    stop = threading.Event()
    memory = []
    def sample():
        while not stop.wait(.1):
            value = psutil.Process().memory_full_info()
            memory.append({'rss': value.rss, 'private': getattr(value, 'private', None)})
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    records, cells, GPU_points = [], [], []
    start = time.monotonic()
    report = {'run_id': run.name, 'status': 'STARTED', 'cells': cells,
              'new_VLM_ASR_calls': 0, 'live_learning_or_rule_changed': False,
              'P0_only_not_quality_or_retention_PASS': True}
    def point():
        value = gpu_headroom()
        GPU_points.append(value)
        if value['free_MiB'] < 512:
            raise MemoryError('Physical GPU reserve below registered512MiB')
    try:
        point()
        for seed in SEEDS:
            sequences = fixture[str(seed)]
            for arm in ARMS:
                point()
                cpu = HistoryALIF(seed, arm, budget)
                gpu = HistoryALIF(seed, arm, budget).to_device('gpu')
                assert all(type(a).__module__.startswith('cupy') for a in gpu.arrays().values())
                original = LocalALIF(seed) if arm == 'current_hidden' else None
                initial = {k: gpu.host(v).copy() for k, v in gpu.core.arrays().items() if k.startswith('w_')}
                for step, sequence in enumerate(sequences):
                    evidence = {'kind': 'machine_verifier', 'id': f'P0-numerical-fixture:{seed}:{step}'}
                    kwargs = {'head': 'relevance', 'evidence': evidence}
                    expected = cpu.learn(sequence, step % 2, **kwargs)
                    actual = gpu.learn(sequence, step % 2, **kwargs)
                    assert abs(expected['loss'] - actual['loss']) < 3e-5
                    max_error = 0.0
                    for key, reference in cpu.arrays().items():
                        measured = gpu.host(gpu.arrays()[key])
                        assert reference.dtype == measured.dtype
                        np.testing.assert_allclose(reference, measured, rtol=3e-5, atol=3e-5, err_msg=f'{seed}/{arm}/{step}/{key}')
                        max_error = max(max_error, float(np.max(np.abs(reference - measured))))
                    if original:
                        budget.charge(arm, seed, 'production_reference')
                        baseline = original.learn(sequence, step % 2, **kwargs)
                        assert baseline['loss'] == expected['loss']
                        for key, value in original.arrays().items():np.testing.assert_array_equal(value, cpu.arrays()[key])
                    record = {'seed': seed, 'arm': arm, 'step': step,
                              'actual_GPU_loss': actual['loss'], 'actual_GPU_weight_delta': actual['weight_delta'],
                              'CPU_GPU_max_array_abs_error': max_error, 'GPU_updates': gpu.updates,
                              'GPU_hidden_updates': gpu.hidden_updates, 'actual_lab_not_live_update': True}
                    if gpu.last_math:
                        math = gpu.last_math
                        path = run / 'arrays' / f'{seed}-{arm}-{step}.npz'
                        path.parent.mkdir(exist_ok=True)
                        np.savez(path, **{k: v for k, v in math.items() if isinstance(v, np.ndarray)})
                        record.update({k: v for k, v in math.items() if not isinstance(v, np.ndarray)})
                        record['arrays'] = path.relative_to(run).as_posix()
                    records.append(record)
                    trace.event('work_trace.activity', {'kind': 'P0_actual_GPU_update', 'run_id': run.name, **record}, 'orchestrator')
                    with (run / 'steps.jsonl').open('a', encoding='utf-8') as handle:handle.write(json.dumps(record) + '\n')
                weight_deltas = {k: float(np.linalg.norm(gpu.host(getattr(gpu.core, k)) - v)) for k, v in initial.items()}
                if arm not in ('frozen', 'readout_only'):
                    assert weight_deltas['w_in'] > 0 and weight_deltas['w_rec'] > 0
                else:
                    assert weight_deltas['w_in'] == weight_deltas['w_rec'] == 0
                payload = gpu.dumps({'purpose': 'P0-exact-continuation'})
                (checkpoints / f'{seed}-{arm}.research.zip').write_bytes(payload)
                restored, controller = HistoryALIF.loads(payload, budget=budget, expected_arm=arm, device='gpu')
                assert restored.dumps(controller) == payload
                next_sequence = sequences[0]
                kwargs = {'head': 'relevance', 'evidence': {'kind': 'machine_verifier', 'id': 'P0-next-update'}}
                gpu.learn(next_sequence, 1, **kwargs)
                restored.learn(next_sequence, 1, **kwargs)
                assert gpu.dumps() == restored.dumps()
                if arm != 'frozen':
                    gpu.consolidate()
                    restored.consolidate()
                else:
                    # A frozen model has no replay; two explicit no-op calls
                    # preserve the registered 490-call execution accounting.
                    gpu.learn(next_sequence, 1, **kwargs)
                    restored.learn(next_sequence, 1, **kwargs)
                assert gpu.dumps() == restored.dumps()
                rollback = gpu.dumps({'purpose': 'rollback'})
                gpu.learn(next_sequence, 0, **kwargs)
                gpu, controller = HistoryALIF.loads(rollback, budget=budget, expected_arm=arm, device='gpu')
                assert gpu.dumps(controller) == rollback
                cell = {'seed': seed, 'arm': arm, 'CPU_GPU_parity': True,
                        'same_device_cold_next_update_replay_and_rollback_exact': True,
                        'weight_deltas_after_twenty_GPU_calls': weight_deltas}
                cells.append(cell)
                (run / 'partial-cells.json').write_text(json.dumps(cells, indent=2), encoding='utf-8')
        attempts = budget.records()
        assert len(attempts) == 490
        assert time.time() - attempts[0]['epoch'] <= 300
        assert all(sha(ROOT / name) == digest for name, digest in pins.items())
        after_live = live_snapshot()
        (private / 'private-live-after.npz').write_bytes(after_live)
        with np.load(io.BytesIO(before_live), allow_pickle=False) as before, np.load(io.BytesIO(after_live), allow_pickle=False) as after:
            bm = json.loads(before['metadata'].tobytes());am = json.loads(after['metadata'].tobytes())
            assert bm['updates'] == am['updates']
            for key in ('core.w_in', 'core.w_rec', 'core.w_out', 'core.bias'):
                np.testing.assert_array_equal(before[key], after[key])
        report.update(status='P0_NUMERICS_PERSISTENCE_PASS_NOT_DEV_OR_LEARNING_QUALITY',
                      actual_attempts=len(attempts), attempts_by_arm_seed=dict(Counter(f"{r['arm']}/{r['seed']}" for r in attempts)),
                      live_updates_unchanged=am['updates'])
    except Exception as exc:
        report.update(status='P0_STOPPED_AT_REGISTERED_GATE_NOT_SUCCESS',
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                      actual_attempts=len(budget.records()))
    finally:
        stop.set();thread.join(timeout=2)
        report.update(seconds=time.monotonic()-start, GPU_point_samples=GPU_points,
                      research_process_peak_RSS=max((x['rss'] for x in memory),default=0),
                      research_process_peak_private_commit=max((x['private'] or 0 for x in memory),default=0),
                      resource_scope='research process only; VLM/Hermes not included, GPU points not continuous peaks')
        if budget.path.exists():
            (run / 'charged-attempts.jsonl').write_bytes(budget.path.read_bytes())
        (run / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        trace.event('work_trace.activity', {'kind': 'P0_completed', **report}, 'orchestrator')
        trace.close()
        print(json.dumps(report), flush=True)
    return 0 if report['status'].startswith('P0_NUMERICS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
