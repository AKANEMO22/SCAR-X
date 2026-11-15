"""Three seeds, matched update magnitude, GPU-only local protection DEV study."""
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import subprocess, sys, time, uuid
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.evaluation.semantic_metaplastic_alif import ProtectedALIF
from scarx.hybrid.store import Store, encode
from run_semantic_contrastive_credit_dev import summarize_curve


class Budget:
    def __init__(self, run):
        self.run, self.started = run, time.monotonic()
        self.counts, self.resources, self.last = {'update': 0, 'forward': 0}, [], 0.
        self.file = (run / 'attempts.jsonl').open('x', encoding='utf-8')
    def charge(self, kind):
        if self.counts[kind] >= {'update': 4000, 'forward': 30000}[kind] or time.monotonic() - self.started > 240:
            raise TimeoutError('Metaplasticity DEV quota')
        self.counts[kind] += 1
        self.file.write(encode({'kind': kind, 'number': self.counts[kind], 'epoch': time.time()}) + '\n')
        self.file.flush()
        if time.monotonic() - self.last > 5:
            rss = psutil.Process().memory_info().rss
            free = int(subprocess.check_output([str(Path(os.environ['WINDIR']) / 'System32/nvidia-smi.exe'),
                '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW).strip()) * 1024**2
            self.resources.append({'RSS': rss, 'GPU_free': free})
            with (self.run / 'resources.jsonl').open('a', encoding='utf-8') as f:
                f.write(encode(self.resources[-1]) + '\n')
            self.last = time.monotonic()
            if rss > 2_000_000_000 or free < 512 * 1024**2:
                raise MemoryError('Metaplasticity physical guard')


def main():
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-metaplasticity-DEV-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    with (ROOT / 'local/semantic-metaplasticity-DEV-v1-execution.json').open('x', encoding='utf-8') as f:
        json.dump({'run': str(run)}, f)
    (ROOT / 'local/current-semantic-metaplasticity-DEV.txt').write_text(str(run), encoding='utf-8')
    artifact = ROOT / 'artifacts' / run.name
    artifact.mkdir()
    source = Path((ROOT / 'local/current-semantic-credit-dev.txt').read_text(encoding='utf-8').strip())
    feature_dir = Path((ROOT / 'local/current-semantic-features.txt').read_text(encoding='utf-8').strip())
    data = json.loads((feature_dir / 'data.json').read_text(encoding='utf-8'))
    with np.load(feature_dir / 'features.npz', allow_pickle=False) as f:
        train, dev = f['train'].copy(), f['dev'].copy()
        selected = np.concatenate((train, dev))
        x, labels = {int(i): f['x'][i].copy() for i in selected}, {int(i): int(f['labels'][i]) for i in selected}
    schedule_dir = Path((ROOT / 'local/current-semantic-local-ALIF.txt').read_text(encoding='utf-8').strip())
    schedules = json.loads((schedule_dir / 'schedules.json').read_text(encoding='utf-8'))
    sources = ('scripts/run_semantic_metaplasticity_dev.py', 'src/scarx/evaluation/semantic_metaplastic_alif.py',
               'src/scarx/evaluation/semantic_credit_alif.py', 'experiments/semantic-credit-metaplasticity-dev-v1.md')
    (run / 'manifest.json').write_text(encode({'source_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sources': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sources},
        'feature_run': feature_dir.name, 'reused_baseline_source': source.name,
        'schedule_source': schedule_dir.name, 'seeds': [260113, 260137, 260171],
        'CPU_training': False, 'TEST_calls': 0, 'max_updates': 4000, 'max_forwards': 30000, 'max_seconds': 240}), encoding='utf-8')
    for name in sources:
        (run / Path(name).name).write_bytes((ROOT / name).read_bytes())
    budget = Budget(run)
    trace = Store(ROOT / 'local/hybrid-v1')
    losses, arrays, curves = [], {}, {}
    report = {'completed': False, 'failures': [], 'CPU_training': False, 'TEST_calls': 0,
              'quality_confirmation': False, 'whole_or_module_PASS': False, 'main_or_live_changed': False}
    def predict(model):
        return np.stack([model.model.host(model.forward(x[int(i)])[0]) for i in dev])
    try:
        a = ProtectedALIF(260113, 'protected', budget)
        a.learn(np.eye(64, dtype=np.float32)[0], 0)
        payload = a.dumps({'replay': [0], 'stage': 'P0'})
        b, controller = ProtectedALIF.loads(payload, budget)
        assert b.dumps(controller) == payload
        for step in (1, 2):
            assert a.learn(np.eye(64, dtype=np.float32)[step], step) == b.learn(np.eye(64, dtype=np.float32)[step], step)
            assert a.dumps() == b.dumps()
        assert float(a.model.xp.sum(a.model.importance)) > 0
        matched = ProtectedALIF(260113, 'norm_matched', budget)
        record = matched.learn(np.eye(64, dtype=np.float32)[0], 0, .001)
        assert record['norm_match_error'] <= 1e-6
        report['GPU_P0_charged_counts'] = budget.counts.copy()
        for seed in (260113, 260137, 260171):
            curves[seed] = {}
            stages = schedules[str(seed)]
            protected = ProtectedALIF(seed, 'protected', budget)
            matched = ProtectedALIF(seed, 'norm_matched', budget)
            for rule in ('protected', 'norm_matched'):
                curves[seed][rule] = []
            for stage in stages:
                metrics = {rule: [] for rule in ('protected', 'norm_matched')}
                for kind in ('current', 'replay_samples'):
                    for index in stage[kind]:
                        assert data[index]['split'] == 'TRAIN'
                        a = protected.learn(x[index], labels[index])
                        b = matched.learn(x[index], labels[index], a['weight_delta_L2'])
                        for rule, result in (('protected', a), ('norm_matched', b)):
                            record = {'seed': seed, 'rule': rule, 'stage': stage['stage'],
                                      'sample_index': index, 'kind': kind, **result}
                            losses.append(record)
                            metrics[rule].append(record)
                for rule, model in (('protected', protected), ('norm_matched', matched)):
                    p = predict(model)
                    arrays[f'{seed}_{rule}_stage{stage["stage"]:02d}'] = p
                    prediction = p.argmax(axis=1)
                    scores = [float(np.mean(prediction[[labels[int(i)] == c for i in dev]] == c)) for c in range(6)]
                    curves[seed][rule].append(scores)
                    (artifact / f'{seed}-{rule}-stage{stage["stage"]:02d}.npz').write_bytes(model.dumps(stage['controller']))
                    trace.event('work_trace.activity', {'title': 'GPU importance và fast/stable synapses',
                        'run_id': run.name, 'seed': seed, 'rule': rule, 'stage': stage['stage'] + 1,
                        'DEV_accuracy': float(np.mean(scores)), 'DEV_by_intent': scores,
                        'loss': float(np.mean([r['loss'] for r in metrics[rule]])),
                        'weight_delta_L2_sum': sum(r['weight_delta_L2'] for r in metrics[rule]),
                        'importance_sum': metrics[rule][-1]['importance_sum'],
                        'anchor_drift': metrics[rule][-1]['anchor_drift_L2'],
                        'norm_match_error_max': max((r['norm_match_error'] or 0 for r in metrics[rule])),
                        'updates': model.credit.updates, 'device': 'CUDA0', 'TEST_calls': 0}, 'orchestrator')
            for rule, model in (('protected', protected), ('norm_matched', matched)):
                clone, _ = ProtectedALIF.loads(model.dumps(), budget)
                np.testing.assert_array_equal(arrays[f'{seed}_{rule}_stage19'], predict(clone))
            if seed == 260113:
                old_curves = json.loads((source / 'curves.json').read_text(encoding='utf-8'))
                curves[seed]['unprotected'] = old_curves['hidden_eta20']
                with np.load(source / 'DEV-predictions.npz', allow_pickle=False) as f:
                    for stage in range(20):
                        arrays[f'{seed}_unprotected_stage{stage:02d}'] = f[f'hidden_eta20_DEV_stage{stage:02d}'].copy()
            else:
                model = ProtectedALIF(seed, 'unprotected', budget)
                curves[seed]['unprotected'] = []
                for stage in stages:
                    for kind in ('current', 'replay_samples'):
                        for index in stage[kind]:
                            losses.append({'seed': seed, 'rule': 'unprotected', 'stage': stage['stage'],
                                'sample_index': index, 'kind': kind, **model.learn(x[index], labels[index])})
                    p = predict(model)
                    arrays[f'{seed}_unprotected_stage{stage["stage"]:02d}'] = p
                    prediction = p.argmax(axis=1)
                    scores = [float(np.mean(prediction[[labels[int(i)] == c for i in dev]] == c)) for c in range(6)]
                    curves[seed]['unprotected'].append(scores)
                    (artifact / f'{seed}-unprotected-stage{stage["stage"]:02d}.npz').write_bytes(model.dumps(stage['controller']))
                    trace.event('work_trace.activity', {'title': 'Đối chứng GPU không protection', 'run_id': run.name,
                        'seed': seed, 'stage': stage['stage'] + 1, 'DEV_accuracy': float(np.mean(scores)), 'device': 'CUDA0'}, 'orchestrator')
            if sum(p.stat().st_size for p in artifact.glob('*')) > 100_000_000:
                raise MemoryError('100MB research artifact cap')
        summary = {str(seed): {rule: summarize_curve(curve) for rule, curve in rules.items()} for seed, rules in curves.items()}
        if time.monotonic() - budget.started > 240:
            raise TimeoutError('Metaplasticity totalwall')
        report.update(completed=True, verdict='THREE_SEED_METAPLASTICITY_DEV_MEASURED_NOT_CONFIRMATION', summary=summary,
            protected_all_seed_acquisition_retention_gate=all(summary[str(s)]['protected']['pilot_acquisition_and_retention_gate'] for s in (260113, 260137, 260171)),
            max_norm_match_error=max(r['norm_match_error'] for r in losses if r['rule'] == 'norm_matched'),
            formal_new_updates=len(losses), reused_unprotected_updates=480)
    except Exception as exc:
        report['failures'].append(type(exc).__name__ + ': ' + str(exc))
        report['verdict'] = 'METAPLASTICITY_GPU_DEV_STOPPED_NOT_PASS'
    finally:
        (run / 'curves.json').write_text(encode(curves), encoding='utf-8')
        (run / 'losses.jsonl').write_text(''.join(encode(r) + '\n' for r in losses), encoding='utf-8')
        np.savez(run / 'DEV-predictions.npz', **arrays)
        report.update(charged_counts=budget.counts, elapsed_seconds=time.monotonic() - budget.started,
            artifacts=str(artifact.relative_to(ROOT)), peak_sampled_RSS=max((r['RSS'] for r in budget.resources), default=None),
            min_sampled_GPU_free=min((r['GPU_free'] for r in budget.resources), default=None))
        (run / 'result.json').write_text(encode(report), encoding='utf-8')
        trace.event('work_trace.activity', {'title': report['verdict'], 'run_id': run.name, **report}, 'orchestrator')
        trace.close()
        budget.file.close()
    print(encode({'run': str(run), **report}), flush=True)
    if not report['completed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
