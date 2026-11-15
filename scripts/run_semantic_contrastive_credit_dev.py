"""Two preregistered signed-credit rates, original ALIF, GPU-only DEV screen."""
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


def summarize_curve(curve):
    trajectory = np.asarray(curve)
    forgotten = np.array([max(0, trajectory[c:, c].max() - trajectory[-1, c]) for c in range(6)])
    score = float(trajectory[-1].mean())
    return {'final_DEV_accuracy': score, 'mean_forgetting': float(forgotten.mean()),
        'worst_forgetting': float(forgotten.max()), 'per_intent_forgetting': forgotten.tolist(),
        'pilot_acquisition_and_retention_gate': bool(score >= .65 and forgotten.mean() <= .03 + 1e-9 and forgotten.max() <= .05 + 1e-9)}


class Budget:
    def __init__(self, run):
        self.run, self.started = run, time.monotonic()
        self.counts = {'update': 0, 'forward': 0}
        self.resources, self.last_resource = [], 0.
        self.journal = (run / 'attempts.jsonl').open('x', encoding='utf-8')
    def charge(self, kind):
        if self.counts[kind] >= {'update': 1500, 'forward': 12000}[kind] or time.monotonic() - self.started > 180:
            raise TimeoutError('ContrastiveDEV pilot persistent quota')
        self.counts[kind] += 1
        self.journal.write(encode({'kind': kind, 'number': self.counts[kind], 'epoch': time.time()}) + '\n')
        self.journal.flush()
        if time.monotonic() - self.last_resource > 5:
            free = int(subprocess.check_output([str(Path(os.environ['WINDIR']) / 'System32/nvidia-smi.exe'),
                '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW).strip()) * 1024**2
            memory = psutil.Process().memory_info()
            row = {'RSS': memory.rss, 'private_commit': getattr(memory, 'private', None), 'GPU_free': free}
            self.resources.append(row)
            with (self.run / 'resources.jsonl').open('a', encoding='utf-8') as f:
                f.write(encode(row) + '\n')
            if free < 512 * 1024**2 or memory.rss > 2_000_000_000:
                raise MemoryError('2GB research RSS /512MiB discrete reserve')
            self.last_resource = time.monotonic()


def main():
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-contrastive-credit-dev-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    with (ROOT / 'local/semantic-contrastive-credit-dev-v1-execution.json').open('x', encoding='utf-8') as f:
        json.dump({'run': str(run)}, f)
    (ROOT / 'local/current-semantic-credit-dev.txt').write_text(str(run), encoding='utf-8')
    artifact = ROOT / 'artifacts' / run.name
    artifact.mkdir()
    feature_dir = Path((ROOT / 'local/current-semantic-features.txt').read_text(encoding='utf-8').strip())
    rows = json.loads((feature_dir / 'data.json').read_text(encoding='utf-8'))
    with np.load(feature_dir / 'features.npz', allow_pickle=False) as f:
        # Do not even load originalTEST feature rows into pilot input arrays.
        dev, train = f['dev'].copy(), f['train'].copy()
        indices = np.concatenate((train, dev))
        x = {int(i): f['x'][i].copy() for i in indices}
        y = {int(i): int(f['labels'][i]) for i in indices}
    source = Path((ROOT / 'local/current-semantic-local-ALIF.txt').read_text(encoding='utf-8').strip())
    stages = json.loads((source / 'schedules.json').read_text(encoding='utf-8'))['260113']
    assert all(rows[i]['split'] == 'TRAIN' for st in stages for k in ('current', 'replay_samples') for i in st[k])
    sources = ('scripts/run_semantic_contrastive_credit_dev.py', 'src/scarx/evaluation/semantic_credit_alif.py',
               'src/scarx/hybrid/learner.py', 'experiments/semantic-alif-contrastive-credit-dev-v1.md')
    (run / 'manifest.json').write_text(encode({'source_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sources': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sources}, 'feature_run': feature_dir.name,
        'schedule_source': source.name, 'seed': 260113, 'rates': [.05, .2], 'TEST_calls': 0, 'CPU_training': False,
        'max_updates': 1500, 'max_forwards': 12000, 'max_seconds': 180}), encoding='utf-8')
    for name in sources:
        (run / Path(name).name).write_bytes((ROOT / name).read_bytes())
    budget = Budget(run)
    trace = Store(ROOT / 'local/hybrid-v1')
    report = {'completed': False, 'failures': [], 'TEST_calls': 0, 'CPU_training': False,
              'quality_or_retention_confirmation': False, 'main_or_live_changed': False}
    curves, losses, arrays = {}, [], {}
    def predict(model, selected):
        return np.stack([model.model.host(model.forward(x[int(i)])[0]) for i in selected])
    try:
        a, b = CreditALIF(260113, 'hidden', budget=budget), CreditALIF(260113, 'hidden', budget=budget)
        for k in range(2):
            vector = np.eye(64, dtype=np.float32)[k]
            assert a.learn(vector, k) == b.learn(vector, k)
        payload = a.dumps({'replay': [0, 1], 'stage': 'P0'})
        b, controller = CreditALIF.loads(payload, budget)
        assert b.dumps(controller) == payload
        assert a.learn(np.eye(64, dtype=np.float32)[2], 2) == b.learn(np.eye(64, dtype=np.float32)[2], 2)
        assert a.dumps() == b.dumps()
        try:
            CreditALIF(260113, 'hidden', device='cpu')
        except ValueError:
            pass
        else:
            raise AssertionError('CPU fallback must be rejected')
        report['P0_GPU_clone_resume_exact'] = True
        report['P0_charged_counts'] = budget.counts.copy()
        frozen = CreditALIF(260113, 'frozen', budget=budget)
        frozen_p = predict(frozen, dev)
        arrays['frozen_DEV'] = frozen_p
        for name, arm, rate in (('hidden_eta05', 'hidden', .05), ('hidden_eta20', 'hidden', .2), ('readout_only', 'readout_only', .05)):
            model = CreditALIF(260113, arm, rate, budget)
            initial_rec = model.model.host(model.model.core.w_rec).copy()
            curve = []
            for stage in stages:
                stage_records = []
                for kind in ('current', 'replay_samples'):
                    for index in stage[kind]:
                        record = {'arm': name, 'stage': stage['stage'], 'sample_index': index, 'kind': kind,
                                  **model.learn(x[index], y[index])}
                        stage_records.append(record)
                        losses.append(record)
                p = predict(model, dev)
                arrays[f'{name}_DEV_stage{stage["stage"]:02d}'] = p
                predicted = p.argmax(axis=1)
                scores = [float(np.mean(predicted[[y[int(i)] == c for i in dev]] == c)) for c in range(6)]
                curve.append(scores)
                (artifact / f'{name}-stage{stage["stage"]:02d}.npz').write_bytes(model.dumps(stage['controller']))
                trace.event('work_trace.activity', {'title': 'GPU local credit: học từ lỗi phân loại thật',
                    'run_id': run.name, 'arm': name, 'stage': stage['stage'] + 1,
                    'loss': float(np.mean([r['loss'] for r in stage_records])), 'DEV_accuracy': float(np.mean(scores)),
                    'DEV_by_intent': scores, 'hidden_updates': model.hidden_updates, 'updates': model.updates,
                    'weight_delta_L2_sum': sum(r['weight_delta_L2'] for r in stage_records),
                    'active_last_tick': model.model.active_spikes(), 'device': 'CUDA0',
                    'CPU_training': False, 'TEST_calls': 0, 'resources': budget.resources[-1]}, 'orchestrator')
            curves[name] = curve
            restored, _ = CreditALIF.loads(model.dumps(), budget)
            np.testing.assert_array_equal(p, predict(restored, dev))
            np.testing.assert_array_equal(initial_rec, model.model.host(model.model.core.w_rec))
            np.testing.assert_array_equal(model.model.host(model.model.f_mean), np.zeros(128, dtype=np.float32))
            np.testing.assert_array_equal(model.model.host(model.model.f_var), np.ones(128, dtype=np.float32))
            if sum(p.stat().st_size for p in artifact.glob('*')) > 30_000_000:
                raise MemoryError('30MB artifacts bound')
        summary = {}
        for name, curve in curves.items():
            summary[name] = summarize_curve(curve)
        report.update(completed=True, verdict='GPU_DEV_SCREEN_COMPLETE_NOT_CONFIRMATION', summary=summary,
            strongest_raw_feature_DEV_control=.7916666666666666,
            pilot_any_candidate_gate=any(summary[n]['pilot_acquisition_and_retention_gate'] for n in ('hidden_eta05', 'hidden_eta20')),
            frozen_DEV_accuracy=float(np.mean(frozen_p.argmax(axis=1) == [y[int(i)] for i in dev])))
        if time.monotonic() - budget.started > 180:
            raise TimeoutError('Pilot totalwall budget including finalsummary')
    except Exception as exc:
        report.update(completed=False, verdict='GPU_CONTRASTIVE_DEV_STOPPED_NOT_PASS')
        report['failures'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
        np.savez(run / 'DEV-predictions.npz', **arrays)
        (run / 'losses.jsonl').write_text(''.join(encode(r) + '\n' for r in losses), encoding='utf-8')
        (run / 'curves.json').write_text(encode(curves), encoding='utf-8')
        report.update(charged_counts=budget.counts, elapsed_seconds=time.monotonic() - budget.started,
            peak_sampled_RSS=max((r['RSS'] for r in budget.resources), default=None),
            min_sampled_GPU_free=min((r['GPU_free'] for r in budget.resources), default=None), artifacts=str(artifact.relative_to(ROOT)))
        (run / 'result.json').write_text(encode(report), encoding='utf-8')
        trace.event('work_trace.activity', {'title': report['verdict'], 'run_id': run.name, **report}, 'orchestrator')
        budget.journal.close()
        trace.close()
    print(encode({'run': str(run), **report}), flush=True)
    if not report['completed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
